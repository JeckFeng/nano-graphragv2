"""Application layer adapters for running the top supervisor agent."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Optional

from agents.top_supervisor import create_top_supervisor
from config.settings import get_settings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command


class AgentRunner:
    """Abstract runner that streams agent output.

    Implementations must yield dict events with at least:
      - type: "token" | "final" | "error"
      - content / delta as applicable
    """

    async def astream(
        self,
        thread_id: str,
        user_content: str,
        enable_streaming: bool = False,
    ) -> AsyncIterator[dict]:
        """Stream agent events for a user message.

        Args:
            thread_id: Thread identifier.
            user_content: User message content.
            enable_streaming: Whether to enable streaming events.

        Yields:
            dict: Event payloads.
        """
        raise NotImplementedError


class TopSupervisorRunner(AgentRunner):
    """Runner that wraps the top supervisor agent."""

    def __init__(self, db_uri: Optional[str] = None) -> None:
        """Initialize the runner.

        Args:
            db_uri: Optional database URI for LangGraph memory.
        """
        settings = get_settings()
        self._db_uri = db_uri or settings.langgraph_memory_database_url

    async def astream(
        self,
        thread_id: str,
        user_content: str,
        enable_streaming: bool = False,
    ) -> AsyncIterator[dict]:
        """Stream agent output for a user message.

        Args:
            thread_id: Thread identifier.
            user_content: User message content.
            enable_streaming: Whether to enable streaming events.

        Yields:
            dict: Event payloads.
        """
        config = {"configurable": {"thread_id": thread_id}}
        async with AsyncPostgresSaver.from_conn_string(self._db_uri) as checkpointer:
            await checkpointer.setup()
            agent, _ = create_top_supervisor(checkpointer)

            if enable_streaming and hasattr(agent, "astream_events"):
                async for event in agent.astream_events(
                    {"messages": [{"role": "user", "content": user_content}]},
                    config=config,
                ):
                    mapped = self._map_event(event)
                    if mapped:
                        yield mapped
                return

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_content}]},
                config=config,
            )
            result = await self._auto_approve_interrupts(result, config, agent)
            yield {"type": "final", "content": result["messages"][-1].content}

    async def _auto_approve_interrupts(self, result: dict, config: dict, agent: Any) -> dict:
        """Automatically approve tool call interrupts.

        Args:
            result: Agent invocation result.
            config: Agent configuration containing thread_id.
            agent: Agent instance.

        Returns:
            dict: Final agent result after approvals.
        """
        while result.get("__interrupt__"):
            resume_map = self._build_resume_map(result["__interrupt__"])
            result = await agent.ainvoke(Command(resume=resume_map), config=config)
        return result

    def _build_resume_map(self, interrupts_list: list) -> dict:
        """Build a resume map that approves all tool calls.

        Args:
            interrupts_list: List of interrupt objects from the agent.

        Returns:
            dict: Resume map for Command.
        """
        resume_map: dict = {}
        for interrupt_obj in interrupts_list:
            interrupts = interrupt_obj.value
            action_requests = interrupts.get("action_requests", [])
            decisions = [{"type": "approve"} for _ in action_requests]
            resume_map[interrupt_obj.id] = {"decisions": decisions}
        return resume_map

    def _map_event(self, event: Any) -> Optional[dict]:
        """Map a streaming event to a standard payload.

        Args:
            event: Raw event from the agent streaming API.

        Returns:
            Optional[dict]: Mapped event payload or None.
        """
        if not isinstance(event, dict):
            return None
        event_type = event.get("event")
        if event_type is None:
            return None
        data = event.get("data") or {}
        chunk = data.get("chunk") or data.get("delta") or data.get("text")
        content = self._extract_chunk_content(chunk)
        if content:
            return {"type": "token", "delta": content}
        return None

    @staticmethod
    def _extract_chunk_content(chunk: Any) -> Optional[str]:
        """Extract text content from a streaming chunk.

        Args:
            chunk: Chunk object from streaming output.

        Returns:
            Optional[str]: Extracted text content.
        """
        if chunk is None:
            return None
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            return chunk.get("content")
        if hasattr(chunk, "content"):
            return getattr(chunk, "content")
        return None

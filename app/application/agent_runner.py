"""Application layer adapters for running the top supervisor agent."""

from __future__ import annotations

from collections.abc import AsyncIterator
import ast
import re
import hashlib
import json
import time
import uuid
from typing import Any, Dict, Optional

from agents.top_supervisor import create_top_supervisor
from app.services.approval_service import ApprovalService
from app.observability import log_event, set_run_id
from app.observability.trace_publisher import build_trace_event, trace_publish
from config.settings import get_settings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


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
        user_id: str,
        enable_streaming: bool = False,
    ) -> AsyncIterator[dict]:
        """Stream agent events for a user message.

        Args:
            thread_id: Thread identifier.
            user_content: User message content.
            user_id: External user identifier.
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
        user_id: str,
        enable_streaming: bool = False,
    ) -> AsyncIterator[dict]:
        """Stream agent output for a user message.

        Args:
            thread_id: Thread identifier.
            user_content: User message content.
            user_id: External user identifier.
            enable_streaming: Whether to enable streaming events.

        Yields:
            dict: Event payloads.
        """
        run_id = uuid.uuid4().hex
        set_run_id(run_id)
        start = time.perf_counter()
        ok = True
        log_event(
            "agent",
            {
                "source": "backend",
                "component": "agent",
                "event_type": "system",
                "event_name": "agent_run_start",
                "graph_name": "top_supervisor",
                "ok": True,
            },
        )
        await trace_publish(
            build_trace_event(
                trace_kind="agent_run_start",
                phase="start",
                payload={"user_message": user_content},
                component="agent",
            )
        )
        try:
            config = {"configurable": {"thread_id": thread_id}}
            async with AsyncPostgresSaver.from_conn_string(self._db_uri) as checkpointer:
                await checkpointer.setup()
                agent, _ = create_top_supervisor(checkpointer)
                if enable_streaming and hasattr(agent, "astream_events"):
                    buffered_tokens: list[str] = []
                    interrupts_list: Optional[list] = None
                    async for event in agent.astream_events(
                        {"messages": [{"role": "user", "content": user_content}]},
                        config=config,
                    ):
                        interrupts_list = self._extract_interrupts(event)
                        if interrupts_list:
                            break
                        if isinstance(event, dict) and event.get("event") == "on_tool_error":
                            error_detail = (event.get("data") or {}).get("error")
                            error_message = str(error_detail) if error_detail is not None else "Tool execution failed"
                            yield {
                                "type": "error",
                                "code": "TOOL_ERROR",
                                "message": error_message,
                            }
                            return
                        mapped = self._map_event(event)
                        if mapped:
                            delta = mapped.get("delta", "")
                            if isinstance(delta, str) and delta:
                                buffered_tokens.append(delta)
                            yield mapped

                    if interrupts_list:
                        # 保存审批前的 partial 输出到消息表
                        if buffered_tokens:
                            partial_content = "".join(buffered_tokens)
                            await self._save_partial_message(thread_id, partial_content)

                        await self._emit_hitl_trace(interrupts_list)
                        approval_service = ApprovalService()
                        approval = await approval_service.create_approval(
                            user_id=user_id,
                            thread_id=thread_id,
                            interrupts_list=interrupts_list,
                        )
                        yield self._build_approval_event(approval)
                        return

                    if buffered_tokens:
                        yield {"type": "final", "content": "".join(buffered_tokens)}
                        return

                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_content}]},
                    config=config,
                )
                if result.get("__interrupt__"):
                    await self._emit_hitl_trace(result["__interrupt__"])
                    approval_service = ApprovalService()
                    approval = await approval_service.create_approval(
                        user_id=user_id,
                        thread_id=thread_id,
                        interrupts_list=result["__interrupt__"],
                    )
                    yield self._build_approval_event(approval)
                    return

                content = result["messages"][-1].content
                yield {"type": "final", "content": content}
        except Exception as exc:
            ok = False
            log_event(
                "agent",
                {
                    "source": "backend",
                    "component": "agent",
                    "event_type": "error",
                    "event_name": "agent_run_error",
                    "graph_name": "top_supervisor",
                    "ok": False,
                    "error": repr(exc),
                },
            )
            raise
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_event(
                "agent",
                {
                    "source": "backend",
                    "component": "agent",
                    "event_type": "system",
                    "event_name": "agent_run_end",
                    "graph_name": "top_supervisor",
                    "ok": ok,
                    "latency_ms": latency_ms,
                },
            )
            set_run_id(None)

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
        metadata = event.get("metadata") or {}
        checkpoint_ns = metadata.get("langgraph_checkpoint_ns") or ""
        if "tools:" in checkpoint_ns:
            return None
        data = event.get("data") or {}
        chunk = data.get("chunk") or data.get("delta") or data.get("text")
        content = self._extract_chunk_content(chunk)
        if content:
            return {"type": "token", "delta": content}
        return None

    @staticmethod
    def _build_approval_event(approval: Any) -> dict:
        """Build an approval-required event payload.

        Args:
            approval: Approval record instance.

        Returns:
            dict: Approval-required event payload.
        """
        interrupts_payload = [
            {
                "interrupt_id": interrupt.interrupt_id,
                "action_requests": interrupt.action_requests,
                "review_configs": interrupt.review_configs,
            }
            for interrupt in approval.interrupts
        ]
        return {
            "type": "approval_required",
            "approval_id": approval.approval_id,
            "status": approval.status,
            "interrupts": interrupts_payload,
        }

    @staticmethod
    def _extract_interrupts(event: Any) -> Optional[list]:
        """Extract interrupts from a streaming event if present.

        Args:
            event: Streaming event payload.

        Returns:
            Optional[list]: Interrupts list if found.
        """
        if not isinstance(event, dict):
            return None
        data = event.get("data") or {}
        if "__interrupt__" in data:
            return data.get("__interrupt__")
        output = data.get("output")
        if isinstance(output, dict) and "__interrupt__" in output:
            return output.get("__interrupt__")
        if event.get("event") == "on_tool_error":
            error_detail = data.get("error")
            if error_detail is None:
                return None
            if hasattr(error_detail, "id") and hasattr(error_detail, "value"):
                interrupt_id = str(getattr(error_detail, "id"))
                value = getattr(error_detail, "value")
                if isinstance(value, dict):
                    return [ParsedInterrupt(interrupt_id=interrupt_id, value=value)]
            return TopSupervisorRunner._parse_interrupts_from_error(str(error_detail))
        return None

    @staticmethod
    def _parse_interrupts_from_error(error_detail: str) -> Optional[list]:
        """Parse interrupts from a tool error string payload.

        Args:
            error_detail: Error string potentially containing Interrupt(value=...).

        Returns:
            Optional[list]: List of parsed interrupts if present.
        """
        marker = "Interrupt(value="
        start_idx = error_detail.find(marker)
        if start_idx == -1:
            return None

        brace_start = error_detail.find("{", start_idx + len(marker))
        if brace_start == -1:
            return None

        payload = TopSupervisorRunner._extract_braced_payload(error_detail, brace_start)
        if not payload:
            return None

        try:
            value = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            return None

        if not isinstance(value, dict):
            return None

        tail = error_detail[brace_start + len(payload) :]
        patterns = [
            r"id=UUID\\('([^']+)'\\)",
            r"id=UUID\\(\"([^\"]+)\"\\)",
            r"id='([^']+)'",
            r'id=\"([^\"]+)\"',
        ]
        interrupt_id = None
        for pattern in patterns:
            match = re.search(pattern, tail)
            if match:
                interrupt_id = match.group(1)
                break
        if not interrupt_id:
            interrupt_id = "unknown"
        return [ParsedInterrupt(interrupt_id=interrupt_id, value=value)]

    @staticmethod
    def _extract_braced_payload(text: str, start: int) -> Optional[str]:
        """Extract a brace-balanced payload starting at index.

        Args:
            text: Input text containing a dict literal.
            start: Start index where '{' is located.

        Returns:
            Optional[str]: Extracted payload or None.
        """
        if start < 0 or start >= len(text) or text[start] != "{":
            return None

        depth = 0
        in_string = False
        quote_char = ""
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == quote_char:
                    in_string = False
                    quote_char = ""
                continue
            if char in ("'", '"'):
                in_string = True
                quote_char = char
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
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

    @staticmethod
    def _summarize_value(value: Any) -> Dict[str, Any]:
        """Summarize a Python value into keys/bytes/hash.

        Args:
            value: Value to summarize.

        Returns:
            Dict[str, Any]: Summary dictionary.
        """
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except TypeError:
            encoded = repr(value).encode("utf-8")
        keys = None
        if isinstance(value, dict):
            keys = list(value.keys())
        elif isinstance(value, list):
            keys = ["<array>"]
        return {
            "keys": keys,
            "bytes": len(encoded),
            "hash": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        }

    async def _emit_hitl_trace(self, interrupts_list: list) -> None:
        """Emit HITL interrupt trace events.

        Args:
            interrupts_list: Interrupt objects from LangGraph.
        """
        actions: list[Dict[str, Any]] = []
        for interrupt_obj in interrupts_list:
            interrupts = getattr(interrupt_obj, "value", None)
            if not isinstance(interrupts, dict):
                continue
            action_requests = interrupts.get("action_requests", [])
            review_configs = interrupts.get("review_configs", [])
            review_map = {
                cfg.get("action_name"): cfg for cfg in review_configs if isinstance(cfg, dict)
            }
            for action in action_requests:
                if not isinstance(action, dict):
                    continue
                tool_name = action.get("name")
                args = action.get("args")
                review_config = review_map.get(tool_name)
                actions.append(
                    {
                        "tool_name": tool_name,
                        "args_summary": self._summarize_value(args),
                        "review_config": review_config,
                    }
                )

        payload = {"pending_actions": len(actions), "actions": actions}
        await trace_publish(
            build_trace_event(
                trace_kind="hitl_interrupt",
                phase="pending",
                payload=payload,
                component="agent",
            )
        )

    async def _save_partial_message(self, thread_id: str, content: str) -> None:
        """Save partial assistant message before approval interrupt.

        Args:
            thread_id: Thread identifier.
            content: Partial message content.
        """
        if not content.strip():
            return

        from app.infra.db import ASYNC_SESSION_FACTORY
        from app.observability import get_ctx
        from app.services.message_service import MessageService

        async with ASYNC_SESSION_FACTORY() as session:
            message_service = MessageService(session)
            ctx = get_ctx()
            if ctx.get("message_id"):
                return
            run_id = ctx.get("run_id")
            await message_service.append_message(
                thread_id=uuid.UUID(thread_id),
                role="assistant",
                content=content,
                run_id=run_id,
                tool_payload={"partial": True, "reason": "approval_interrupt"},
            )


class ParsedInterrupt:
    """Minimal interrupt wrapper extracted from tool error payloads."""

    def __init__(self, interrupt_id: str, value: dict) -> None:
        """Initialize the parsed interrupt wrapper.

        Args:
            interrupt_id: Interrupt identifier.
            value: Interrupt payload value.
        """
        self.id = interrupt_id
        self.value = value

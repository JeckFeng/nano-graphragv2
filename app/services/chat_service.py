"""Chat service handling user messages and agent responses."""

from __future__ import annotations

import uuid
from typing import AsyncIterator, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.application.agent_runner import AgentRunner
from app.observability import set_message_id
from app.services.message_service import MessageService
from app.services.thread_service import ThreadService
from app.services.user_service import UserService


class ChatService:
    """Service for chat streaming orchestration.

    Invariants:
        - The session must remain valid during the service lifecycle.
        - AgentRunner must be configured and reachable.
    """

    def __init__(self, session: AsyncSession, agent_runner: AgentRunner) -> None:
        """Initialize the chat service.

        Args:
            session: SQLAlchemy async session.
            agent_runner: Runner that streams agent output.
        """
        self._session = session
        self._agent_runner = agent_runner
        self._user_service = UserService(session)
        self._thread_service = ThreadService(session)
        self._message_service = MessageService(session)

    async def stream_chat(
        self,
        external_user_id: str,
        thread_id: uuid.UUID,
        user_content: str,
        metadata: Optional[dict] = None,
        enable_streaming: bool = True,
    ) -> AsyncIterator[dict]:
        """Stream chat responses for a user message.

        Args:
            external_user_id: External user identifier.
            thread_id: Thread identifier.
            user_content: User message content.
            metadata: Optional metadata attached to the message.
            enable_streaming: Whether to enable streaming events.

        Yields:
            dict: Event payloads for the WebSocket layer.

        Raises:
            PermissionError: If the user does not own the thread.
        """
        _ = metadata
        user = await self._user_service.get_user_by_external_id(external_user_id)
        if not user:
            raise PermissionError("User not found")
        if not await self._thread_service.verify_thread_owner(user.id, thread_id):
            raise PermissionError("Thread not found")

        await self._message_service.append_message(thread_id, "user", user_content)

        assistant_message_id: Optional[int] = None
        buffered_tokens: list[str] = []
        async for event in self._agent_runner.astream(
            thread_id=str(thread_id),
            user_content=user_content,
            user_id=external_user_id,
            enable_streaming=enable_streaming,
        ):
            event_type = event.get("type")
            if event_type == "token":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    buffered_tokens.append(delta)
                    content = "".join(buffered_tokens)
                    if assistant_message_id is None:
                        assistant_message = await self._message_service.append_message(
                            thread_id,
                            "assistant",
                            content,
                        )
                        assistant_message_id = assistant_message.id
                    else:
                        await self._message_service.update_message_content(
                            assistant_message_id,
                            content,
                        )
                    event["message_id"] = assistant_message_id
                    set_message_id(assistant_message_id)
                yield event
                continue

            if event_type == "final":
                content = event.get("content") or "".join(buffered_tokens)
                if assistant_message_id is None:
                    assistant_message = await self._message_service.append_message(
                        thread_id,
                        "assistant",
                        content,
                    )
                    assistant_message_id = assistant_message.id
                else:
                    await self._message_service.update_message_content(
                        assistant_message_id,
                        content,
                    )
                event["message_id"] = assistant_message_id
                event["content"] = content
                set_message_id(assistant_message_id)
                yield event
                continue

            yield event

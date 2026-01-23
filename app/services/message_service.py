"""Message service providing access to conversation messages."""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.models import Message


class MessageService:
    """Service for message data access.

    Invariants:
        - The session must remain valid during the service lifecycle.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service with an async session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    async def append_message(
        self,
        thread_id: uuid.UUID,
        role: str,
        content: str,
        run_id: Optional[str] = None,
        tool_payload: Optional[dict] = None,
    ) -> Message:
        """Append a new message to a thread.

        Args:
            thread_id: Thread identifier.
            role: Message role (user/assistant/tool).
            content: Message content.
            tool_payload: Optional tool payload.

        Returns:
            Message: Newly created message.
        """
        message = Message(
            thread_id=thread_id,
            run_id=run_id,
            role=role,
            content=content,
            tool_payload=tool_payload,
        )
        self._session.add(message)
        await self._session.commit()
        await self._session.refresh(message)
        return message

    async def list_messages(
        self,
        thread_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Message]:
        """List messages for a thread.

        Args:
            thread_id: Thread identifier.
            limit: Maximum number of messages to return.
            offset: Offset for pagination.

        Returns:
            List[Message]: Messages ordered by creation time ascending.
        """
        result = await self._session.execute(
            select(Message)
            .where(Message.thread_id == thread_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def update_message_content(self, message_id: int, content: str) -> None:
        """Update message content for a given message.

        Args:
            message_id: Message identifier.
            content: Updated message content.
        """
        result = await self._session.execute(
            select(Message).where(Message.id == message_id)
        )
        message = result.scalar_one_or_none()
        if message is None:
            return
        message.content = content
        await self._session.commit()

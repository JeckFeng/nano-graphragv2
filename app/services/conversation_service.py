"""Conversation service orchestrating user and thread creation."""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.models import Message, Thread
from app.services.message_service import MessageService
from app.services.thread_service import ThreadService
from app.services.user_service import UserService


class ConversationService:
    """Service for conversation orchestration.

    Invariants:
        - The session must remain valid during the service lifecycle.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service with an async session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session
        self._user_service = UserService(session)
        self._thread_service = ThreadService(session)
        self._message_service = MessageService(session)

    async def create_conversation(self, external_user_id: str, title: Optional[str]) -> Thread:
        """Create a conversation thread for a user.

        Args:
            external_user_id: External user identifier.
            title: Optional thread title.

        Returns:
            Thread: Newly created thread.
        """
        user = await self._user_service.get_or_create_user(external_user_id)
        thread = await self._thread_service.create_thread(user.id, title)
        return thread

    async def list_conversations(
        self,
        external_user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Thread]:
        """List conversations for a user.

        Args:
            external_user_id: External user identifier.
            limit: Maximum number of threads to return.
            offset: Offset for pagination.

        Returns:
            List[Thread]: List of threads owned by the user.
        """
        user = await self._user_service.get_user_by_external_id(external_user_id)
        if not user:
            return []
        return await self._thread_service.list_threads(user.id, limit=limit, offset=offset)

    async def list_messages(
        self,
        external_user_id: str,
        thread_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> Optional[List[Message]]:
        """List messages for a thread owned by the user.

        Args:
            external_user_id: External user identifier.
            thread_id: Thread identifier.
            limit: Maximum number of messages to return.
            offset: Offset for pagination.

        Returns:
            Optional[List[Message]]: Messages if thread ownership is verified.
        """
        user = await self._user_service.get_user_by_external_id(external_user_id)
        if not user:
            return None
        if not await self._thread_service.verify_thread_owner(user.id, thread_id):
            return None
        return await self._message_service.list_messages(thread_id, limit=limit, offset=offset)

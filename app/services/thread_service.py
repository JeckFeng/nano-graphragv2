"""Thread service providing access to conversation threads."""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.models import Thread


class ThreadService:
    """Service for thread data access.

    Invariants:
        - The session must remain valid during the service lifecycle.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service with an async session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    async def create_thread(self, user_id: int, title: Optional[str] = None) -> Thread:
        """Create a new thread for a user.

        Args:
            user_id: Internal user identifier.
            title: Optional thread title.

        Returns:
            Thread: Newly created thread.
        """
        thread = Thread(user_id=user_id, title=title)
        self._session.add(thread)
        await self._session.commit()
        await self._session.refresh(thread)
        return thread

    async def list_threads(self, user_id: int, limit: int = 20, offset: int = 0) -> List[Thread]:
        """List threads for a user.

        Args:
            user_id: Internal user identifier.
            limit: Maximum number of threads to return.
            offset: Offset for pagination.

        Returns:
            List[Thread]: List of threads ordered by creation time descending.
        """
        result = await self._session.execute(
            select(Thread)
            .where(Thread.user_id == user_id)
            .order_by(Thread.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_thread(self, thread_id: uuid.UUID) -> Optional[Thread]:
        """Fetch a thread by its identifier.

        Args:
            thread_id: Thread identifier.

        Returns:
            Optional[Thread]: Thread if found, otherwise None.
        """
        result = await self._session.execute(select(Thread).where(Thread.id == thread_id))
        return result.scalar_one_or_none()

    async def verify_thread_owner(self, user_id: int, thread_id: uuid.UUID) -> bool:
        """Verify that a thread belongs to a user.

        Args:
            user_id: Internal user identifier.
            thread_id: Thread identifier.

        Returns:
            bool: True if the thread belongs to the user, otherwise False.
        """
        result = await self._session.execute(
            select(Thread.id).where(Thread.id == thread_id, Thread.user_id == user_id)
        )
        return result.scalar_one_or_none() is not None

    async def delete_thread(self, thread_id: uuid.UUID) -> bool:
        """Delete a thread and all related data (cascade).

        Args:
            thread_id: Thread identifier.

        Returns:
            bool: True if deleted, False if not found.
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            return False
        await self._session.delete(thread)
        await self._session.commit()
        return True

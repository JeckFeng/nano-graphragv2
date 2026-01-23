"""Trace service providing access to trace events."""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.models import Thread, TraceEvent
from app.services.thread_service import ThreadService
from app.services.user_service import UserService


class TraceService:
    """Service for trace event data access.

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

    async def list_by_thread(
        self,
        *,
        user_id: str,
        thread_id: uuid.UUID,
        limit: int,
        offset: int,
        run_id: Optional[str] = None,
        message_id: Optional[int] = None,
        trace_kind: Optional[str] = None,
        phase: Optional[str] = None,
        order: str = "asc",
    ) -> Optional[List[TraceEvent]]:
        """List trace events for a thread owned by a user.

        Args:
            user_id: External user identifier.
            thread_id: Thread identifier.
            limit: Maximum number of records to return.
            offset: Pagination offset.
            run_id: Optional run filter.
            message_id: Optional message filter.
            trace_kind: Optional trace kind filter.
            phase: Optional phase filter.
            order: Ordering direction ("asc" or "desc").

        Returns:
            Optional[List[TraceEvent]]: Trace events if ownership verified.
        """
        user = await self._user_service.get_user_by_external_id(user_id)
        if not user:
            return None
        if not await self._thread_service.verify_thread_owner(user.id, thread_id):
            return None

        stmt = select(TraceEvent).where(TraceEvent.thread_id == thread_id)
        if run_id:
            stmt = stmt.where(TraceEvent.run_id == run_id)
        if message_id is not None:
            stmt = stmt.where(TraceEvent.message_id == message_id)
        if trace_kind:
            stmt = stmt.where(TraceEvent.trace_kind == trace_kind)
        if phase:
            stmt = stmt.where(TraceEvent.phase == phase)

        if order == "desc":
            stmt = stmt.order_by(TraceEvent.event_time.desc(), TraceEvent.id.desc())
        else:
            stmt = stmt.order_by(TraceEvent.event_time.asc(), TraceEvent.id.asc())

        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_run(
        self,
        *,
        user_id: str,
        run_id: str,
        limit: int,
        offset: int,
        order: str = "asc",
    ) -> Optional[List[TraceEvent]]:
        """List trace events for a run owned by a user.

        Args:
            user_id: External user identifier.
            run_id: Run identifier.
            limit: Maximum number of records to return.
            offset: Pagination offset.
            order: Ordering direction ("asc" or "desc").

        Returns:
            Optional[List[TraceEvent]]: Trace events if user exists.
        """
        user = await self._user_service.get_user_by_external_id(user_id)
        if not user:
            return None

        stmt = (
            select(TraceEvent)
            .join(Thread, TraceEvent.thread_id == Thread.id)
            .where(TraceEvent.run_id == run_id, Thread.user_id == user.id)
        )
        if order == "desc":
            stmt = stmt.order_by(TraceEvent.event_time.desc(), TraceEvent.id.desc())
        else:
            stmt = stmt.order_by(TraceEvent.event_time.asc(), TraceEvent.id.asc())

        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

"""Artifact service providing access to stored artifacts."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.models import Artifact, Thread, User


class ArtifactService:
    """Service for artifact data access.

    Invariants:
        - The session must remain valid during the service lifecycle.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service with an async session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    async def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Fetch an artifact by its identifier.

        Args:
            artifact_id: Artifact identifier.

        Returns:
            Optional[Artifact]: Artifact if found, otherwise None.
        """
        result = await self._session.execute(
            select(Artifact).where(Artifact.id == artifact_id)
        )
        return result.scalar_one_or_none()

    async def get_artifact_for_user(
        self, external_user_id: str, artifact_id: str
    ) -> Optional[Artifact]:
        """Fetch an artifact after verifying user ownership.

        Args:
            external_user_id: External user identifier.
            artifact_id: Artifact identifier.

        Returns:
            Optional[Artifact]: Artifact if user owns the thread, otherwise None.
        """
        result = await self._session.execute(
            select(Artifact)
            .join(Thread, Artifact.thread_id == Thread.id)
            .join(User, Thread.user_id == User.id)
            .where(Artifact.id == artifact_id, User.external_user_id == external_user_id)
        )
        return result.scalar_one_or_none()

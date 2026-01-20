"""User service providing access to user records."""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.models import User

LOGGER = logging.getLogger(__name__)


class UserService:
    """Service for user data access.

    Invariants:
        - The session must remain valid during the service lifecycle.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service with an async session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    async def get_user_by_external_id(self, external_user_id: str) -> Optional[User]:
        """Fetch a user by external identifier.

        Args:
            external_user_id: External user identifier.

        Returns:
            Optional[User]: User if found, otherwise None.
        """
        result = await self._session.execute(
            select(User).where(User.external_user_id == external_user_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create_user(self, external_user_id: str) -> User:
        """Get an existing user or create a new one.

        Args:
            external_user_id: External user identifier.

        Returns:
            User: Existing or newly created user.
        """
        existing_user = await self.get_user_by_external_id(external_user_id)
        if existing_user:
            return existing_user

        user = User(external_user_id=external_user_id)
        self._session.add(user)

        try:
            await self._session.commit()
        except IntegrityError:
            LOGGER.warning("User creation race detected for %s", external_user_id)
            await self._session.rollback()
            existing_user = await self.get_user_by_external_id(external_user_id)
            if existing_user:
                return existing_user
            raise

        await self._session.refresh(user)
        return user

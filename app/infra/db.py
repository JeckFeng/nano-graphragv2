"""Database session and engine setup for the MVP API service."""

from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from config.settings import get_settings


def build_async_database_url(database_url: str) -> str:
    """Build an async database URL with asyncpg driver.

    Args:
        database_url: Base database URL from settings.

    Returns:
        str: Database URL configured for asyncpg.
    """
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return database_url


def create_async_engine_from_settings() -> AsyncEngine:
    """Create an async SQLAlchemy engine using application settings.

    Returns:
        AsyncEngine: Configured SQLAlchemy async engine.
    """
    settings = get_settings()
    async_url = build_async_database_url(settings.database_url)
    max_overflow = max(settings.db_pool_max_size - settings.db_pool_min_size, 0)
    return create_async_engine(
        async_url,
        pool_size=settings.db_pool_min_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
    )


ASYNC_ENGINE = create_async_engine_from_settings()
ASYNC_SESSION_FACTORY = async_sessionmaker(ASYNC_ENGINE, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """Provide a scoped async session for request handling.

    Yields:
        AsyncSession: SQLAlchemy async session instance.
    """
    async with ASYNC_SESSION_FACTORY() as session:
        yield session

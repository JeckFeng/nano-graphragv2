"""Add run_id column to agent_app.messages if missing."""

from __future__ import annotations

import argparse
import asyncio
from urllib.parse import quote_plus

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from config.settings import get_settings


def _build_db_url(db_name: str) -> str:
    """Build async database URL for a specific database.

    Args:
        db_name: Target database name.

    Returns:
        str: Async database URL.
    """
    settings = get_settings()
    password = settings.db_password.get_secret_value()
    encoded_password = quote_plus(password) if password else ""
    return (
        "postgresql+asyncpg://"
        f"{settings.db_user}:{encoded_password}@{settings.db_host}:{settings.db_port}/{db_name}"
    )


async def _run(db_name: str, schema: str) -> None:
    """Add run_id column to messages table.

    Args:
        db_name: Target database name.
        schema: Target schema name.
    """
    engine = create_async_engine(_build_db_url(db_name), pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.execute(
            text(f"ALTER TABLE {schema}.messages ADD COLUMN IF NOT EXISTS run_id VARCHAR(64)")
        )
        await conn.execute(
            text(f"CREATE INDEX IF NOT EXISTS messages_run_id_idx ON {schema}.messages (run_id)")
        )
    await engine.dispose()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Add run_id column to messages table.")
    parser.add_argument("--db-name", default="mydb", help="Target database name (default: mydb).")
    parser.add_argument("--schema", default="agent_app", help="Target schema (default: agent_app).")
    args = parser.parse_args()
    asyncio.run(_run(args.db_name, args.schema))
    print(f"messages.run_id added in {args.db_name}.{args.schema}")


if __name__ == "__main__":
    main()

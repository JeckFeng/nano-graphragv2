"""Initialize the trace_events table in mydb.agent_app."""

from __future__ import annotations

import argparse
import asyncio
from urllib.parse import quote_plus

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from config.settings import get_settings


TRACE_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS {schema}.trace_events (
  id BIGSERIAL PRIMARY KEY,
  event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  request_id VARCHAR(64),
  thread_id UUID NOT NULL,
  run_id VARCHAR(64),
  message_id BIGINT,
  user_id VARCHAR(128),
  event_type VARCHAR(32) NOT NULL DEFAULT 'trace',
  event_name VARCHAR(64) NOT NULL DEFAULT 'trace_event',
  trace_kind VARCHAR(32) NOT NULL,
  phase VARCHAR(16) NOT NULL,
  source VARCHAR(32) NOT NULL DEFAULT 'backend',
  component VARCHAR(32) NOT NULL DEFAULT 'agent',
  tool_name VARCHAR(128),
  subagent_type VARCHAR(128),
  ok BOOLEAN,
  latency_ms INTEGER,
  error TEXT,
  payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
  seq INTEGER,
  CONSTRAINT trace_events_kind_chk
    CHECK (trace_kind IN ('todo_update','tool_span','subagent_dispatch','hitl_interrupt','hitl_resume','agent_run_start','supervisor_route')),
  CONSTRAINT trace_events_phase_chk
    CHECK (phase IN ('start','end','update','error','pending','resume'))
);
"""


TRACE_EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS trace_events_thread_time_idx "
    "ON {schema}.trace_events (thread_id, event_time);",
    "CREATE INDEX IF NOT EXISTS trace_events_run_seq_idx "
    "ON {schema}.trace_events (run_id, seq);",
    "CREATE INDEX IF NOT EXISTS trace_events_user_time_idx "
    "ON {schema}.trace_events (user_id, event_time);",
]

TRACE_EVENTS_KIND_CONSTRAINT = (
    "ALTER TABLE {schema}.trace_events DROP CONSTRAINT IF EXISTS trace_events_kind_chk;",
    "ALTER TABLE {schema}.trace_events ADD CONSTRAINT trace_events_kind_chk "
    "CHECK (trace_kind IN ('todo_update','tool_span','subagent_dispatch','hitl_interrupt',"
    "'hitl_resume','agent_run_start','supervisor_route'));",
)


def _build_db_url(db_name: str) -> str:
    """Build async database URL for a specific database.

    Args:
        db_name: Target database name.

    Returns:
        str: Async database URL for SQLAlchemy.
    """
    settings = get_settings()
    password = settings.db_password.get_secret_value()
    encoded_password = quote_plus(password) if password else ""
    return (
        "postgresql+asyncpg://"
        f"{settings.db_user}:{encoded_password}@{settings.db_host}:{settings.db_port}/{db_name}"
    )


async def _run(db_name: str, schema: str) -> None:
    """Create schema and trace_events table.

    Args:
        db_name: Target database name.
        schema: Target schema name.
    """
    engine = create_async_engine(_build_db_url(db_name), pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        await conn.execute(text(TRACE_EVENTS_DDL.format(schema=schema)))
        for stmt in TRACE_EVENTS_KIND_CONSTRAINT:
            await conn.execute(text(stmt.format(schema=schema)))
        for stmt in TRACE_EVENTS_INDEXES:
            await conn.execute(text(stmt.format(schema=schema)))
    await engine.dispose()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Initialize trace_events table.")
    parser.add_argument("--db-name", default="mydb", help="Target database name (default: mydb).")
    parser.add_argument("--schema", default="agent_app", help="Target schema (default: agent_app).")
    args = parser.parse_args()
    asyncio.run(_run(args.db_name, args.schema))
    print(f"trace_events initialized in {args.db_name}.{args.schema}")


if __name__ == "__main__":
    main()

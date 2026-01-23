"""Trace event publisher utilities.

This module emits structured trace events for agent/tool observability. It can
optionally persist trace events and push them to WebSocket clients in real time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from app.infra.db import get_session
from app.infra.models import TraceEvent
from app.observability.config import get_obs_config
from app.observability.context import get_ctx
from app.observability.logger import log_event
from app.observability.summarize import summarize_payload
from app.services.ws_manager import ws_manager

logger = logging.getLogger(__name__)


def build_trace_event(
    trace_kind: str,
    phase: str,
    payload: Optional[dict] = None,
    *,
    component: str = "agent",
    tool_name: Optional[str] = None,
    subagent_type: Optional[str] = None,
    ok: Optional[bool] = None,
    latency_ms: Optional[int] = None,
    error: Optional[str] = None,
    seq: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a trace event payload merged with context.

    Args:
        trace_kind: Trace event kind.
        phase: Trace event phase.
        payload: Optional structured payload.
        component: Component name for the event.
        tool_name: Optional tool name.
        subagent_type: Optional subagent type name.
        ok: Optional success flag.
        latency_ms: Optional latency in milliseconds.
        error: Optional error message.
        seq: Optional sequence number.

    Returns:
        Dict[str, Any]: Trace event payload.
    """
    event: Dict[str, Any] = {
        **get_ctx(),
        "source": "backend",
        "component": component,
        "event_type": "trace",
        "event_name": "trace_event",
        "trace_kind": trace_kind,
        "phase": phase,
        "payload": payload or {},
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if tool_name:
        event["tool_name"] = tool_name
    if subagent_type:
        event["subagent_type"] = subagent_type
    if ok is not None:
        event["ok"] = ok
    if latency_ms is not None:
        event["latency_ms"] = latency_ms
    if error:
        event["error"] = error
    if seq is not None:
        event["seq"] = seq
    return event


async def trace_publish(
    event: Dict[str, Any],
    *,
    push_ws: bool = True,
    persist_db: bool = True,
) -> Optional[int]:
    """Publish a trace event to logs, DB, and WebSocket.

    Args:
        event: Trace event payload.
        push_ws: Whether to push the event to the active WebSocket.
        persist_db: Whether to persist the event into the database.

    Returns:
        Optional[int]: Persisted trace event id if available.
    """
    config = get_obs_config()
    if not config.enabled_trace:
        return None

    trace_event_id: Optional[int] = None
    if persist_db:
        try:
            trace_event_id = await _persist_trace_event(event)
        except Exception as exc:  # pragma: no cover - best-effort persistence
            logger.error("持久化 trace 事件失败: %s", exc, exc_info=True)
            trace_event_id = None
        if trace_event_id is not None:
            event["trace_event_id"] = trace_event_id

    log_event("agent", event)

    if push_ws:
        thread_id = event.get("thread_id")
        if thread_id:
            sent = await ws_manager.send_to_thread(str(thread_id), event)
            if sent:
                summary = summarize_payload(event)
                log_event(
                    "ws",
                    {
                        "source": "backend",
                        "component": "ws",
                        "event_type": "exchange",
                        "event_name": "ws_trace_out",
                        "ws_event_type": "trace",
                        "payload_keys": summary.get("keys"),
                        "payload_bytes": summary.get("bytes"),
                    },
                )

    return trace_event_id


async def _persist_trace_event(event: Dict[str, Any]) -> Optional[int]:
    """Persist a trace event to database.

    Args:
        event: Trace event payload.

    Returns:
        Optional[int]: Trace event id if persisted.
    """
    trace_kind = event.get("trace_kind")
    phase = event.get("phase")
    thread_id = event.get("thread_id")
    if not trace_kind or not phase or not thread_id:
        logger.warning("Trace event missing required fields; skipped persistence.")
        return None

    try:
        thread_uuid = UUID(str(thread_id))
    except ValueError:
        logger.warning("Trace event has invalid thread_id: %s", thread_id)
        return None

    record = TraceEvent(
        event_time=datetime.now(timezone.utc),
        request_id=event.get("request_id"),
        thread_id=thread_uuid,
        run_id=event.get("run_id"),
        message_id=event.get("message_id"),
        user_id=event.get("user_id"),
        event_type=event.get("event_type", "trace"),
        event_name=event.get("event_name", "trace_event"),
        trace_kind=str(trace_kind),
        phase=str(phase),
        source=event.get("source", "backend"),
        component=event.get("component", "agent"),
        tool_name=event.get("tool_name"),
        subagent_type=event.get("subagent_type"),
        ok=event.get("ok"),
        latency_ms=event.get("latency_ms"),
        error=event.get("error"),
        payload=event.get("payload") or {},
        seq=event.get("seq"),
    )

    async for session in get_session():
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record.id

    return None

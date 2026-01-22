"""Context variables for observability correlation."""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Dict, Optional
from urllib.parse import parse_qs


_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_thread_id: ContextVar[Optional[str]] = ContextVar("thread_id", default=None)
_message_id: ContextVar[Optional[str]] = ContextVar("message_id", default=None)
_run_id: ContextVar[Optional[str]] = ContextVar("run_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_conn_id: ContextVar[Optional[str]] = ContextVar("conn_id", default=None)


def _parse_query_params(scope: dict) -> Dict[str, str]:
    """Parse query params from ASGI scope.

    Args:
        scope: ASGI scope.

    Returns:
        Dict[str, str]: Flattened query params.
    """
    raw = scope.get("query_string", b"")
    try:
        parsed = parse_qs(raw.decode("utf-8"))
    except Exception:
        return {}
    return {key: values[0] for key, values in parsed.items() if values}


def _extract_thread_id(scope: dict) -> Optional[str]:
    """Extract thread_id from path if present."""
    path = scope.get("path", "") or ""
    parts = [p for p in path.split("/") if p]
    for idx, part in enumerate(parts):
        if part == "conversations" and idx + 1 < len(parts):
            candidate = parts[idx + 1]
            try:
                uuid.UUID(candidate)
            except ValueError:
                return None
            return candidate
    return None


def set_ctx_http(scope: dict, request_id: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Set context variables for an HTTP request.

    Args:
        scope: ASGI scope.
        request_id: Optional pre-generated request id.

    Returns:
        Dict[str, Optional[str]]: Context snapshot.
    """
    params = _parse_query_params(scope)
    user_id = params.get("user_id")
    thread_id = params.get("thread_id") or _extract_thread_id(scope)
    req_id = request_id or uuid.uuid4().hex

    _request_id.set(req_id)
    _thread_id.set(thread_id)
    _user_id.set(user_id)
    _conn_id.set(None)
    _message_id.set(None)
    _run_id.set(None)
    return get_ctx()


def set_ctx_ws(
    *,
    request_id: Optional[str],
    user_id: Optional[str],
    thread_id: Optional[str],
    conn_id: Optional[str],
) -> Dict[str, Optional[str]]:
    """Set context variables for a WebSocket connection."""
    req_id = request_id or uuid.uuid4().hex
    _request_id.set(req_id)
    _thread_id.set(thread_id)
    _user_id.set(user_id)
    _conn_id.set(conn_id)
    _message_id.set(None)
    _run_id.set(None)
    return get_ctx()


def set_run_id(run_id: Optional[str]) -> None:
    """Set run_id in context."""
    _run_id.set(run_id)


def set_message_id(message_id: Optional[str]) -> None:
    """Set message_id in context."""
    _message_id.set(message_id)


def set_user_id(user_id: Optional[str]) -> None:
    """Set user_id in context."""
    _user_id.set(user_id)


def set_thread_id(thread_id: Optional[str]) -> None:
    """Set thread_id in context."""
    _thread_id.set(thread_id)


def clear_ctx() -> None:
    """Clear context variables."""
    _request_id.set(None)
    _thread_id.set(None)
    _message_id.set(None)
    _run_id.set(None)
    _user_id.set(None)
    _conn_id.set(None)


def get_ctx() -> Dict[str, Optional[str]]:
    """Get current context snapshot."""
    return {
        "request_id": _request_id.get(),
        "thread_id": _thread_id.get(),
        "message_id": _message_id.get(),
        "run_id": _run_id.get(),
        "user_id": _user_id.get(),
        "conn_id": _conn_id.get(),
    }

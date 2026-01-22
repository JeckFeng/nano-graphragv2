"""ASGI middleware for HTTP observability."""

from __future__ import annotations

import json
import random
import time
from typing import Optional
from uuid import UUID

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability.config import get_obs_config
from app.observability.context import clear_ctx, get_ctx, set_ctx_http, set_thread_id, set_user_id
from app.observability.logger import log_event
from app.observability.summarize import summarize_json_bytes, redact_json_bytes


def _safe_load_json(payload: bytes) -> Optional[object]:
    """Parse JSON payload safely."""
    if not payload:
        return None
    data = payload.strip()
    if not data:
        return None
    if data[0] not in (ord("{"), ord("[")):
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _extract_ids(payload: object) -> tuple[Optional[str], Optional[str]]:
    """Extract user_id/thread_id from JSON payload."""
    if isinstance(payload, dict):
        user_id = payload.get("user_id")
        thread_id = payload.get("thread_id")
        return (
            str(user_id) if user_id is not None else None,
            str(thread_id) if thread_id is not None else None,
        )
    return None, None


def _is_valid_uuid(value: Optional[str]) -> bool:
    """Validate UUID string."""
    if not value:
        return False
    try:
        UUID(str(value))
    except (ValueError, TypeError):
        return False
    return True


class ObservabilityHTTPMiddleware:
    """HTTP middleware capturing request/response summaries.

    Invariants:
        - Does not consume request body without replaying.
        - Does not alter response payloads.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware."""
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        config = get_obs_config()
        if not config.enabled_api:
            await self._app(scope, receive, send)
            return

        start = time.perf_counter()
        set_ctx_http(scope)
        ctx = get_ctx()
        req_body_buf = bytearray()
        req_total_bytes = 0
        resp_body_buf = bytearray()
        resp_total_bytes = 0
        status_code: Optional[int] = None
        sampled = random.random() < config.sample_rate_fullbody if config.sample_rate_fullbody > 0 else False
        is_debug = False
        if ctx.get("thread_id") and ctx["thread_id"] in config.debug_thread_ids:
            is_debug = True
        if ctx.get("user_id") and ctx["user_id"] in config.debug_user_ids:
            is_debug = True
        capture_req = config.capture_req_body or sampled or is_debug
        capture_resp = config.capture_resp_body or sampled or is_debug

        log_event(
            "app",
            {
                "source": "backend",
                "component": "api",
                "event_type": "exchange",
                "event_name": "http_request_start",
                "method": scope.get("method"),
                "path": scope.get("path"),
            },
        )

        async def receive_wrapped() -> Message:
            nonlocal req_total_bytes, req_body_buf
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body") or b""
                req_total_bytes += len(body)
                if len(req_body_buf) < config.body_max_bytes:
                    remain = config.body_max_bytes - len(req_body_buf)
                    req_body_buf.extend(body[:remain])
            return message

        async def send_wrapped(message: Message) -> None:
            nonlocal resp_total_bytes, resp_body_buf, status_code
            if message.get("type") == "http.response.start":
                status_code = message.get("status")
            elif message.get("type") == "http.response.body":
                body = message.get("body") or b""
                resp_total_bytes += len(body)
                if len(resp_body_buf) < config.body_max_bytes:
                    remain = config.body_max_bytes - len(resp_body_buf)
                    resp_body_buf.extend(body[:remain])
            await send(message)

        try:
            await self._app(scope, receive_wrapped, send_wrapped)
            latency_ms = int((time.perf_counter() - start) * 1000)

            req_payload = _safe_load_json(bytes(req_body_buf))
            user_id, thread_id = _extract_ids(req_payload)
            ctx = get_ctx()
            if user_id and not ctx.get("user_id"):
                set_user_id(user_id)
            if _is_valid_uuid(thread_id) and not ctx.get("thread_id"):
                set_thread_id(thread_id)

            resp_payload = _safe_load_json(bytes(resp_body_buf))
            _, resp_thread_id = _extract_ids(resp_payload)
            if _is_valid_uuid(resp_thread_id) and not get_ctx().get("thread_id"):
                set_thread_id(resp_thread_id)

            req_summary = summarize_json_bytes(bytes(req_body_buf), config.redact_keys)
            resp_summary = summarize_json_bytes(bytes(resp_body_buf), config.redact_keys)
            req_body = None
            resp_body = None
            if capture_req:
                req_body = redact_json_bytes(bytes(req_body_buf), config.redact_keys, config.body_max_bytes)
            if capture_resp:
                resp_body = redact_json_bytes(bytes(resp_body_buf), config.redact_keys, config.body_max_bytes)

            log_event(
                "app",
                {
                    "source": "backend",
                    "component": "api",
                    "event_type": "exchange",
                    "event_name": "http_request_end",
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status_code,
                    "latency_ms": latency_ms,
                    "req_bytes": req_total_bytes,
                    "resp_bytes": resp_total_bytes,
                    "req_keys": req_summary.get("keys"),
                    "resp_keys": resp_summary.get("keys"),
                    "req_hash": req_summary.get("hash"),
                    "resp_hash": resp_summary.get("hash"),
                    "req_body": req_body,
                    "resp_body": resp_body,
                },
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            req_payload = _safe_load_json(bytes(req_body_buf))
            user_id, thread_id = _extract_ids(req_payload)
            ctx = get_ctx()
            if user_id and not ctx.get("user_id"):
                set_user_id(user_id)
            if _is_valid_uuid(thread_id) and not ctx.get("thread_id"):
                set_thread_id(thread_id)
            req_body = None
            if capture_req or req_total_bytes <= config.body_max_bytes:
                req_body = redact_json_bytes(bytes(req_body_buf), config.redact_keys, config.body_max_bytes)
            log_event(
                "app",
                {
                    "source": "backend",
                    "component": "api",
                    "event_type": "error",
                    "event_name": "http_error",
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "latency_ms": latency_ms,
                    "error": repr(exc),
                    "req_body": req_body,
                },
            )
            raise
        finally:
            clear_ctx()

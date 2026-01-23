"""Trace middleware for deepagents tool calls.

This middleware intercepts tool calls to emit structured trace events.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.observability.context import get_ctx, set_run_id, set_thread_id, set_user_id
from app.observability.trace_publisher import build_trace_event, trace_publish


def _top_keys(value: Any) -> Optional[list[str]]:
    """Return top-level keys for dict/list payloads.

    Args:
        value: Payload value to inspect.

    Returns:
        Optional[list[str]]: Top-level keys or None.
    """
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, list):
        return ["<array>"]
    return None


def _summarize_value(value: Any) -> Dict[str, Any]:
    """Summarize a Python value into keys/bytes/hash.

    Args:
        value: Any JSON-serializable value.

    Returns:
        Dict[str, Any]: Summary payload.
    """
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except TypeError:
        encoded = repr(value).encode("utf-8")
    return {
        "keys": _top_keys(value),
        "bytes": len(encoded),
        "hash": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    }


def _extract_result_payload(result: ToolMessage | Command) -> Any:
    """Extract result payload from ToolMessage or Command.

    Args:
        result: Tool execution result.

    Returns:
        Any: Extracted payload value.
    """
    if isinstance(result, ToolMessage):
        return result.content
    if isinstance(result, Command):
        return result.update or {}
    return result


def _ensure_ctx_from_runtime(request: ToolCallRequest) -> None:
    """Fill missing context fields from the tool runtime when available.

    Args:
        request: Tool call request carrying runtime metadata.
    """
    ctx = get_ctx()
    runtime = request.runtime
    if runtime is None:
        return

    config = getattr(runtime, "config", None) or {}
    configurable = {}
    if isinstance(config, dict):
        configurable = config.get("configurable", {}) if isinstance(config.get("configurable"), dict) else {}

    if ctx.get("thread_id") is None:
        thread_id = configurable.get("thread_id")
        if thread_id:
            set_thread_id(str(thread_id))

    if ctx.get("run_id") is None:
        run_id = config.get("run_id") if isinstance(config, dict) else None
        if run_id:
            set_run_id(str(run_id))

    if ctx.get("user_id") is None:
        context = getattr(runtime, "context", None)
        if isinstance(context, dict) and context.get("user_id"):
            set_user_id(str(context.get("user_id")))


class TraceMiddleware(AgentMiddleware):
    """Capture tool traces for deepagents.

    Invariants:
        - Does not modify tool inputs/outputs.
        - Emits only structured trace events.
    """

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Intercept async tool execution and emit trace events.

        Args:
            request: Tool call request.
            handler: Tool execution handler.

        Returns:
            ToolMessage | Command: Tool execution result.
        """
        _ensure_ctx_from_runtime(request)

        tool_call = request.tool_call or {}
        tool_name = str(tool_call.get("name", ""))
        args = tool_call.get("args", {})
        args_summary = _summarize_value(args)

        await trace_publish(
            build_trace_event(
                trace_kind="tool_span",
                phase="start",
                payload={"args_summary": args_summary},
                component="tool",
                tool_name=tool_name,
            )
        )

        subagent_type: Optional[str] = None
        task_summary: Optional[str] = None
        if tool_name == "task" and isinstance(args, dict):
            subagent_type = str(args.get("subagent_type") or "")
            task_summary = str(args.get("description") or "")
            await trace_publish(
                build_trace_event(
                    trace_kind="subagent_dispatch",
                    phase="start",
                    payload={"task_summary": task_summary},
                    component="agent",
                    subagent_type=subagent_type or None,
                )
            )

        start = time.perf_counter()
        try:
            result = await handler(request)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            await trace_publish(
                build_trace_event(
                    trace_kind="tool_span",
                    phase="error",
                    payload={"args_summary": args_summary},
                    component="tool",
                    tool_name=tool_name,
                    ok=False,
                    latency_ms=latency_ms,
                    error=repr(exc),
                )
            )
            if tool_name == "task" and subagent_type:
                await trace_publish(
                    build_trace_event(
                        trace_kind="subagent_dispatch",
                        phase="error",
                        payload={"task_summary": task_summary or ""},
                        component="agent",
                        subagent_type=subagent_type,
                        ok=False,
                        latency_ms=latency_ms,
                        error=repr(exc),
                    )
                )
            raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        result_payload = _extract_result_payload(result)
        result_summary = _summarize_value(result_payload)

        await trace_publish(
            build_trace_event(
                trace_kind="tool_span",
                phase="end",
                payload={
                    "args_summary": args_summary,
                    "result_summary": result_summary,
                },
                component="tool",
                tool_name=tool_name,
                ok=True,
                latency_ms=latency_ms,
            )
        )

        if tool_name == "write_todos" and isinstance(args, dict):
            todos = args.get("todos")
            if todos is not None:
                await trace_publish(
                    build_trace_event(
                        trace_kind="todo_update",
                        phase="update",
                        payload={"todos": todos},
                        component="agent",
                    )
                )

        if tool_name == "task" and subagent_type:
            await trace_publish(
                build_trace_event(
                    trace_kind="subagent_dispatch",
                    phase="end",
                    payload={"task_summary": task_summary or ""},
                    component="agent",
                    subagent_type=subagent_type,
                    ok=True,
                    latency_ms=latency_ms,
                )
            )

        return result

"""Probe deepagents astream_events to check tool outputs in events.

This script runs the top supervisor with a short prompt and inspects
LangGraph/DeepAgents stream events to see whether tool outputs are present.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.top_supervisor import create_top_supervisor
from config.settings import get_settings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def _is_tool_event(event: Dict[str, Any]) -> bool:
    """Return True if event looks like a tool-related event.

    Args:
        event: Raw event dict.

    Returns:
        bool: True if tool-related.
    """
    name = str(event.get("event", "")).lower()
    return "tool" in name or name in {"on_tool_start", "on_tool_end", "on_tool_error"}


def _extract_output(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    """Extract likely tool output field from event data.

    Args:
        data: Event data dict.

    Returns:
        tuple: (field_name, value) or (None, None).
    """
    for key in ("output", "result", "response", "tool_result", "tool_output", "return"):
        if key in data:
            return key, data.get(key)
    return None, None


def _summarize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact summary for an event.

    Args:
        event: Raw event dict.

    Returns:
        dict: Summary information.
    """
    data = event.get("data") or {}
    tool_name = data.get("name") or data.get("tool") or data.get("tool_name")
    output_key, output_value = _extract_output(data if isinstance(data, dict) else {})
    summary = {
        "event": event.get("event"),
        "tool_name": tool_name,
        "data_keys": list(data.keys()) if isinstance(data, dict) else [],
        "output_key": output_key,
        "output_type": type(output_value).__name__ if output_key else None,
    }
    return summary


async def _probe_events(
    message: str,
    thread_id: str,
    max_events: int,
    only_tools: bool,
) -> Iterable[Dict[str, Any]]:
    """Stream events from the top supervisor and collect them.

    Args:
        message: User prompt.
        thread_id: Thread identifier.
        max_events: Maximum number of events to capture.
        only_tools: Whether to keep only tool-related events.

    Returns:
        Iterable[dict]: Captured events.
    """
    settings = get_settings()
    events: list[Dict[str, Any]] = []
    async with AsyncPostgresSaver.from_conn_string(
        settings.langgraph_memory_database_url
    ) as checkpointer:
        await checkpointer.setup()
        agent, _ = create_top_supervisor(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        async for event in agent.astream_events(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        ):
            if not isinstance(event, dict):
                continue
            if only_tools and not _is_tool_event(event):
                continue
            events.append(event)
            if len(events) >= max_events:
                break
    return events


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed args.
    """
    parser = argparse.ArgumentParser(description="Probe deepagents event payloads.")
    parser.add_argument(
        "--message",
        default="请列出当前工作目录下的前 5 个文件名。",
        help="User prompt to trigger tool calls.",
    )
    parser.add_argument(
        "--thread-id",
        default="",
        help="Thread ID (uuid). If empty, a random one is generated.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=200,
        help="Maximum number of events to capture.",
    )
    parser.add_argument(
        "--only-tools",
        action="store_true",
        help="Only keep tool-related events.",
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Print raw output content (may be very large).",
    )
    parser.add_argument(
        "--dump-json",
        default="",
        help="Optional path to dump raw events as JSON.",
    )
    return parser.parse_args()


def _safe_json_dump(data: Any) -> str:
    """Serialize data to JSON safely.

    Args:
        data: Any payload.

    Returns:
        str: JSON string.
    """
    return json.dumps(data, ensure_ascii=False, default=str, indent=2)


def main() -> None:
    """Entry point for probing deepagents event payloads."""
    args = _parse_args()
    thread_id = args.thread_id or str(uuid.uuid4())
    events = asyncio.run(
        _probe_events(
            message=args.message,
            thread_id=thread_id,
            max_events=args.max_events,
            only_tools=args.only_tools,
        )
    )

    tool_events = [e for e in events if _is_tool_event(e)]
    summaries = [_summarize_event(e) for e in tool_events]

    print(f"thread_id={thread_id}")
    print(f"captured_events={len(events)} tool_events={len(tool_events)}")
    print("tool_event_summaries:")
    print(_safe_json_dump(summaries))

    if args.print_output:
        for event in tool_events:
            data = event.get("data") or {}
            output_key, output_value = _extract_output(
                data if isinstance(data, dict) else {}
            )
            if output_key:
                print("\n==== tool output ====")
                print(f"event={event.get('event')} key={output_key}")
                print(_safe_json_dump(output_value))

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            f.write(_safe_json_dump(events))
        print(f"raw events dumped to: {args.dump_json}")


if __name__ == "__main__":
    main()

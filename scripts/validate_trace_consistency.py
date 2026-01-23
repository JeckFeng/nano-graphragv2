"""Validate consistency between trace logs and database records."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

from sqlalchemy import select

from app.infra.db import ASYNC_SESSION_FACTORY
from app.infra.models import TraceEvent


def _parse_date(value: str) -> date:
    """Parse a YYYYMMDD date string.

    Args:
        value: Date string in YYYYMMDD format.

    Returns:
        date: Parsed date.
    """
    return datetime.strptime(value, "%Y%m%d").date()


def _default_log_path(target_date: date) -> Path:
    """Build default log path for a given date.

    Args:
        target_date: Target UTC date.

    Returns:
        Path: Log file path.
    """
    return Path("Logs") / "agent" / f"agent_{target_date.strftime('%Y%m%d')}.jsonl"


def _derive_date_from_path(path: Path) -> Optional[date]:
    """Try to parse date suffix from a log file name.

    Args:
        path: Log file path.

    Returns:
        Optional[date]: Parsed date or None.
    """
    stem = path.stem
    if "_" not in stem:
        return None
    maybe_date = stem.split("_")[-1]
    if len(maybe_date) != 8 or not maybe_date.isdigit():
        return None
    try:
        return _parse_date(maybe_date)
    except ValueError:
        return None


def _iter_log_entries(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a log file.

    Args:
        path: Log file path.

    Yields:
        dict: Parsed log entry.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _collect_log_ids(path: Path) -> Tuple[set[int], int, int]:
    """Collect trace_event_id values from a log file.

    Args:
        path: Log file path.

    Returns:
        Tuple[set[int], int, int]: (trace_ids, missing_id_count, total_trace_events)
    """
    trace_ids: set[int] = set()
    missing_id_count = 0
    total = 0

    for entry in _iter_log_entries(path):
        if entry.get("event_type") != "trace" and "trace_kind" not in entry:
            continue
        total += 1
        trace_id = entry.get("trace_event_id")
        if trace_id is None:
            missing_id_count += 1
            continue
        try:
            trace_ids.add(int(trace_id))
        except (TypeError, ValueError):
            missing_id_count += 1

    return trace_ids, missing_id_count, total


async def _load_db_ids(start: Optional[datetime], end: Optional[datetime]) -> set[int]:
    """Load trace event ids from database within a date range.

    Args:
        start: Range start (UTC).
        end: Range end (UTC).

    Returns:
        set[int]: Trace event id set.
    """
    async with ASYNC_SESSION_FACTORY() as session:
        stmt = select(TraceEvent.id)
        if start and end:
            stmt = stmt.where(TraceEvent.event_time >= start, TraceEvent.event_time < end)
        result = await session.execute(stmt)
        return {row[0] for row in result.all()}


def _format_sample(items: Iterable[int], limit: int = 20) -> str:
    """Format a sample list for printing.

    Args:
        items: Iterable of ids.
        limit: Max items to show.

    Returns:
        str: Formatted string.
    """
    sample = list(items)[:limit]
    if not sample:
        return "[]"
    return f"[{', '.join(str(item) for item in sample)}]"


async def _run(args: argparse.Namespace) -> int:
    log_path = Path(args.log_file) if args.log_file else _default_log_path(args.date)
    if not log_path.exists():
        print(f"日志文件不存在: {log_path}")
        return 1

    target_date = args.date
    if args.log_file and not args.date:
        derived = _derive_date_from_path(log_path)
        if derived:
            target_date = derived

    db_start = None
    db_end = None
    if target_date:
        db_start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
        db_end = db_start + timedelta(days=1)

    log_ids, missing_ids, total = _collect_log_ids(log_path)
    db_ids = await _load_db_ids(db_start, db_end)

    log_only = log_ids - db_ids
    db_only = db_ids - log_ids

    print("Trace 日志一致性校验结果")
    print(f"- 日志文件: {log_path}")
    if db_start and db_end:
        print(f"- DB 范围: {db_start.isoformat()} ~ {db_end.isoformat()}")
    print(f"- 日志 trace 事件数: {total}")
    print(f"- 日志含 trace_event_id 数: {len(log_ids)}")
    print(f"- 日志缺失 trace_event_id 数: {missing_ids}")
    print(f"- DB trace 事件数: {len(db_ids)}")
    print(f"- log_only: {len(log_only)} 示例: {_format_sample(sorted(log_only))}")
    print(f"- db_only: {len(db_only)} 示例: {_format_sample(sorted(db_only))}")

    return 0


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Validate trace log vs database consistency.")
    parser.add_argument(
        "--date",
        type=_parse_date,
        default=date.today(),
        help="日志日期 (YYYYMMDD)，默认当天 UTC 日期。",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="指定日志文件路径，设置后可覆盖默认路径。",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
可观测日志清理脚本

只清理 Logs/app、Logs/ws、Logs/agent、Logs/tool_logs 下的日志文件。
注意：
    - 只删除文件，不删除任何目录
    - 支持 .log/.json/.jsonl
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SUBDIRS = ["app", "ws", "agent", "tool_logs"]
SUPPORTED_PATTERNS = ["*.log", "*.json", "*.jsonl"]


def parse_size(size_str: str) -> int:
    """解析大小字符串为字节数。"""
    size_str = size_str.upper().strip()
    if size_str.endswith("B"):
        size_str = size_str[:-1]
    multipliers = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            number = float(size_str[:-1])
            return int(number * multiplier)
    return int(size_str)


def find_log_files(
    log_dir: Path,
    days: int | None = None,
    min_size: int | None = None,
) -> List[Tuple[Path, dict]]:
    """查找指定目录下的日志文件。"""
    log_files: List[Tuple[Path, dict]] = []
    if not log_dir.exists():
        return log_files

    cutoff_time = None
    if days is not None:
        cutoff_time = datetime.now() - timedelta(days=days)

    for pattern in SUPPORTED_PATTERNS:
        for file_path in log_dir.rglob(pattern):
            if not file_path.is_file():
                continue
            try:
                stat = file_path.stat()
                file_info = {
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                }
                if cutoff_time and file_info["modified"] > cutoff_time:
                    continue
                if min_size and file_info["size"] < min_size:
                    continue
                log_files.append((file_path, file_info))
            except Exception as exc:
                print(f"警告: 无法读取文件 {file_path}: {exc}", file=sys.stderr)
    return log_files


def format_size(size: int) -> str:
    """格式化大小。"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description="清理 Logs 可观测日志文件")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不删除")
    parser.add_argument("--days", type=int, default=None, help="只删除指定天数之前的文件")
    parser.add_argument("--size", type=str, default=None, help="只删除大于指定大小的文件（如 100MB）")
    args = parser.parse_args()

    min_size = parse_size(args.size) if args.size else None
    base_dir = PROJECT_ROOT / "Logs"

    total_files = 0
    total_bytes = 0

    for subdir in DEFAULT_SUBDIRS:
        target_dir = base_dir / subdir
        files = find_log_files(target_dir, days=args.days, min_size=min_size)
        if not files:
            continue
        print(f"\n目录: {target_dir} (匹配 {len(files)} 个文件)")
        for file_path, info in files:
            total_files += 1
            total_bytes += info["size"]
            print(f"- {file_path} ({format_size(info['size'])}, {info['modified']})")
            if not args.dry_run:
                try:
                    file_path.unlink()
                except Exception as exc:
                    print(f"删除失败: {file_path}: {exc}", file=sys.stderr)

    print(f"\n合计: {total_files} 个文件, {format_size(total_bytes)}")
    if args.dry_run:
        print("（dry-run 模式，未删除）")


if __name__ == "__main__":
    main()

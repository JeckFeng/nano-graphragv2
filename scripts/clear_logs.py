#!/usr/bin/env python3
"""
日志清理脚本

用于清理 Logs 目录下的所有 .log 和 .json 文件。

注意：
    - 只删除文件，不删除任何目录（包括文件夹、子文件夹、孙子文件夹等）
    - 即使目录变为空目录，也不会被删除

使用方法:
    # 预览要删除的文件（不实际删除）
    python -m scripts.clear_logs --dry-run
    
    # 实际删除所有日志文件
    python -m scripts.clear_logs
    
    # 删除指定目录下的日志文件
    python -m scripts.clear_logs --dir Logs/top_supervisor_logs
    
    # 删除指定天数之前的日志文件
    python -m scripts.clear_logs --days 7
    
    # 删除指定大小以上的日志文件
    python -m scripts.clear_logs --size 100MB
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_size(size_str: str) -> int:
    """
    解析大小字符串（如 "100MB", "1GB"）为字节数
    
    Args:
        size_str: 大小字符串，支持 B, KB, MB, GB
        
    Returns:
        int: 字节数
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith("B"):
        size_str = size_str[:-1]
    
    multipliers = {
        "K": 1024,
        "M": 1024 ** 2,
        "G": 1024 ** 3,
    }
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            number = float(size_str[:-1])
            return int(number * multiplier)
    
    # 默认单位为字节
    return int(size_str)


def find_log_files(
    log_dir: Path,
    days: int = None,
    min_size: int = None,
) -> List[Tuple[Path, dict]]:
    """
    查找所有日志文件
    
    Args:
        log_dir: 日志目录
        days: 只查找指定天数之前的文件（None 表示不限制）
        min_size: 只查找大于指定大小的文件（None 表示不限制）
        
    Returns:
        List[Tuple[Path, dict]]: 文件路径和文件信息的列表
    """
    log_files = []
    
    if not log_dir.exists():
        return log_files
    
    # 计算截止时间
    cutoff_time = None
    if days is not None:
        cutoff_time = datetime.now() - timedelta(days=days)
    
    # 遍历所有 .log 和 .json 文件（只处理文件，不处理目录）
    for file_path in log_dir.rglob("*.log"):
        # 确保是文件，不是目录
        if not file_path.is_file():
            continue
            
        try:
            stat = file_path.stat()
            file_info = {
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
            }
            
            # 检查时间条件
            if cutoff_time and file_info["modified"] > cutoff_time:
                continue
            
            # 检查大小条件
            if min_size and file_info["size"] < min_size:
                continue
            
            log_files.append((file_path, file_info))
        except Exception as e:
            print(f"警告: 无法读取文件 {file_path}: {e}", file=sys.stderr)
    
    for file_path in log_dir.rglob("*.json"):
        # 确保是文件，不是目录
        if not file_path.is_file():
            continue
            
        try:
            stat = file_path.stat()
            file_info = {
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
            }
            
            # 检查时间条件
            if cutoff_time and file_info["modified"] > cutoff_time:
                continue
            
            # 检查大小条件
            if min_size and file_info["size"] < min_size:
                continue
            
            log_files.append((file_path, file_info))
        except Exception as e:
            print(f"警告: 无法读取文件 {file_path}: {e}", file=sys.stderr)
    
    return log_files


def format_size(size: int) -> str:
    """
    格式化文件大小
    
    Args:
        size: 字节数
        
    Returns:
        str: 格式化后的大小字符串
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="清理 Logs 目录下的日志文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览要删除的文件
  python -m scripts.clear_logs --dry-run
  
  # 删除所有日志文件
  python -m scripts.clear_logs
  
  # 删除 7 天前的日志文件
  python -m scripts.clear_logs --days 7
  
  # 删除大于 100MB 的日志文件
  python -m scripts.clear_logs --size 100MB
  
  # 删除指定目录下的日志文件
  python -m scripts.clear_logs --dir Logs/top_supervisor_logs
        """,
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        default="Logs",
        help="日志目录路径（默认: Logs）",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="只删除指定天数之前的文件（例如: --days 7 表示删除 7 天前的文件）",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="只删除大于指定大小的文件（例如: --size 100MB）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，只显示要删除的文件，不实际删除",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细信息",
    )
    
    args = parser.parse_args()
    
    # 解析日志目录
    log_dir = Path(project_root) / args.dir
    if not log_dir.exists():
        print(f"错误: 日志目录不存在: {log_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 解析大小限制
    min_size = None
    if args.size:
        try:
            min_size = parse_size(args.size)
        except ValueError as e:
            print(f"错误: 无效的大小格式: {args.size}", file=sys.stderr)
            print(f"      支持格式: 100B, 100KB, 100MB, 1GB", file=sys.stderr)
            sys.exit(1)
    
    # 查找日志文件
    print(f"正在扫描日志目录: {log_dir}")
    log_files = find_log_files(log_dir, days=args.days, min_size=min_size)
    
    if not log_files:
        print("未找到符合条件的日志文件")
        return
    
    # 统计信息
    total_size = sum(info["size"] for _, info in log_files)
    total_count = len(log_files)
    
    print(f"\n找到 {total_count} 个日志文件，总大小: {format_size(total_size)}")
    
    if args.dry_run:
        print("\n[预览模式] 以下文件将被删除:\n")
    else:
        print("\n以下文件将被删除:\n")
    
    # 显示文件列表
    for file_path, file_info in sorted(log_files, key=lambda x: x[0]):
        relative_path = file_path.relative_to(project_root)
        size_str = format_size(file_info["size"])
        time_str = file_info["modified"].strftime("%Y-%m-%d %H:%M:%S")
        
        if args.verbose:
            print(f"  {relative_path}")
            print(f"    大小: {size_str}, 修改时间: {time_str}")
        else:
            print(f"  {relative_path} ({size_str}, {time_str})")
    
    if args.dry_run:
        print(f"\n[预览模式] 共 {total_count} 个文件，总大小: {format_size(total_size)}")
        print("使用 --dry-run=false 或移除 --dry-run 参数来实际删除文件")
        return
    
    # 确认删除
    print(f"\n确认删除 {total_count} 个文件，总大小: {format_size(total_size)}? (yes/no): ", end="")
    try:
        confirmation = input().strip().lower()
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(0)
    
    if confirmation not in ("yes", "y"):
        print("操作已取消")
        return
    
    # 删除文件（只删除文件，不删除任何目录）
    print("\n正在删除文件...")
    deleted_count = 0
    deleted_size = 0
    errors = []
    
    for file_path, file_info in log_files:
        # 双重检查：确保是文件，不是目录
        if not file_path.is_file():
            error_msg = f"跳过（不是文件）: {file_path.relative_to(project_root)}"
            errors.append(error_msg)
            if args.verbose:
                print(f"  警告: {error_msg}", file=sys.stderr)
            continue
            
        try:
            file_path.unlink()
            deleted_count += 1
            deleted_size += file_info["size"]
            
            if args.verbose:
                print(f"  已删除: {file_path.relative_to(project_root)}")
        except Exception as e:
            error_msg = f"删除失败: {file_path.relative_to(project_root)} - {e}"
            errors.append(error_msg)
            print(f"  错误: {error_msg}", file=sys.stderr)
    
    # 显示结果
    print(f"\n完成!")
    print(f"  已删除: {deleted_count} 个文件")
    print(f"  释放空间: {format_size(deleted_size)}")
    
    if errors:
        print(f"\n  警告: {len(errors)} 个文件处理失败或跳过")
        for error in errors:
            print(f"    {error}", file=sys.stderr)
    
    print("\n注意: 只删除文件，不删除任何目录（包括空目录）")


if __name__ == "__main__":
    main()


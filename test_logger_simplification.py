"""测试简化后的日志配置"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 测试 logger_config
print("=" * 60)
print("测试 logger_config.py")
print("=" * 60)

from core.logger_config import get_logger_config

config = get_logger_config('tool')
print(f"✓ 成功加载 tool 配置")
print(f"  log_dir: {config.get('log_dir')}")
print(f"  level: {config.get('level')}")
print(f"  enabled: {config.get('enabled')}")
print(f"  fields: {config.get('fields')}")

# 测试 logging_context 已删除
print("\n" + "=" * 60)
print("验证 logging_context.py 已删除")
print("=" * 60)

try:
    from core import logging_context
    print("✗ logging_context.py 仍然存在")
except ImportError:
    print("✓ logging_context.py 已成功删除")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

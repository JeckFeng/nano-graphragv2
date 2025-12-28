"""
测试工具配置系统

验证从 tool.yaml 读取配置的功能，包括自动检测工具所属 Agent
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.tool_config import get_tool_config


def test_tool_config():
    """测试工具配置读取"""
    print("=" * 60)
    print("测试工具配置系统")
    print("=" * 60)
    
    config = get_tool_config()
    
    # 测试用例 1: 显式指定 agent_name
    print("\n【测试 1】显式指定 agent_name")
    print("-" * 60)
    
    test_cases_explicit = [
        # (tool_name, agent_name, expected_timeout, expected_max_calls)
        ("add_numbers", "addition-agent", 20.0, 5),
        ("subtract_numbers", "subtraction-agent", 25.0, 4),
        ("multiply_numbers", "multiplication-agent", 35.0, 6),
        ("divide_numbers", "division-agent", 30.0, 5),
        ("unknown_tool", "addition-agent", 30.0, 3),  # Agent 默认配置
        ("unknown_tool", "unknown-agent", 60.0, 5),  # 全局默认配置
    ]
    
    for tool_name, agent_name, expected_timeout, expected_max_calls in test_cases_explicit:
        timeout = config.get_tool_timeout(tool_name, agent_name)
        max_calls = config.get_tool_max_calls(tool_name, agent_name)
        
        timeout_match = "✓" if timeout == expected_timeout else "✗"
        max_calls_match = "✓" if max_calls == expected_max_calls else "✗"
        
        print(f"\n工具: {tool_name}, Agent: {agent_name}")
        print(f"  timeout: {timeout}s (期望: {expected_timeout}s) {timeout_match}")
        print(f"  max_calls: {max_calls} (期望: {expected_max_calls}) {max_calls_match}")
    
    # 测试用例 2: 自动检测 agent_name
    print("\n\n【测试 2】自动检测 agent_name（agent_name=None）")
    print("-" * 60)
    
    test_cases_auto = [
        # (tool_name, expected_timeout, expected_max_calls, expected_agent)
        ("add_numbers", 20.0, 5, "addition-agent"),
        ("subtract_numbers", 25.0, 4, "subtraction-agent"),
        ("multiply_numbers", 35.0, 6, "multiplication-agent"),
        ("divide_numbers", 30.0, 5, "division-agent"),
        ("unknown_tool", 60.0, 5, None),  # 未知工具，使用全局默认
    ]
    
    for tool_name, expected_timeout, expected_max_calls, expected_agent in test_cases_auto:
        timeout = config.get_tool_timeout(tool_name, None)  # agent_name=None
        max_calls = config.get_tool_max_calls(tool_name, None)
        detected_agent = config._find_agent_for_tool(tool_name)
        
        timeout_match = "✓" if timeout == expected_timeout else "✗"
        max_calls_match = "✓" if max_calls == expected_max_calls else "✗"
        agent_match = "✓" if detected_agent == expected_agent else "✗"
        
        print(f"\n工具: {tool_name}")
        print(f"  检测到的 Agent: {detected_agent} (期望: {expected_agent}) {agent_match}")
        print(f"  timeout: {timeout}s (期望: {expected_timeout}s) {timeout_match}")
        print(f"  max_calls: {max_calls} (期望: {expected_max_calls}) {max_calls_match}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_tool_config()

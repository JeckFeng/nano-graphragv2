"""
Map Team Agent - 高德地图服务团队代理

本模块实现一个团队代理，它不直接使用工具，而是通过调用子代理完成路径规划任务。
Map Team Agent 管理一个子代理：map_worker。

技术栈:
    - LangGraph 1.0
    - Deep Agents (CompiledSubAgent 功能)
    - DashScope (qwen-plus)
    - Python 3.12

设计约束:
    - 必须配置 DASHSCOPE_API_KEY 环境变量
    - 使用 MemorySaver 作为 checkpointer 以支持状态持久化
    - 子代理的工具调用需要人工审核批准
    - Map Team Agent 本身不使用工具，只负责任务分发
    - 复用已有的 map_worker 实现
"""

import asyncio
import os
import uuid
from typing import Any

from deepagents import CompiledSubAgent, create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from core.tool_context import wrap_runnable_with_tool_context

# 导入已有的代理实现
from newAgents.worker.map_worker import create_map_agent as create_map_worker_agent

# 加载环境变量
load_dotenv()

# Team Supervisor 系统提示词
MAP_TEAM_PROMPT = """你是高德地图服务团队的 Supervisor，负责协调地图相关任务。

## 团队成员

你可以委托任务给以下子代理：
- **map_worker**: 路线规划专家，负责规划各种出行路线

## 工作流程

1. **理解需求**: 分析用户的出行规划需求
2. **提取信息**: 识别起点、终点坐标或地名
3. **委托任务**:  将任务委托给 route_planner 规划路线
4. **整理结果**: 将路线信息整理成用户友好的格式

## 坐标说明

- 坐标格式为 "经度,纬度"，例如 "116.481028,39.989643"
- 如果用户提供地名而非坐标，需要先进行地理编码（未来功能）
- 当前仅支持坐标直接输入

## 输出要求

- 清晰展示路线总距离和预计时长
- 如果规划失败，说明可能的原因
- 建议用户检查坐标格式是否正确
"""


def create_map_team_agent() -> tuple:
    """
    创建 Map Team Agent（团队代理）。

    该函数创建一个团队代理，它管理一个子代理：
    - map_worker: 负责路线规划（复用 map_worker 实现）

    团队代理本身不使用工具，而是将任务委派给合适的子代理执行。

    Returns:
        tuple: (agent, checkpointer) - 代理实例和检查点保存器

    Raises:
        ValueError: 当 DASHSCOPE_API_KEY 未设置时抛出
    """
    # 验证 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

    # 初始化 DashScope 模型（使用 ChatOpenAI 封装）
    model = ChatOpenAI(
        model="qwen-plus",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key=api_key,
    )

    # 创建内存检查点保存器（Human-in-the-loop 必需）
    checkpointer = MemorySaver()

    # 创建地图子代理（复用已有实现）
    map_worker_agent, _ = create_map_worker_agent()
    map_worker_subagent = CompiledSubAgent(
        name="map_worker",
        description="路线规划专家，负责规划各种出行路线",
        runnable=map_worker_agent,
    )

    # 定义子代理列表
    subagents = [map_worker_subagent]

    # 创建 Map Team Agent（团队代理）
    agent = create_deep_agent(
        model=model,
        system_prompt=MAP_TEAM_PROMPT,
        subagents=subagents,
        checkpointer=checkpointer,
    )

    agent = wrap_runnable_with_tool_context(agent)
    return agent, checkpointer


def handle_human_review(result: dict, config: dict, agent: Any) -> dict:
    """
    处理人工审核流程。

    当子代理执行被中断时，显示待审核的工具调用信息，并等待用户决策。

    Args:
        result: 代理执行结果，包含中断信息
        config: 配置字典，包含 thread_id 用于状态持久化
        agent: Deep Agent 实例

    Returns:
        dict: 恢复执行后的最终结果
    """
    # 提取中断信息
    interrupts = result["__interrupt__"][0].value
    action_requests = interrupts["action_requests"]
    review_configs = interrupts["review_configs"]

    # 创建工具名到审核配置的映射
    config_map = {cfg["action_name"]: cfg for cfg in review_configs}

    # 显示待审核的工具调用信息
    print("\n" + "=" * 50)
    print("检测到子代理工具调用，需要人工审核：")
    print("=" * 50)

    for action in action_requests:
        review_config = config_map[action["name"]]
        print(f"\n工具名称: {action['name']}")
        print(f"调用参数: {action['args']}")
        print(f"允许的决策: {review_config['allowed_decisions']}")

    # 获取用户决策
    user_input = input("\n请输入决策 (approve/reject/edit): ").strip().lower()

    # 构建决策列表
    if user_input == "approve":
        decisions = [{"type": "approve"}]
        print("✓ 已批准工具调用")
    elif user_input == "reject":
        decisions = [{"type": "reject"}]
        print("✗ 已拒绝工具调用")
    else:
        # 默认批准
        decisions = [{"type": "approve"}]
        print("✓ 默认批准工具调用")

    # 使用相同的 config 恢复执行
    result = asyncio.run(
        agent.ainvoke(
            Command(resume={"decisions": decisions}),
            config=config,
        )
    )

    return result


def main() -> None:
    """
    主函数：演示 Map Team Agent 的多轮对话功能。

    执行流程:
        1. 创建 Map Team Agent（包含 map_worker 子代理）
        2. 进入多轮对话循环
        3. 每次请求如果触发中断，进行人工审核
        4. 显示结果并继续下一轮对话
    """
    try:
        # 创建团队代理
        agent, checkpointer = create_map_team_agent()

        # 创建配置，包含唯一的 thread_id 用于状态持久化
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        print("=" * 60)
        print("Map Team Agent")
        print("=" * 60)
        print(f"Thread ID: {thread_id}")
        print("\n功能说明：")
        print("- 记忆系统由 top_supervisor 统一管理")
        print("- 输入 'quit' 或 'exit' 退出")
        print("=" * 60)

        # 多轮对话循环
        while True:
            # 获取用户输入
            user_input = input("\n你: ").strip()

            # 退出条件
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("\n再见！")
                break

            if not user_input:
                continue

            # 发送用户请求
            result = asyncio.run(
                agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                )
            )

            # 检查是否触发中断（需要人工审核）
            if result.get("__interrupt__"):
                result = handle_human_review(result, config, agent)

            # 显示助手回复
            print(f"\n助手: {result['messages'][-1].content}")

    except ValueError as e:
        print(f"配置错误: {e}")
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"执行错误: {e}")


if __name__ == "__main__":
    main()

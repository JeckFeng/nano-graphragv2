"""
SQL Team Agent - 数据库查询团队代理

本模块实现一个团队代理，它不直接使用工具，而是通过调用子代理完成数据库检索任务。
SQL Team Agent 管理一个子代理：sql_worker。

技术栈:
    - LangGraph 1.0
    - Deep Agents (CompiledSubAgent 功能)
    - DashScope (qwen-plus)
    - Python 3.12

设计约束:
    - 必须配置 DASHSCOPE_API_KEY 环境变量
    - 使用 MemorySaver 作为 checkpointer 以支持状态持久化
    - 子代理的工具调用需要人工审核批准
    - SQL Team Agent 本身不使用工具，只负责任务分发
    - 复用已有的 sql_worker 实现
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
from newAgents.worker.sql_worker import create_sql_worker as create_sql_worker_agent

# 加载环境变量
load_dotenv()

# Team Supervisor 系统提示词
SQL_TEAM_PROMPT = """你是 SQL 团队的 Supervisor，负责协调数据库查询任务。

## 团队成员

你可以委托任务给以下 Worker：
- **sql_worker**: SQL 数据库查询 Worker，负责执行具体的数据库查询任务，包括：
  - 检查数据库连接
  - 获取数据库结构
  - 生成 SQL 语句
  - 验证和修正 SQL
  - 执行 SQL 查询

## 工作流程

1. **理解需求**: 分析用户的查询需求
2. **委托任务**: 将任务委托给 sql_worker 执行查询
3. **解释结果**: 将查询结果转换为用户可理解的答案



## 输出要求

- 说明使用的表和字段
- 提供执行的 SQL 语句
- 以表格形式展示结果
- 用自然语言解释查询结果的含义

## 约束与异常处理

- 仅委托一次 sql_worker，避免重复调用导致循环
- 若 sql_worker 返回包含 "call_exhausted": true 的结果，立即回复：
  调用次数已用尽，需要用户确认/缩小范围
- 当用户只询问表清单或表结构时，优先要求 sql_worker 返回表名与列名即可
"""


def create_sql_team_agent() -> tuple:
    """
    创建 SQL Team Agent（团队代理）。

    该函数创建一个团队代理，它管理一个子代理：
    - sql_worker: 负责数据库查询（复用 sql_worker 实现）

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

    # 创建 SQL 子代理（复用已有实现）
    sql_worker_agent, _ = create_sql_worker_agent()
    sql_worker_subagent = CompiledSubAgent(
        name="sql_worker",
        description="SQL 数据库查询 Worker，负责执行具体的数据库查询任务",
        runnable=sql_worker_agent,
    )

    # 定义子代理列表
    subagents = [sql_worker_subagent]

    # 创建 SQL Team Agent（团队代理）
    agent = create_deep_agent(
        model=model,
        system_prompt=SQL_TEAM_PROMPT,
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
    主函数：演示 SQL Team Agent 的多轮对话功能。

    执行流程:
        1. 创建 SQL Team Agent（包含 sql_worker 子代理）
        2. 进入多轮对话循环
        3. 每次请求如果触发中断，进行人工审核
        4. 显示结果并继续下一轮对话
    """
    try:
        # 创建团队代理
        agent, checkpointer = create_sql_team_agent()

        # 创建配置，包含唯一的 thread_id 用于状态持久化
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        print("=" * 60)
        print("SQL Team Agent")
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

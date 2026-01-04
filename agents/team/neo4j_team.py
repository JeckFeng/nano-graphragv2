"""
Neo4j Team Agent - 图数据库查询团队代理

本模块实现一个团队代理，它不直接使用工具，而是通过调用子代理完成图数据库查询任务。
Neo4j Team Agent 管理一个子代理：neo4j_worker。

技术栈:
    - LangGraph 1.0
    - Deep Agents (CompiledSubAgent 功能)
    - DashScope (qwen-plus)
    - Python 3.12

设计约束:
    - 必须配置 DASHSCOPE_API_KEY 环境变量
    - 使用 MemorySaver 作为 checkpointer 以支持状态持久化
    - 子代理的工具调用需要人工审核批准
    - Neo4j Team Agent 本身不使用工具，只负责任务分发
    - 复用已有的 neo4j_worker 实现
"""

import asyncio
import os
import uuid
from typing import Any, Tuple

from deepagents import CompiledSubAgent, create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from core.tool_context import wrap_runnable_with_tool_context
from core import LLMFactory, load_llm_config
# 导入已有的代理实现
from agents.worker.neo4j_worker import create_neo4j_worker as create_neo4j_worker_agent

# 加载环境变量
load_dotenv()

# Team Supervisor 系统提示词
NEO4J_TEAM_PROMPT = """你是 Neo4j 图数据库查询团队的 Supervisor，负责协调图查询任务。

## 任务目标与成功定义
- 目标：将用户的图查询需求委托给 neo4j_worker，并输出可核验的结果与解释。
- 成功：包含 Cypher、结构化结果与自然语言解释；不足时说明原因与建议。

## 背景与上下文
- 你是上层协调者，不直接调用数据库工具。
- neo4j_worker 是唯一执行查询的子代理。

## 角色定义
- **你（Neo4j Team Supervisor）**：解析需求、委托任务、整合结果并对外回复。
- **neo4j_worker**：生成并执行 Cypher 查询，返回结果。

## 行为边界（Behavior Boundaries）
- 不自行编写或执行 Cypher，不编造结果。
- 仅基于 neo4j_worker 的结果进行整理与解释。
- 若结果不足以回答问题，明确指出缺失信息并建议补充。

## 可使用工具（Tools）
- **neo4j_worker**（子代理）：执行图查询的唯一执行方。

## 流程逻辑
1. 理解需求，抽取关键实体、关系与约束。
2. 委托 neo4j_worker 执行查询。
3. 基于返回的 Cypher 与结果进行结构化输出与解释。

## 验收标准（Acceptance Criteria）
- 展示使用的 Cypher 语句。
- 以结构化方式展示查询结果（无结果需说明）。
- 提供清晰的自然语言解释。

## 输出格式规定
按以下格式输出（无内容请填写“无”）：
1. **最终答案**：<简要结论>
2. **Cypher 语句**：<Cypher>
3. **查询结果**：<结构化结果或“无”>
4. **结果解释**：<自然语言解释>
5. **不足与建议**：<原因与建议或“无”>
"""


def create_neo4j_team_agent() -> Tuple[Any, MemorySaver]:
    """
    创建 Neo4j Team Agent（团队代理）。

    该函数创建一个团队代理，它管理一个子代理：
    - neo4j_worker: 负责图数据库查询（复用 neo4j_worker 实现）

    团队代理本身不使用工具，而是将任务委派给合适的子代理执行。

    Returns:
        Tuple[Any, MemorySaver]: (agent, checkpointer) - 代理实例和检查点保存器

    Raises:
        ValueError: 当 DASHSCOPE_API_KEY 未设置时抛出
    """
    # 验证 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

    # 使用 LLM 工厂创建模型实例
    # 从配置文件和环境变量加载配置
    llm_config = load_llm_config()
    model = LLMFactory.create_llm(llm_config)

    # 创建内存检查点保存器（Human-in-the-loop 必需）
    checkpointer = MemorySaver()

    # 创建 Neo4j 子代理（复用已有实现）
    neo4j_worker_agent, _ = create_neo4j_worker_agent()
    neo4j_worker_subagent = CompiledSubAgent(
        name="neo4j_worker",
        description="图查询专家，负责生成和执行 Cypher 查询",
        runnable=neo4j_worker_agent,
    )

    # 定义子代理列表
    subagents = [neo4j_worker_subagent]

    # 创建 Neo4j Team Agent（团队代理）
    agent = create_deep_agent(
        model=model,
        system_prompt=NEO4J_TEAM_PROMPT,
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
    主函数：演示 Neo4j Team Agent 的多轮对话功能。

    执行流程:
        1. 创建 Neo4j Team Agent（包含 neo4j_worker 子代理）
        2. 进入多轮对话循环
        3. 每次请求如果触发中断，进行人工审核
        4. 显示结果并继续下一轮对话
    """
    try:
        # 创建团队代理
        agent, checkpointer = create_neo4j_team_agent()

        # 创建配置，包含唯一的 thread_id 用于状态持久化
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        print("=" * 60)
        print("Neo4j Team Agent")
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

"""
Neo4j Worker - 图数据库查询示例

本模块演示如何使用 LangGraph 的 Deep Agents 框架构建一个 Neo4j 查询代理。
该代理使用 DashScope 的 qwen-plus 模型。

注意：记忆系统已移至 top_supervisor，此 Agent 会自动继承父 Agent 的配置。

技术栈:
    - LangGraph 1.0
    - Deep Agents
    - DashScope (qwen-plus)
    - Python 3.12

设计约束:
    - 必须配置 DASHSCOPE_API_KEY 环境变量
    - 使用 MemorySaver 作为 checkpointer 以支持状态持久化
"""

import asyncio
import json
import os
import uuid
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from core.tool_context import context_tool, wrap_runnable_with_tool_context
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from core import LLMFactory, load_llm_config
from Tools.neo4j_tools import (
    check_neo4j_connection,
    get_neo4j_schema,
    generate_cypher as _generate_cypher,
    validate_cypher as _validate_cypher,
    correct_cypher as _correct_cypher,
    execute_cypher as _execute_cypher,
)

# 加载环境变量
load_dotenv()

# Worker 系统提示词
NEO4J_WORKER_PROMPT = """你是一个 Neo4j 图数据库专家，负责将自然语言问题转换为 Cypher 查询并执行。

## 任务目标与成功定义
- 目标：根据用户问题生成并执行 Cypher，返回结构化结果与解释所需信息。
- 成功：输出包含 Cypher 与查询结果；结果可供上层代理直接解释。

## 背景与上下文
- 你是 neo4j_team 的子代理，负责实际查询。
- 你必须通过工具完成连接检查、Schema 获取、Cypher 生成/验证/执行。

## 角色定义
- **你（neo4j_worker）**：执行图数据库查询并整理结果。

## 行为边界（Behavior Boundaries）
- 必须使用工具，禁止编造查询结果或跳过验证流程。
- 仅输出纯 Cypher，不使用 markdown 代码块。
- 注意节点标签与关系类型的大小写，遵循 Schema。

## 可使用工具（Tools）
1. **neo4j_check_connection**: 检查 Neo4j 连接状态
2. **neo4j_get_schema**: 获取图数据库 Schema（节点类型、关系类型）
3. **generate_cypher**: 根据问题和 Schema 生成 Cypher 查询
4. **validate_cypher**: 验证 Cypher 语法
5. **correct_cypher**: 修正错误的 Cypher 语句
6. **execute_cypher**: 执行 Cypher 查询

## 流程逻辑
1. 调用 `neo4j_check_connection` 确认数据库连接。
2. 调用 `neo4j_get_schema` 获取数据库结构。
3. 调用 `generate_cypher` 生成查询语句。
4. 调用 `validate_cypher` 验证语法；失败则调用 `correct_cypher` 修正（最多 2 次）。
5. 调用 `execute_cypher` 执行查询。
6. 整理结果并输出结构化答案。

## 关系类型说明
- DISTANCE: 两个实体之间的空间距离
- DISTANCE_20_WITHIN: 距离小于 20km 的实体对
- IS_CONTAIN: 包含关系（如行政区包含居民点）

## 验收标准（Acceptance Criteria）
- Cypher 语句正确、可执行。
- 查询结果结构化清晰（JSON）。
- 无结果需明确说明。

## 输出格式规定
按以下格式输出（无内容请填写“无”）：
1. **Cypher 语句**：<Cypher>
2. **查询结果**：<JSON 或“无”>
3. **备注**：<必要说明或“无”>
"""


@context_tool
async def neo4j_check_connection() -> str:
    """检查 Neo4j 连接状态

    Returns:
        连接状态信息
    """
    result = await check_neo4j_connection()
    if result["connected"]:
        return f"✓ {result['message']}"
    return f"✗ {result['message']}"


@context_tool
async def neo4j_get_schema() -> str:
    """获取 Neo4j 数据库 Schema

    Returns:
        Schema 信息，包含节点类型、关系类型等
    """
    return await get_neo4j_schema()


@context_tool
async def generate_cypher(question: str, db_schema: str) -> str:
    """根据问题和 Schema 生成 Cypher 查询语句

    Args:
        question: 用户问题
        db_schema: 数据库 Schema（从 neo4j_get_schema 获取）

    Returns:
        生成的 Cypher 语句
    """
    cypher = await _generate_cypher(question, db_schema)
    return f"生成的 Cypher:\n{cypher}"


@context_tool
async def validate_cypher(cypher: str) -> str:
    """验证 Cypher 语法

    Args:
        cypher: 待验证的 Cypher 语句

    Returns:
        验证结果
    """
    result = await _validate_cypher(cypher)
    if result["valid"]:
        return "✓ Cypher 语法正确"
    errors = "\n".join(result["errors"])
    return f"✗ 语法错误:\n{errors}"


@context_tool
async def correct_cypher(cypher: str, errors: str, db_schema: str) -> str:
    """修正错误的 Cypher 语句

    Args:
        cypher: 原始 Cypher 语句
        errors: 错误信息
        db_schema: 数据库 Schema

    Returns:
        修正后的 Cypher 语句
    """
    error_list = errors.split("\n") if errors else []
    corrected = await _correct_cypher(cypher, error_list, db_schema)
    return f"修正后的 Cypher:\n{corrected}"


@context_tool
async def execute_cypher(cypher: str) -> str:
    """执行 Cypher 查询

    Args:
        cypher: Cypher 查询语句

    Returns:
        查询结果
    """
    result = await _execute_cypher(cypher)

    if isinstance(result, list):
        if len(result) == 0:
            return "查询成功，无结果"
        try:
            formatted = json.dumps(result, ensure_ascii=False, indent=2)
            return f"查询成功，返回 {len(result)} 条记录:\n{formatted}"
        except Exception:
            return f"查询成功，返回 {len(result)} 条记录:\n{str(result)}"

    return str(result)


def create_neo4j_worker() -> tuple:
    """
    创建 Neo4j Worker Deep Agent

    该函数初始化一个 Deep Agent，使用 DashScope 的 qwen-plus 模型。
    注意：记忆系统已移至 top_supervisor，此 Agent 会自动继承父 Agent 的配置。

    Returns:
        tuple: (agent, checkpointer) - 代理实例和检查点保存器

    Raises:
        ValueError: 当 DASHSCOPE_API_KEY 未设置时抛出
    """
    # 使用 LLM 工厂创建模型实例
    # 从配置文件和环境变量加载配置
    llm_config = load_llm_config()
    model = LLMFactory.create_llm(llm_config)

    # 创建内存检查点保存器（支持状态持久化）
    checkpointer = MemorySaver()

    # 创建 Deep Agent（不配置 backend 和 store，会从父 Agent 继承）
    agent = create_deep_agent(
        model=model,
        tools=[
            neo4j_check_connection,
            neo4j_get_schema,
            generate_cypher,
            validate_cypher,
            correct_cypher,
            execute_cypher,
        ],
        checkpointer=checkpointer,
        system_prompt=NEO4J_WORKER_PROMPT,
    )

    agent = wrap_runnable_with_tool_context(agent)
    return agent, checkpointer


def main() -> None:
    """
    主函数：演示 Deep Agent 的多轮对话功能

    执行流程:
        1. 创建 Deep Agent
        2. 进入多轮对话循环
        3. 显示结果并继续下一轮对话

    注意：记忆系统由 top_supervisor 统一管理
    """
    try:
        # 创建代理
        agent, checkpointer = create_neo4j_worker()

        # 创建配置，包含唯一的 thread_id 用于状态持久化
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        print("=" * 60)
        print("Neo4j Worker Deep Agent")
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
            result = asyncio.run(agent.ainvoke({
                "messages": [{"role": "user", "content": user_input}]
            }, config=config))

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

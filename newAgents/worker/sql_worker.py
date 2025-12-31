"""
SQL Worker - PostgreSQL 数据库查询示例

本模块演示如何使用 LangGraph 的 Deep Agents 框架构建一个数据库查询代理。
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
import uuid

from core import LLMFactory, load_llm_config
from deepagents import create_deep_agent
from dotenv import load_dotenv
from core.tool_context import context_tool, wrap_runnable_with_tool_context
from langgraph.checkpoint.memory import MemorySaver

# 加载环境变量
load_dotenv()

# Worker 系统提示词（请勿修改）
SQL_WORKER_PROMPT = """你是一个 PostgreSQL 数据库查询专家，负责将用户的自然语言问题转换为 SQL 查询并执行。

## 可用工具

1. **sql_check_connection**: 检查数据库连接状态
2. **sql_get_schema**: 获取数据库结构信息（表名、列名、注释）
3. **generate_sql**: 根据数据库信息和用户问题生成 SQL 语句
4. **validate_sql**: 验证 SQL 语法是否正确
5. **correct_sql**: 修正错误的 SQL 语句
6. **execute_sql**: 执行 SQL 查询并返回结果

## 工作流程

1. 首先调用 `sql_check_connection` 确认数据库连接正常
2. 调用 `sql_get_schema` 获取数据库结构信息
3. 调用 `generate_sql` 生成 SQL 查询语句
4. 调用 `validate_sql` 验证 SQL 语法
5. 如果验证失败，调用 `correct_sql` 修正，然后重新验证（最多 3 次）
6. 验证通过后，调用 `execute_sql` 执行查询
7. 解析结果并以清晰的格式返回给用户

## 注意事项

- 不要在查询结果中返回 geom 字段（会导致数据过大）
- 查询结果默认限制 50 条记录
- 如果涉及 PostGIS 空间查询，使用正确的空间函数
- 对于 INSERT/UPDATE/DELETE 等修改操作，需要谨慎执行
- sql_get_schema 仅返回表名与列名，如需更多表请使用 limit/offset（返回为空表示无更多）
- 如果用户仅询问有哪些表或表结构，直接使用 sql_get_schema 并返回结果，不必生成 SQL

## 输出格式

回答应包含：
1. 查询的表名和 Schema
2. 使用的 SQL 语句
3. 查询结果（表格形式）
4. 结果的自然语言解释

## 终止规则

- 当任何工具返回结果中包含 "call_exhausted": true 时，立即停止并回复：
  调用次数已用尽，需要用户确认/缩小范围
"""

_SQL_TOOLS = None


def _get_sql_tools():
    global _SQL_TOOLS
    if _SQL_TOOLS is None:
        from Tools.sql_tools import (
            check_database_connection,
            get_database_schema,
            generate_sql as _generate_sql,
            validate_sql as _validate_sql,
            correct_sql as _correct_sql,
            execute_sql as _execute_sql,
        )
        _SQL_TOOLS = {
            "check_database_connection": check_database_connection,
            "get_database_schema": get_database_schema,
            "generate_sql": _generate_sql,
            "validate_sql": _validate_sql,
            "correct_sql": _correct_sql,
            "execute_sql": _execute_sql,
        }
    return _SQL_TOOLS


@context_tool
async def sql_check_connection() -> str:
    """检查数据库连接是否正常

    Returns:
        连接状态信息
    """
    tools = _get_sql_tools()
    result = await tools["check_database_connection"]()
    if result["connected"]:
        return f"✓ {result['message']}，数据库: {result['database']}"
    return f"✗ {result['message']}"


@context_tool
async def sql_get_schema() -> str:
    """获取数据库结构信息（表名、列名、注释等）

    Returns:
        JSON 格式的数据库结构信息
    """
    tools = _get_sql_tools()
    schema_json = await tools["get_database_schema"]()
    return schema_json


@context_tool
async def generate_sql(database_info: str, task_description: str) -> str:
    """根据数据库信息和任务描述生成 SQL 语句

    Args:
        database_info: 数据库结构信息（从 sql_get_schema 获取）
        task_description: 用户的查询需求描述

    Returns:
        生成的 SQL 语句
    """
    tools = _get_sql_tools()
    sql = await tools["generate_sql"](database_info, task_description)
    return sql


@context_tool
async def validate_sql(sql: str) -> str:
    """验证 SQL 语句语法是否正确

    Args:
        sql: 待验证的 SQL 语句

    Returns:
        验证结果
    """
    tools = _get_sql_tools()
    result = await tools["validate_sql"](sql)
    if result["valid"]:
        return f"✓ {result['message']}"
    return f"✗ {result['message']}\n错误详情: {result['errors']}"


@context_tool
async def correct_sql(sql: str, error_message: str, database_info: str) -> str:
    """修正错误的 SQL 语句

    Args:
        sql: 原始错误的 SQL 语句
        error_message: 错误信息
        database_info: 数据库结构信息

    Returns:
        修正后的 SQL 语句
    """
    tools = _get_sql_tools()
    corrected = await tools["correct_sql"](sql, error_message, database_info)
    return f"修正后的 SQL:\n{corrected}"


@context_tool
async def execute_sql(query: str) -> str:
    """执行 SQL 查询

    Args:
        query: SQL 查询语句

    Returns:
        查询结果
    """
    tools = _get_sql_tools()
    result = await tools["execute_sql"](query)
    row_count = result.get("row_count", 0)
    rows = result.get("rows", [])
    content = json.dumps(rows, ensure_ascii=False, default=str)

    if row_count > 50:
        return f"查询成功，返回 {row_count} 条记录（结果较多，显示部分）:\n{content[:2000]}..."
    return f"查询成功，返回 {row_count} 条记录:\n{content}"


def create_sql_worker() -> tuple:
    """
    创建 SQL Worker Deep Agent

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
            sql_check_connection,
            sql_get_schema,
            generate_sql,
            validate_sql,
            correct_sql,
            execute_sql,
        ],
        checkpointer=checkpointer,
        system_prompt=SQL_WORKER_PROMPT,
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
        agent, checkpointer = create_sql_worker()

        # 创建配置，包含唯一的 thread_id 用于状态持久化
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        print("=" * 60)
        print("SQL Worker Deep Agent")
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

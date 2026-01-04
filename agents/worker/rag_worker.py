"""
RAG Worker - 知识库检索示例

本模块演示如何使用 LangGraph 的 Deep Agents 框架构建一个知识检索代理。
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
from typing import Any, Dict, Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from core.tool_context import context_tool, wrap_runnable_with_tool_context
from core.tool_errors import ToolError
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from core import LLMFactory, load_llm_config
from config.settings import get_settings
from core.prompts.worker.Prompts import RAG_WORKER_PROMPT
from Tools.rag_tools import (
    identify_query_intent,
    rewrite_query_and_extract_keywords as _rewrite_query_and_extract_keywords,
    graphrag_local_search,
    graphrag_global_search,
    search_images_by_keyword,
    search_tables_by_keyword,
    evaluate_retrieval_quality,
    json_to_markdown_table,
)

# 加载环境变量
load_dotenv()



def _get_graphrag_community_level() -> int:
    settings = get_settings()
    return settings.graphrag_community_level or 3


def _format_tool_error(error: Exception) -> Dict[str, Any]:
    """将工具异常转换为结构化错误信息。

    Args:
        error: 捕获到的异常实例

    Returns:
        Dict[str, Any]: 结构化错误字典
    """
    if isinstance(error, ToolError):
        return error.to_dict()
    if isinstance(error, dict) and "error" in error:
        return error
    return {"error": str(error)}


@context_tool
def identify_intent(query: str) -> str:
    """识别查询意图

    Args:
        query: 用户查询字符串

    Returns:
        意图类型: text_query / image_query / table_query
    """
    intent = identify_query_intent(query)
    intent_desc = {
        "text_query": "文本查询 - 建议使用 search_local 或 search_global",
        "image_query": "图片查询 - 建议使用 search_images",
        "table_query": "表格查询 - 建议使用 search_tables",
    }
    return f"识别结果: {intent}\n{intent_desc.get(intent, '')}"


@context_tool
async def rewrite_query_and_extract_keywords(query: str) -> str:
    """改写查询并提取关键词（必须首先调用）

    将原始用户问题改写为多个版本的同义问题，并提取关键检索词。
    这是 RAG Worker 流程的第一步，必须首先调用。

    Args:
        query: 原始查询字符串

    Returns:
        JSON 字符串，包含以下字段：
        - rewrite_query_list: List[str] - 改写后的查询列表（最多3个）
        - query_keywords_list: List[str] - 关键词列表（最多5个）
        - original_query: str - 原始查询
    """
    result = await _rewrite_query_and_extract_keywords(query)
    return json.dumps(result, ensure_ascii=False, indent=2)


@context_tool
async def parallel_identify_intent(queries_json: str) -> str:
    """并行执行多个意图识别

    对多个改写后的问题并行执行意图识别，确定每个问题的意图类型。

    Args:
        queries_json: JSON 格式的查询列表，例如：["查询1", "查询2", "查询3"]

    Returns:
        每个查询的意图识别结果汇总（JSON 格式），包含：
        - 每个查询的意图类型（text_query / image_query / table_query）
        - 识别结果说明
    """
    try:
        queries = json.loads(queries_json)
        if not isinstance(queries, list):
            queries = [queries]
    except json.JSONDecodeError:
        return f"错误: 无法解析查询列表 JSON: {queries_json}"

    async def identify_single(query: str) -> dict:
        try:
            intent = identify_query_intent(query)
            intent_desc = {
                "text_query": "文本查询 - 建议使用 search_local 或 search_global",
                "image_query": "图片查询 - 建议使用 search_images",
                "table_query": "表格查询 - 建议使用 search_tables",
            }
            return {
                "query": query,
                "intent": intent,
                "description": intent_desc.get(intent, ""),
                "error": None,
            }
        except Exception as exc:
            return {
                "query": query,
                "intent": "text_query",
                "description": "识别失败，使用默认文本查询",
                "error": str(exc),
            }

    tasks = [identify_single(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "query": queries[i] if i < len(queries) else "unknown",
                "intent": "text_query",
                "description": "识别异常",
                "error": str(result),
            })
        else:
            processed_results.append(result)

    summary = {
        "total": len(processed_results),
        "results": processed_results,
    }

    return json.dumps(summary, ensure_ascii=False, indent=2)


@context_tool
async def parallel_execute_queries(queries_with_intents_json: str) -> str:
    """根据意图并行执行检索

    根据每个改写问题的意图类型，并行执行相应的检索工具。

    Args:
        queries_with_intents_json: JSON 格式的查询和意图列表，例如：
            [
                {"query": "查询1", "intent": "text_query"},
                {"query": "查询2", "intent": "image_query"},
                {"query": "查询3", "intent": "table_query"}
            ]

    Returns:
        检索结果汇总（JSON 格式），包含：
        - total: 总查询数
        - results: 每个查询的检索结果列表，每个结果包含：
          * query: 原始查询
          * intent: 意图类型（text_query/image_query/table_query）
          * result: 检索结果文本
            - 对于 image_query，result 包含格式化的图片链接，格式为 "- ![](URI)"
            - 对于 table_query，result 包含表格数据，格式为 "完整数据 JSON: {...}"
          * tool_used: 使用的工具名称
          * error: 错误信息（如果有）

    重要提示：
    - 如果 results 中某个结果的 intent 是 "image_query" 且 result 包含 "- ![](URI)" 格式的图片链接，
      你必须将这些图片链接提取出来，并在最终答案中附加它们。
    - 如果 results 中某个结果的 intent 是 "table_query" 且 result 包含 "完整数据 JSON:"，
      你必须提取 JSON 数据并调用 json_to_markdown 工具进行转换。
    """
    try:
        queries_with_intents = json.loads(queries_with_intents_json)
        if not isinstance(queries_with_intents, list):
            queries_with_intents = [queries_with_intents]
    except json.JSONDecodeError:
        return f"错误: 无法解析查询和意图列表 JSON: {queries_with_intents_json}"

    community_level = _get_graphrag_community_level()

    async def execute_single_query(query_info: dict) -> dict:
        query = query_info.get("query", "")
        intent = query_info.get("intent", "text_query")

        try:
            if intent == "text_query":
                try:
                    result = await graphrag_local_search(query, community_level=community_level)
                    tool_used = "search_local"
                except ToolError:
                    try:
                        result = await graphrag_global_search(query)
                        tool_used = "search_global"
                    except ToolError as exc:
                        return {
                            "query": query,
                            "intent": intent,
                            "result": None,
                            "tool_used": "search_global",
                            "error": _format_tool_error(exc),
                        }
                return {
                    "query": query,
                    "intent": intent,
                    "result": result,
                    "tool_used": tool_used,
                    "error": None,
                }
            if intent == "image_query":
                try:
                    results = await search_images_by_keyword(query)
                except ToolError as exc:
                    return {
                        "query": query,
                        "intent": intent,
                        "result": None,
                        "tool_used": "search_images",
                        "error": _format_tool_error(exc),
                    }
                uris = [r.get("uri") for r in results if r.get("uri")]
                result_text = f"找到 {len(uris)} 张相关图片:\n"
                for uri in uris:
                    result_text += f"- ![]({uri})\n"
                return {
                    "query": query,
                    "intent": intent,
                    "result": result_text,
                    "tool_used": "search_images",
                    "error": None,
                }
            if intent == "table_query":
                try:
                    results = await search_tables_by_keyword(query)
                except ToolError as exc:
                    return {
                        "query": query,
                        "intent": intent,
                        "result": None,
                        "tool_used": "search_tables",
                        "error": _format_tool_error(exc),
                    }
                result_text = f"找到 {len(results)} 个相关表格:\n"
                for i, tbl in enumerate(results, 1):
                    caption = tbl.get("caption", "未命名表格")
                    result_text += f"{i}. {caption}\n"
                    if tbl.get("data"):
                        data_json = json.dumps(tbl.get("data"), ensure_ascii=False)
                        result_text += f"   完整数据 JSON: {data_json}\n"
                return {
                    "query": query,
                    "intent": intent,
                    "result": result_text,
                    "tool_used": "search_tables",
                    "error": None,
                }
            result = await graphrag_local_search(query, community_level=community_level)
            return {
                "query": query,
                "intent": intent,
                "result": result,
                "tool_used": "search_local",
                "error": None,
            }
        except Exception as exc:
            return {
                "query": query,
                "intent": intent,
                "result": None,
                "tool_used": "unknown",
                "error": _format_tool_error(exc),
            }

    tasks = [execute_single_query(qi) for qi in queries_with_intents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "query": queries_with_intents[i].get("query", "unknown") if i < len(queries_with_intents) else "unknown",
                "intent": queries_with_intents[i].get("intent", "text_query") if i < len(queries_with_intents) else "text_query",
                "result": None,
                "tool_used": "unknown",
                "error": _format_tool_error(result),
            })
        else:
            processed_results.append(result)

    summary = {
        "total": len(processed_results),
        "results": processed_results,
    }

    return json.dumps(summary, ensure_ascii=False, indent=2)


@context_tool
async def search_local(query: str, community_level: int = 0) -> str:
    """使用 GraphRAG 进行本地检索（适合细节问题）

    Args:
        query: 检索查询
        community_level: 社区层级（1-5），数字越小范围越小

    Returns:
        检索结果
    """
    if community_level <= 0:
        community_level = _get_graphrag_community_level()
    result = await graphrag_local_search(query, community_level)
    return f"本地检索结果:\n{result}"


@context_tool
async def search_global(query: str) -> str:
    """使用 GraphRAG 进行全局检索（适合宏观问题，耗时较长）

    Args:
        query: 检索查询

    Returns:
        检索结果
    """
    result = await graphrag_global_search(query)
    return f"全局检索结果:\n{result}"


@context_tool
async def search_images(keyword: str) -> str:
    """按关键词检索图片

    Args:
        keyword: 搜索关键词

    Returns:
        图片信息列表（包含图片 URI，格式为 markdown 图片链接格式：- ![](URI)）
    """
    results = await search_images_by_keyword(keyword)

    if not results:
        return "未找到相关图片"

    uris = [r.get("uri") for r in results if r.get("uri")]
    if not uris:
        return f"找到 {len(results)} 条相关记录，但无图片 URI"

    output = f"找到 {len(uris)} 张相关图片:\n"
    for uri in uris:
        output += f"- ![]({uri})\n"

    return output


@context_tool
async def search_tables(keyword: str) -> str:
    """按关键词检索表格

    Args:
        keyword: 搜索关键词

    Returns:
        表格信息列表（包含完整的表格数据 JSON，可用于 json_to_markdown 工具转换）
    """
    results = await search_tables_by_keyword(keyword)

    if not results:
        return "未找到相关表格"

    output = f"找到 {len(results)} 个相关表格:\n"
    for i, tbl in enumerate(results, 1):
        caption = tbl.get("caption", "未命名表格")
        output += f"{i}. {caption}\n"
        if tbl.get("data"):
            data_json = json.dumps(tbl.get("data"), ensure_ascii=False)
            output += f"   完整数据 JSON: {data_json}\n"
            output += f"   数据预览: {str(tbl['data'])[:200]}...\n"

    return output


@context_tool
async def evaluate_quality(query: str, retrieved: str) -> str:
    """评估检索结果质量

    Args:
        query: 用户查询
        retrieved: 检索到的内容

    Returns:
        质量评估结果 (high/low) 及原因
    """
    result = await evaluate_retrieval_quality(query, retrieved)
    score = result.get("score", "unknown")
    reason = result.get("reason", "")

    if score == "high":
        return f"✓ 检索质量: 高\n原因: {reason}\n建议: 可以基于这些结果生成答案"
    return f"✗ 检索质量: 低\n原因: {reason}\n建议: 尝试改写查询或使用其他检索方式"


@context_tool
def json_to_markdown(table_json: str) -> str:
    """将 JSON 表格数据转换为 Markdown 格式

    重要：当 search_tables 返回包含"完整数据 JSON:"的表格数据时，必须调用此工具将 JSON 转换为 Markdown 表格。

    Args:
        table_json: JSON 格式的表格数据。可以是：
          - 完整的 JSON 字符串（如从 search_tables 返回的"完整数据 JSON: {...}"中提取）
          - JSON 对象字符串
          - 支持 {"data": [...]} 格式或直接的二维数组格式

    Returns:
        Markdown 格式的表格字符串，可以直接插入到最终答案中
    """
    return json_to_markdown_table(table_json)


def create_rag_worker() -> tuple:
    """
    创建 RAG Worker Deep Agent

    该函数初始化一个 Deep Agent，使用 DashScope 的 qwen-plus 模型。
    注意：记忆系统已移至 top_supervisor，此 Agent 会自动继承父 Agent 的配置。

    Returns:
        tuple: (agent, checkpointer) - 代理实例和检查点保存器

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

    # 创建内存检查点保存器（支持状态持久化）
    checkpointer = MemorySaver()

    # 创建 Deep Agent（不配置 backend 和 store，会从父 Agent 继承）
    agent = create_deep_agent(
        model=model,
        tools=[
            rewrite_query_and_extract_keywords,
            parallel_identify_intent,
            parallel_execute_queries,
            identify_intent,
            search_local,
            search_global,
            search_images,
            search_tables,
            evaluate_quality,
            json_to_markdown,
        ],
        checkpointer=checkpointer,
        system_prompt=RAG_WORKER_PROMPT,
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
        agent, checkpointer = create_rag_worker()

        # 创建配置，包含唯一的 thread_id 用于状态持久化
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        print("=" * 60)
        print("RAG Worker Deep Agent")
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

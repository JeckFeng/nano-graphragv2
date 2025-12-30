"""
Neo4j 图数据库工具函数模块

提供 Neo4j 数据库操作相关的工具函数，包括：
- 数据库连接检查
- Schema 信息获取
- Cypher 语句生成
- Cypher 语法验证
- Cypher 错误修正
- Cypher 执行

使用示例:
    from tools.neo4j_tools import check_neo4j_connection, execute_cypher
    
    status = await check_neo4j_connection()
    result = await execute_cypher("MATCH (n) RETURN n LIMIT 10")
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
import os

from config.settings import get_settings
from core.tool_errors import ToolError


logger = logging.getLogger(__name__)

# 全局 Neo4jGraph 实例（延迟初始化）
_neo4j_graph = None
_neo4j_schema = ""
_neo4j_structured_schema: Dict[str, Any] = {}


async def _ensure_neo4j_graph():
    """
    延迟初始化 Neo4jGraph
    
    在首次使用时创建连接，避免启动时阻塞。
    
    Returns:
        Neo4jGraph 实例或 None（如果未配置）
    """
    global _neo4j_graph, _neo4j_schema, _neo4j_structured_schema
    
    if _neo4j_graph is not None:
        return _neo4j_graph
    
    settings = get_settings()
    uri = settings.neo4j_uri
    user = settings.neo4j_user
    pwd = settings.neo4j_password.get_secret_value()
    timeout = settings.neo4j_timeout
    
    if not uri or not user or not pwd:
        logger.warning("Neo4j credentials not configured")
        return None
    
    try:
        from langchain_neo4j import Neo4jGraph
        
        def _create_graph():
            return Neo4jGraph(
                url=uri,
                username=user,
                password=pwd,
                timeout=timeout,
                enhanced_schema=True,
            )
        
        # 在线程池中创建，避免阻塞事件循环
        _neo4j_graph = await asyncio.to_thread(_create_graph)
        
        # 缓存 schema
        _neo4j_schema = await asyncio.to_thread(lambda: _neo4j_graph.schema)
        _neo4j_structured_schema = await asyncio.to_thread(
            lambda: _neo4j_graph.structured_schema
        )
        
        logger.info("Neo4j connection established and schema cached")
        return _neo4j_graph
        
    except Exception as e:
        logger.error(f"Failed to create Neo4jGraph: {e}")
        _neo4j_graph = None
        return None


async def check_neo4j_connection() -> Dict[str, Any]:
    """
    检查 Neo4j 连接是否正常
    
    Returns:
        Dict 包含:
            - connected: bool 是否连接成功
            - message: str 连接状态消息
    """
    try:
        graph = await _ensure_neo4j_graph()
        
        if graph is None:
            raise ToolError(
                "Neo4j 未配置或连接失败",
                code="neo4j_not_configured",
            )
        
        return {
            "connected": True,
            "message": "Neo4j 连接正常",
        }
        
    except Exception as e:
        logger.error(f"Neo4j connection check failed: {e}")
        raise ToolError(
            "Neo4j 连接失败",
            code="neo4j_connection_failed",
            cause=e,
        ) from e


async def get_neo4j_schema() -> str:
    """
    获取 Neo4j 数据库 Schema
    
    返回节点类型、关系类型等结构信息。
    
    Returns:
        Schema 字符串
    """
    global _neo4j_schema
    
    graph = await _ensure_neo4j_graph()
    
    if graph is None:
        raise ToolError(
            "Neo4j 未配置或连接失败",
            code="neo4j_not_configured",
        )
    
    return _neo4j_schema


async def generate_cypher(
    question: str,
    schema: str,
    llm: Optional[Any] = None,
) -> str:
    """
    根据问题和 Schema 生成 Cypher 查询语句
    
    Args:
        question: 用户问题
        schema: 数据库 Schema
        llm: LLM 实例
        
    Returns:
        生成的 Cypher 语句
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    if llm is None:
        from core.llm import get_llm
        llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个 Neo4j 专家，负责将自然语言问题转换为准确的 Cypher 查询语句。"
         "请根据问题的语义选择合适的关系类型，只返回 Cypher 语句（不要反引号，不要解释）。"),
        ("human",
         """数据库结构:
{schema}

关系类型说明:
1. DISTANCE: 表示两个实体之间的空间距离
2. DISTANCE_20_WITHIN: 表示两个实体之间的空间距离小于20km
3. IS_CONTAIN: 表示一个实体包含另一个实体

关系选择规则:
- 当问题涉及计算两个实体之间的距离或查找最近实体时，使用 DISTANCE 关系
- 当问题涉及查找距离某实体20km以内的实体时，优先使用 DISTANCE_20_WITHIN 关系
- 当问题涉及实体包含关系时，使用 IS_CONTAIN 关系

用户问题: {question}
Cypher 查询:"""),
    ])
    
    try:
        chain = prompt | llm | StrOutputParser()
        cypher = await chain.ainvoke({"question": question, "schema": schema})
        cypher = cypher.strip()
        
        # 清理可能的 markdown 标记
        if cypher.startswith("```cypher"):
            cypher = cypher[9:]
        if cypher.startswith("```"):
            cypher = cypher[3:]
        if cypher.endswith("```"):
            cypher = cypher[:-3]
        cypher = cypher.strip()
        
        logger.info(f"Cypher generated: {cypher}")
        return cypher
        
    except Exception as e:
        logger.error(f"Cypher generation failed: {e}")
        raise ToolError(
            "Cypher 生成失败",
            code="cypher_generation_failed",
            cause=e,
        ) from e


async def validate_cypher(cypher: str) -> Dict[str, Any]:
    """
    验证 Cypher 语句语法是否正确
    
    使用 EXPLAIN 命令检查语法。
    
    Args:
        cypher: 待验证的 Cypher 语句
        
    Returns:
        Dict 包含:
            - valid: bool 是否有效
            - errors: List[str] 错误列表
    """
    graph = await _ensure_neo4j_graph()
    
    if graph is None:
        raise ToolError(
            "Neo4j 未配置或连接失败",
            code="neo4j_not_configured",
        )
    
    try:
        from neo4j.exceptions import CypherSyntaxError
        
        await asyncio.to_thread(graph.query, f"EXPLAIN {cypher}")
        
        logger.info("Cypher validation passed")
        return {
            "valid": True,
            "errors": [],
        }
        
    except CypherSyntaxError as e:
        error_msg = getattr(e, "message", str(e)) or "Cypher syntax error"
        logger.warning(f"Cypher syntax error: {error_msg}")
        return {
            "valid": False,
            "errors": [error_msg],
        }
    except Exception as e:
        logger.warning(f"Cypher validation failed: {e}")
        raise ToolError(
            "Cypher 验证失败",
            code="cypher_validation_failed",
            cause=e,
        ) from e


async def correct_cypher(
    cypher: str,
    errors: List[str],
    schema: str,
    llm: Optional[Any] = None,
) -> str:
    """
    修正错误的 Cypher 语句
    
    Args:
        cypher: 原始 Cypher 语句
        errors: 错误信息列表
        schema: 数据库 Schema
        llm: LLM 实例
        
    Returns:
        修正后的 Cypher 语句
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    if llm is None:
        from core.llm import get_llm
        llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "修正 Cypher 语句。只返回修正后的 Cypher（不要解释，不要 markdown 标记）。"),
        ("human",
         """Schema:
{schema}

Cypher:
{cypher}

错误:
{errors}

修正后的 Cypher:"""),
    ])
    
    try:
        chain = prompt | llm | StrOutputParser()
        corrected = await chain.ainvoke({
            "cypher": cypher,
            "errors": "\n".join(errors),
            "schema": schema,
        })
        corrected = corrected.strip()
        
        # 清理可能的 markdown 标记
        if corrected.startswith("```cypher"):
            corrected = corrected[9:]
        if corrected.startswith("```"):
            corrected = corrected[3:]
        if corrected.endswith("```"):
            corrected = corrected[:-3]
        corrected = corrected.strip()
        
        logger.info(f"Cypher corrected: {corrected}")
        return corrected
        
    except Exception as e:
        logger.error(f"Cypher correction failed: {e}")
        return cypher  # 返回原始语句



async def execute_cypher(cypher: str) -> Union[List[Dict], str]:
    """
    执行 Cypher 查询
    
    Args:
        cypher: Cypher 查询语句
        
    Returns:
        查询结果列表或错误信息字符串
    """
    graph = await _ensure_neo4j_graph()
    
    if graph is None:
        raise ToolError(
            "Neo4j 未配置或连接失败",
            code="neo4j_not_configured",
        )
    
    try:
        records = await asyncio.to_thread(graph.query, cypher)
        
        if records:
            logger.info(f"Cypher executed successfully, {len(records)} records returned")
            return records
        else:
            return "查询成功，无结果"
            
    except Exception as e:
        logger.error(f"Cypher execution failed: {e}")
        raise ToolError(
            "Cypher 执行失败",
            code="cypher_execute_failed",
            cause=e,
        ) from e


async def execute_cypher_with_retry(
    cypher: str,
    schema: str,
    max_retries: int = 2,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    执行 Cypher 查询，失败时自动修正并重试
    
    Args:
        cypher: Cypher 查询语句
        schema: 数据库 Schema（用于修正）
        max_retries: 最大重试次数
        llm: LLM 实例
        
    Returns:
        Dict 包含:
            - success: bool 是否成功
            - result: 查询结果
            - attempts: int 尝试次数
            - final_cypher: str 最终执行的 Cypher
    """
    current_cypher = cypher
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        # 先验证
        try:
            validation = await validate_cypher(current_cypher)
        except ToolError:
            raise
        
        if not validation["valid"]:
            if attempt < max_retries:
                # 尝试修正
                current_cypher = await correct_cypher(
                    current_cypher,
                    validation["errors"],
                    schema,
                    llm,
                )
                continue
            else:
                last_error = validation["errors"]
                break
        
        # 执行
        try:
            result = await execute_cypher(current_cypher)
        except ToolError as exc:
            last_error = exc.to_dict()
            if attempt < max_retries:
                current_cypher = await correct_cypher(
                    current_cypher,
                    [str(last_error)],
                    schema,
                    llm,
                )
                continue
            break
        
        if isinstance(result, list):
            return {
                "success": True,
                "result": result,
                "attempts": attempt,
                "final_cypher": current_cypher,
            }
        elif isinstance(result, str) and not result.startswith("执行失败"):
            return {
                "success": True,
                "result": result,
                "attempts": attempt,
                "final_cypher": current_cypher,
            }
        else:
            last_error = result
            if attempt < max_retries:
                # 尝试修正
                current_cypher = await correct_cypher(
                    current_cypher,
                    [str(last_error)],
                    schema,
                    llm,
                )
    
    raise ToolError(
        f"Cypher 执行失败（已重试 {max_retries} 次）",
        code="cypher_execute_retry_failed",
        details={
            "attempts": max_retries,
            "final_cypher": current_cypher,
            "last_error": last_error,
        },
    )

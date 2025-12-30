"""
RAG 知识检索工具函数模块

提供 GraphRAG 和 PostgreSQL 相关的知识检索工具函数，包括：
- 查询意图识别
- 问题改写
- GraphRAG 本地/全局检索
- 图片/表格检索
- 检索质量评估
- JSON 转 Markdown 表格

使用示例:
    from Tools.rag_tools import identify_query_intent, graphrag_local_search
    
    intent = identify_query_intent("帮我查找产品说明书中的退货政策")
    result = await graphrag_local_search("退货政策", community_level=2)
    
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import logging
import os

import pandas as pd

from config.settings import get_settings
from core.tool_errors import ToolError

logger = logging.getLogger(__name__)


def identify_query_intent(query: str) -> str:
    """
    识别查询意图
    
    根据关键词简单识别查询是关于文本、图片还是表格。
    
    Args:
        query: 用户查询字符串
        
    Returns:
        意图类型: "text_query" | "image_query" | "table_query"
    """
    lowered = query.lower()
    
    # 图片相关关键词
    if any(k in lowered for k in ["image", "picture", "图像", "图片", "照片", "图", "fig.", "photo"]):
        return "image_query"
    
    # 表格相关关键词
    if any(k in lowered for k in ["table", "data", "表格", "数据表", "表", "统计"]):
        return "table_query"
    
    return "text_query"


async def rewrite_query_and_extract_keywords(
    query: str,
    llm: Optional[Any] = None,
    max_rewrites: int = 3,
    max_keywords: int = 5,
) -> Dict[str, Any]:
    """
    使用 LLM 改写查询并提取关键词
    
    将用户问题改写为多个版本的同义问题，并提取关键检索词，用于提高检索召回率和工具选择。
    
    Args:
        query: 原始查询字符串
        llm: LLM 实例，可选
        max_rewrites: 最大改写数量（默认3）
        max_keywords: 最大关键词数量（默认5）
        
    Returns:
        Dict 包含以下字段：
        - rewrite_query_list: List[str] - 改写后的查询列表（最多 max_rewrites 个）
        - query_keywords_list: List[str] - 关键词列表（最多 max_keywords 个）
        - original_query: str - 原始查询
        
    Example:
        >>> result = await rewrite_query_and_extract_keywords("此次山洪灾害风险隐患调查涉及多少个防治对象？")
        >>> print(result["rewrite_query_list"])
        ['请列出山洪灾害风险隐患调查的所有防治对象及其数量。', ...]
        >>> print(result["query_keywords_list"])
        ['山洪灾害', '风险隐患', '防治对象数量', ...]
    """
    from langchain_core.prompts import ChatPromptTemplate
    import re
    
    if llm is None:
        from core import LLMFactory, load_llm_config
        # 使用 LLM 工厂创建模型实例
        # 从配置文件和环境变量加载配置
        llm_config = load_llm_config()
        llm = LLMFactory.create_llm(llm_config)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是检索问题改写和关键词提取专家。

任务：
1. 将用户问题改写为最多{max_rewrites}个不同表达方式的同义问题，提高检索召回率
2. 提取最多{max_keywords}个关键检索词，用于工具选择和结果评估

改写原则：
- 保留核心语义和关键信息
- 使用不同的表达方式（正式/非正式、专业术语/日常用语）
- 针对知识库的语言风格进行优化
- 确保改写后的问题语义相同但表达不同

关键词提取原则：
- 提取核心概念和实体
- 包含专业术语和同义词
- 用于判断是否需要检索图片/表格（如包含"表格"/"表"/"图"/"图片"/"fig."等）
- 关键词应该简洁、准确、具有检索价值

输出格式（严格JSON格式，不要添加任何解释或markdown标记）：
{{
    "rewrite_query_list": ["改写问题1", "改写问题2", "改写问题3"],
    "query_keywords_list": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]
}}

只返回JSON，不要添加任何解释、markdown代码块标记或其他内容。"""),
        ("human", "原始问题：{query}"),
    ])
    
    try:
        chain = prompt | llm
        response = await chain.ainvoke({
            "query": query,
            "max_rewrites": max_rewrites,
            "max_keywords": max_keywords,
        })
        raw_text = response.content if hasattr(response, "content") else str(response)
        
        # 清理响应文本：移除可能的 markdown 代码块标记
        raw_text = raw_text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()
        
        # 尝试提取 JSON 部分（如果响应包含其他文本）
        json_match = re.search(r'\{[^{}]*"rewrite_query_list"[^{}]*\}', raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)
        
        # 解析 JSON
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试修复常见的 JSON 问题
            # 移除可能的注释或多余字符
            raw_text = re.sub(r'//.*?$', '', raw_text, flags=re.MULTILINE)
            raw_text = re.sub(r'/\*.*?\*/', '', raw_text, flags=re.DOTALL)
            parsed = json.loads(raw_text)
        
        # 验证和限制数量
        rewrite_list = parsed.get("rewrite_query_list", [])
        keywords_list = parsed.get("query_keywords_list", [])
        
        # 确保是列表类型
        if not isinstance(rewrite_list, list):
            rewrite_list = [str(rewrite_list)] if rewrite_list else []
        if not isinstance(keywords_list, list):
            keywords_list = [str(keywords_list)] if keywords_list else []
        
        # 限制数量
        rewrite_list = rewrite_list[:max_rewrites]
        keywords_list = keywords_list[:max_keywords]
        
        # 过滤空值
        rewrite_list = [q.strip() for q in rewrite_list if q and q.strip()]
        keywords_list = [k.strip() for k in keywords_list if k and k.strip()]
        
        # 如果改写列表为空，至少保留原始查询
        if not rewrite_list:
            rewrite_list = [query]
        
        result = {
            "rewrite_query_list": rewrite_list,
            "query_keywords_list": keywords_list,
            "original_query": query,
        }
        
        logger.info(
            f"问题改写和关键词提取完成: 原始='{query}', "
            f"改写数量={len(rewrite_list)}, 关键词数量={len(keywords_list)}"
        )
        return result
        
    except Exception as e:
        logger.warning(f"问题改写和关键词提取失败: {e}，返回降级结果")
        # 降级方案：返回原始查询和简单关键词
        return {
            "rewrite_query_list": [query],
            "query_keywords_list": query.split()[:max_keywords] if query.split() else [],
            "original_query": query,
        }


async def rewrite_query(
    query: str,
    llm: Optional[Any] = None,
) -> str:
    """
    使用 LLM 改写查询以提高检索效果（已弃用，建议使用 rewrite_query_and_extract_keywords）
    
    Args:
        query: 原始查询字符串
        llm: LLM 实例，可选
        
    Returns:
        改写后的查询字符串
    """
    from langchain_core.prompts import ChatPromptTemplate
    
    if llm is None:
        from core import LLMFactory, load_llm_config
        # 使用 LLM 工厂创建模型实例
        # 从配置文件和环境变量加载配置
        llm_config = load_llm_config()
        llm = LLMFactory.create_llm(llm_config)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是检索问题改写助手，请保留关键信息并提高可检索性。只返回改写后的问题，不要添加任何解释。"),
        ("human", "{query}"),
    ])
    
    try:
        chain = prompt | llm
        response = await chain.ainvoke({"query": query})
        rewritten = response.content if hasattr(response, "content") else str(response)
        logger.info(f"问题改写: '{query}' -> '{rewritten}'")
        return rewritten.strip()
    except Exception as e:
        logger.warning(f"问题改写失败: {e}，返回原始查询")
        return query


def _get_graphrag_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """
    获取 GraphRAG 根目录和配置文件路径
    
    按以下优先级查找路径：
    1. 配置文件中的 graphrag_root/graphrag_config
    2. 环境变量 GRAPHRAG_ROOT/GRAPHRAG_CONFIG
    3. 默认路径 ./christmas 或
    
    Returns:
        Tuple[Optional[Path], Optional[Path]]: (根目录路径, 配置文件路径)
        如果路径不存在则返回 (None, None)
    """
    settings = get_settings()
    
    # 优先使用配置
    if settings.graphrag_root:
        root = Path(settings.graphrag_root).expanduser()
    else:
        # 尝试环境变量
        env_root = os.getenv("GRAPHRAG_ROOT")
        if env_root:
            root = Path(env_root).expanduser()
        else:
            # 默认路径
            root = Path(__file__).resolve().parent.parent / "christmas"
            if not root.exists():
                root = Path(__file__).resolve().parent.parent / "graphrag_project"
    
    if settings.graphrag_config:
        config_path = Path(settings.graphrag_config).expanduser()
    else:
        env_config = os.getenv("GRAPHRAG_CONFIG")
        if env_config:
            config_path = Path(env_config).expanduser()
        else:
            config_path = root / "settings.yaml"
    
    if not root.exists():
        logger.warning(f"GraphRAG root not found: {root}")
        return None, None
    
    return root, config_path


def _read_parquet_safe(path: Path) -> Optional[pd.DataFrame]:
    """
    安全读取 Parquet 文件
    
    如果文件不存在或读取失败，返回 None 而不是抛出异常。
    
    Args:
        path: Parquet 文件路径
        
    Returns:
        Optional[pd.DataFrame]: 成功时返回 DataFrame，失败时返回 None
    """
    try:
        if path.exists():
            return pd.read_parquet(path)
        return None
    except Exception as e:
        logger.warning(f"Failed to read parquet {path}: {e}")
        return None


async def graphrag_local_search(
    query: str,
    community_level: int = 3,
    root: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> str:
    """
    使用 GraphRAG 进行本地（社区级别）检索
    
    适合细节问题和具体数据查询。
    
    Args:
        query: 查询字符串
        community_level: 社区层级，默认 3
        root: GraphRAG 根目录
        config_path: GraphRAG 配置文件路径
        
    Returns:
        检索结果文本
    """
    try:
        from graphrag.api.query import local_search
        from graphrag.config.load_config import load_config
        
        # 获取路径
        if root is None or config_path is None:
            default_root, default_config = _get_graphrag_paths()
            root = root or default_root
            config_path = config_path or default_config
        
        if root is None or not root.exists():
            raise ToolError(
                "GraphRAG 根目录不存在",
                code="graphrag_root_not_found",
                details={"root": str(root) if root else None},
            )
        
        # 加载配置
        cfg = load_config(root, config_path)
        
        # 加载数据表
        output_dir = root / "output"
        entities = _read_parquet_safe(output_dir / "entities.parquet")
        communities = _read_parquet_safe(output_dir / "communities.parquet")
        community_reports = _read_parquet_safe(output_dir / "community_reports.parquet")
        text_units = _read_parquet_safe(output_dir / "text_units.parquet")
        relationships = _read_parquet_safe(output_dir / "relationships.parquet")
        covariates = _read_parquet_safe(output_dir / "covariates.parquet")
        
        if entities is None or communities is None or community_reports is None:
            raise ToolError(
                "GraphRAG 必要数据文件缺失",
                code="graphrag_missing_parquet",
                details={"root": str(root)},
            )
        
        # 执行本地搜索
        resp, _ = await local_search(
            config=cfg,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            covariates=covariates,
            community_level=community_level,
            response_type="Multiple Paragraphs",
            query=query,
        )
        
        logger.info(f"GraphRAG local search completed: {query[:50]}...")
        return resp
        
    except Exception as e:
        logger.error(f"GraphRAG local search failed: {e}")
        raise ToolError(
            "GraphRAG 本地检索失败",
            code="graphrag_local_search_failed",
            cause=e,
        ) from e


async def graphrag_global_search(
    query: str,
    root: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> str:
    """
    使用 GraphRAG 进行全局检索
    
    适合宏观问题和跨社区查询。注意：耗时较长。
    
    Args:
        query: 查询字符串
        root: GraphRAG 根目录
        config_path: GraphRAG 配置文件路径
        
    Returns:
        检索结果文本
    """
    try:
        from graphrag.api.query import global_search
        from graphrag.config.load_config import load_config
        
        # 获取路径
        if root is None or config_path is None:
            default_root, default_config = _get_graphrag_paths()
            root = root or default_root
            config_path = config_path or default_config
        
        if root is None or not root.exists():
            raise ToolError(
                "GraphRAG 根目录不存在",
                code="graphrag_root_not_found",
                details={"root": str(root) if root else None},
            )
        
        # 加载配置
        cfg = load_config(root, config_path)
        
        # 加载数据表
        output_dir = root / "output"
        entities = _read_parquet_safe(output_dir / "entities.parquet")
        communities = _read_parquet_safe(output_dir / "communities.parquet")
        community_reports = _read_parquet_safe(output_dir / "community_reports.parquet")
        
        if entities is None or communities is None or community_reports is None:
            raise ToolError(
                "GraphRAG 必要数据文件缺失",
                code="graphrag_missing_parquet",
                details={"root": str(root)},
            )
        
        # 执行全局搜索
        resp, _ = await global_search(
            config=cfg,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            community_level=None,
            dynamic_community_selection=True, # 启用 dynamic_community_selection=True，只选择与查询相关的社区
            response_type="Multiple Paragraphs",
            query=query,
        )
        
        logger.info(f"GraphRAG global search completed: {query[:50]}...")
        return resp
        
    except Exception as e:
        logger.error(f"GraphRAG global search failed: {e}")
        raise ToolError(
            "GraphRAG 全局检索失败",
            code="graphrag_global_search_failed",
            cause=e,
        ) from e


async def search_images_by_keyword(keyword: str) -> List[Dict[str, Any]]:
    """
    在 PostgreSQL 中按关键词检索图片信息
    
    Args:
        keyword: 搜索关键词
        
    Returns:
        图片信息列表
    """
    settings = get_settings()
    schema = settings.rag_media_schema
    
    try:
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row
        from psycopg import sql
        
        conn = await AsyncConnection.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password.get_secret_value(),
        )
        
        query = sql.SQL("""
            SELECT * FROM {}.rag_pictures
            WHERE description ILIKE %s
               OR keywords ILIKE %s
               OR to_tsvector('english', context::text) @@ plainto_tsquery('english', %s)
            LIMIT 10
        """).format(sql.Identifier(schema))
        
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, (f"%{keyword}%", f"%{keyword}%", keyword))
            rows = await cursor.fetchall()
        
        await conn.close()
        
        # 过滤掉大字段
        excluded_fields = {"raw_json", "created_at", "updated_at"}
        filtered_rows = [
            {k: v for k, v in row.items() if k not in excluded_fields}
            for row in rows
        ]
        
        logger.info(f"图片检索完成，找到 {len(filtered_rows)} 条结果")
        return filtered_rows
        
    except Exception as e:
        logger.error(f"图片检索失败: {e}")
        raise ToolError(
            "图片检索失败",
            code="rag_image_search_failed",
            details={"keyword": keyword, "schema": schema},
            cause=e,
        ) from e


async def search_tables_by_keyword(keyword: str) -> List[Dict[str, Any]]:
    """
    在 PostgreSQL 中按关键词检索表格信息
    
    Args:
        keyword: 搜索关键词
        
    Returns:
        表格信息列表
    """
    settings = get_settings()
    schema = settings.rag_media_schema
    
    try:
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row
        from psycopg import sql
        
        conn = await AsyncConnection.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password.get_secret_value(),
        )
        
        query = sql.SQL("""
            SELECT * FROM {}.rag_tables
            WHERE caption ILIKE %s
               OR section ILIKE %s
               OR to_tsvector('english', context::text) @@ plainto_tsquery('english', %s)
            LIMIT 10
        """).format(sql.Identifier(schema))
        
        # 第 1 个 %s → caption ILIKE %keyword%
        # 第 2 个 %s → section ILIKE %keyword%
        # 第 3 个 %s → plainto_tsquery('english', keyword)
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, (f"%{keyword}%",f"%{keyword}%",keyword))
            rows = await cursor.fetchall()
        
        await conn.close()
        
        # 过滤掉大字段
        excluded_fields = {"raw_json", "created_at", "updated_at"}
        filtered_rows = [
            {k: v for k, v in row.items() if k not in excluded_fields}
            for row in rows
        ]
        
        logger.info(f"表格检索完成，找到 {len(filtered_rows)} 条结果")
        return filtered_rows
        
    except Exception as e:
        logger.error(f"表格检索失败: {e}")
        raise ToolError(
            "表格检索失败",
            code="rag_table_search_failed",
            details={"keyword": keyword, "schema": schema},
            cause=e,
        ) from e


async def evaluate_retrieval_quality(
    query: str,
    retrieved: str,
    llm: Optional[Any] = None,
) -> Dict[str, str]:
    """
    评估检索结果质量
    
    Args:
        query: 用户查询
        retrieved: 检索结果
        llm: LLM 实例
        
    Returns:
        Dict 包含 score ("high" | "low") 和 reason
    """
    # 快速空结果判断
    if not retrieved or not str(retrieved).strip():
        return {"score": "low", "reason": "没有检索结果"}
    
    if llm is None:
        from core import LLMFactory, load_llm_config
        # 使用 LLM 工厂创建模型实例
        # 从配置文件和环境变量加载配置
        llm_config = load_llm_config()
        llm = LLMFactory.create_llm(llm_config)

    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是检索质量评估器，判断检索结果能否回答问题。仅输出 JSON，"
         "字段为 score(high/low) 与 reason(中文)。不要输出其它内容。"),
        ("human",
         "问题: {query}\n检索结果:\n{retrieved}\n请给出评分。"),
    ])
    
    try:
        chain = prompt | llm
        response = await chain.ainvoke({"query": query, "retrieved": retrieved})
        raw_text = response.content if hasattr(response, "content") else str(response)
        
        # 尝试解析 JSON
        parsed = json.loads(raw_text)
        score = str(parsed.get("score", "")).lower()
        reason = parsed.get("reason", "")
        
        if score not in {"high", "low"}:
            raise ValueError("invalid score")
        
        return {"score": score, "reason": reason}
        
    except Exception as e:
        logger.debug(f"Evaluation parse failed, fallback to heuristic: {e}")
        # 简单启发式：过短文本视为低质量
        return {
            "score": "high" if len(str(retrieved)) > 200 else "low",
            "reason": "启发式评估" if len(str(retrieved)) > 200 else "内容过少",
        }


def json_to_markdown_table(table_data: Any) -> str:
    """
    将 JSON 表格数据转换为 Markdown 格式
    
    Args:
        table_data: JSON 格式的表格数据
        
    Returns:
        Markdown 格式的表格字符串
    """
    try:
        # 支持字符串 JSON
        if isinstance(table_data, str):
            try:
                table_data = json.loads(table_data)
            except Exception:
                pass
        
        # 常见结构：{"data": [...]} 或直接就是二维数据
        if isinstance(table_data, dict) and "data" in table_data:
            df = pd.DataFrame(table_data["data"])
        else:
            df = pd.DataFrame(table_data)
        
        if df.empty:
            return "空表格"
        
        return df.to_markdown(index=False)
        
    except Exception:
        try:
            return "```json\n" + json.dumps(table_data, ensure_ascii=False, indent=2) + "\n```"
        except Exception:
            return "```json\n<unserializable table>\n```"


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 意图识别
    "identify_query_intent",
    # 查询改写
    "rewrite_query",
    "rewrite_query_and_extract_keywords",
    # GraphRAG 检索
    "graphrag_local_search",
    "graphrag_global_search",
    # PostgreSQL 检索
    "search_images_by_keyword",
    "search_tables_by_keyword",
    # 质量评估
    "evaluate_retrieval_quality",
    # 工具函数
    "json_to_markdown_table",
]

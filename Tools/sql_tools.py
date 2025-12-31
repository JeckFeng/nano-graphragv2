"""
PostgreSQL 数据库工具函数模块

提供 PostgreSQL 数据库操作相关的工具函数，包括：
- 数据库连接检查
- Schema 信息获取
- SQL 语句生成
- SQL 语法验证
- SQL 错误修正
- SQL 执行

使用示例:
    from tools.sql_tools import get_database_schema, execute_sql
    
    schema = await get_database_schema()
    result = await execute_sql("SELECT * FROM users LIMIT 10")
"""

from typing import Dict, Any, Optional, List
import json
import logging

from config.settings import get_settings
from core.tool_errors import ToolError
from Tools.tool_spec import ToolSpec


logger = logging.getLogger(__name__)

# PostGIS 空间查询指南（用于 SQL 生成提示）
POSTGIS_GUIDANCE = """
PostGIS 空间查询指南：
1. 计算两点距离：使用 ST_Distance(geom1::geography, geom2::geography) 返回米
2. 获取多边形中心点：使用 ST_Centroid(geom)
3. 判断点是否在多边形内：使用 ST_Contains(polygon, point)
4. 转换为公里：除以 1000
5. 注意：不要在结果中返回 geom 字段，会导致数据过大
"""


async def check_database_connection() -> Dict[str, Any]:
    """
    检查数据库连接是否正常
    
    Returns:
        Dict 包含:
            - connected: bool 是否连接成功
            - message: str 连接状态消息
            - database: str 数据库名称

    Raises:
        ToolError: 当连接失败时抛出。
    """
    settings = get_settings()
    
    try:
        from psycopg import AsyncConnection
        
        conn = await AsyncConnection.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password.get_secret_value(),
        )
        await conn.close()
        
        logger.info(f"数据库连接成功: {settings.db_name}")
        
        return {
            "connected": True,
            "message": "数据库连接正常",
            "database": settings.db_name,
        }
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise ToolError(
            "数据库连接失败",
            code="db_connection_failed",
            details={
                "database": settings.db_name,
                "host": settings.db_host,
                "port": settings.db_port,
            },
            cause=e,
        ) from e


async def get_database_schema(
    schemas: Optional[List[str]] = None,
    limit: Optional[int] = 200,
    offset: int = 0,
) -> str:
    """
    获取数据库结构信息。

    Args:
        schemas: 要查询的 schema 列表，默认查询 rag_document、public、agent_backend。
        limit: 每次返回的表数量上限（用于分页），默认 200，传 None 表示不分页。
        offset: 分页起始偏移量，默认 0。

    Returns:
        JSON 字符串格式的数据库结构信息，只包含表名与列名。

    Raises:
        ToolError: 当查询失败时抛出。
    """
    settings = get_settings()

    if schemas is None:
        schemas = ["rag_document", "public", "agent_backend"]
    if limit is not None and limit <= 0:
        raise ToolError(
            "分页参数 limit 必须为正数",
            code="db_schema_invalid_limit",
            details={"limit": limit},
        )
    if offset < 0:
        raise ToolError(
            "分页参数 offset 不能为负数",
            code="db_schema_invalid_offset",
            details={"offset": offset},
        )

    try:
        from psycopg import AsyncConnection

        conn = await AsyncConnection.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password.get_secret_value(),
        )

        schema_data: Dict[str, Any] = {}

        async with conn.cursor() as cursor:
            placeholders = ",".join(["%s"] * len(schemas))
            await cursor.execute(
                f"""
                SELECT nspname
                FROM pg_namespace
                WHERE nspname IN ({placeholders})
                ORDER BY nspname;
            """,
                tuple(schemas),
            )
            existing_schemas = [row[0] for row in await cursor.fetchall()]

            await cursor.execute(
                f"""
                SELECT n.nspname AS schema_name,
                       c.relname AS table_name
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relkind = 'r'
                  AND n.nspname IN ({placeholders})
                ORDER BY n.nspname, c.relname;
            """,
                tuple(schemas),
            )
            all_tables = [(row[0], row[1]) for row in await cursor.fetchall()]

            if limit is None:
                paged_tables = all_tables[offset:]
            else:
                paged_tables = all_tables[offset: offset + limit]

            for schema_name, table_name in paged_tables:
                await cursor.execute(
                    """
                    SELECT a.attname AS column_name
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = %s
                      AND n.nspname = %s
                      AND a.attnum > 0
                      AND NOT a.attisdropped
                    ORDER BY a.attnum;
                """,
                    (table_name, schema_name),
                )
                columns = [row[0] for row in await cursor.fetchall()]

                schema_tables = schema_data.setdefault(schema_name, [])
                schema_tables.append(
                    {
                        "table_name": table_name,
                        "columns": columns,
                    }
                )

        await conn.close()

        logger.info("获取数据库结构成功，共 %s 个 schema", len(existing_schemas))

        return json.dumps(schema_data, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error("获取数据库结构失败: %s", e)
        raise ToolError(
            "数据库信息查询失败",
            code="db_schema_fetch_failed",
            details={"schemas": schemas, "limit": limit, "offset": offset},
            cause=e,
        ) from e


async def generate_sql(
    database_info: str,
    task_description: str,
    llm: Optional[Any] = None,
) -> str:
    """
    根据数据库结构和任务描述生成 SQL 语句。

    Args:
        database_info: 数据库结构信息（JSON 格式）。
        task_description: 用户的查询需求描述。
        llm: LLM 实例，可选，默认使用系统配置。

    Returns:
        生成的 SQL 语句。

    Raises:
        ToolError: 当生成失败时抛出。
    """
    from langchain_core.prompts import PromptTemplate
    
    # 动态导入 LLM
    if llm is None:
        from core.llm import get_llm
        llm = get_llm()
    
    prompt = PromptTemplate(
        template="""你是一个SQL语句生成专家，同时也擅长PostGIS空间查询分析。请根据数据库信息和用户的任务描述生成SQL语句。

若涉及PostGIS空间查询分析，请严格遵循下列指南：
{POSTGIS_GUIDANCE}

数据库信息: {database_info}
任务描述: {task_description}

要求：
1. 请严格按照以下格式生成SQL语句，生成纯 SQL 语句，不要包裹任何引号或 markdown 标记。
2. 不要增加任何额外的自然语言解释，输出结果仅为 SQL 语句。
3. 不要在查询结果中返回 'geom' 字段，会导致数据过大。
4. 所有查询记录最多保留 50 条记录（使用 LIMIT 50）。
5. 请确保生成的 SQL 语句语法正确，可以直接在 PostgreSQL 中执行。

示例：
任务为"查询观影数量排名前五的电影名称"
SELECT movie_name FROM schema1.movie ORDER BY Number_views DESC LIMIT 5;
""",
        input_variables=["database_info", "task_description", "POSTGIS_GUIDANCE"],
    )
    
    try:
        chain = prompt | llm
        response = await chain.ainvoke(
            {
                "database_info": database_info,
                "task_description": task_description,
                "POSTGIS_GUIDANCE": POSTGIS_GUIDANCE,
            }
        )
        
        sql = response.content.strip() if hasattr(response, "content") else str(response).strip()
        
        # 清理可能的 markdown 标记
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()
        
        logger.info("SQL 生成成功: %s...", sql[:100])

        return sql

    except Exception as e:
        logger.error("SQL 生成失败: %s", e)
        raise ToolError(
            "SQL 生成失败",
            code="sql_generation_failed",
            cause=e,
        ) from e


async def validate_sql(sql: str) -> Dict[str, Any]:
    """
    验证 SQL 语句语法是否正确（使用 EXPLAIN）。

    Args:
        sql: 待验证的 SQL 语句。

    Returns:
        包含验证状态与错误信息的字典。

    Raises:
        ToolError: 当数据库连接失败时抛出。

    Notes:
        SQL 语法错误会以 valid=False 的结果返回，不视为工具执行失败。
    """
    settings = get_settings()

    try:
        from psycopg import AsyncConnection

        conn = await AsyncConnection.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password.get_secret_value(),
        )
    except Exception as e:
        logger.error("SQL 语法验证连接失败: %s", e)
        raise ToolError(
            "数据库连接失败",
            code="db_connection_failed",
            details={
                "database": settings.db_name,
                "host": settings.db_host,
                "port": settings.db_port,
            },
            cause=e,
        ) from e

    try:
        async with conn.cursor() as cursor:
            await cursor.execute(f"EXPLAIN {sql}")

        logger.info("SQL 语法验证通过")
        return {
            "valid": True,
            "message": "SQL 语法验证通过",
            "errors": [],
        }
    except Exception as e:
        logger.warning("SQL 语法错误: %s", e)
        return {
            "valid": False,
            "message": f"SQL 语法错误: {str(e)}",
            "errors": [str(e)],
        }
    finally:
        await conn.close()


async def correct_sql(
    sql: str,
    error_message: str,
    database_info: str,
    llm: Optional[Any] = None,
) -> str:
    """
    修正错误的 SQL 语句。

    Args:
        sql: 原始 SQL 语句。
        error_message: 错误信息。
        database_info: 数据库结构信息。
        llm: LLM 实例，可选。

    Returns:
        修正后的 SQL 语句。

    Raises:
        ToolError: 当修正失败时抛出。
    """
    from langchain_core.prompts import PromptTemplate
    
    if llm is None:
        from core.llm import get_llm
        llm = get_llm()
    
    prompt = PromptTemplate.from_template(
        """你是一位 PostgreSQL SQL 调试专家。
根据数据库结构信息、错误的 SQL 语句和错误信息，返回修正后的 SQL 语句。

数据库信息:
{database_info}

原始 SQL:
{sql}

错误信息:
{error}

要求:
1. 只返回修正后的 SQL 语句
2. 不要添加解释或注释
3. 不要使用 markdown 标记
4. 保持原始查询意图
5. 不要返回 geom 字段

修正后的 SQL:"""
    )
    
    try:
        chain = prompt | llm
        response = await chain.ainvoke({
            "sql": sql,
            "error": error_message,
            "database_info": database_info,
        })
        
        corrected = response.content.strip() if hasattr(response, "content") else str(response).strip()
        
        # 清理可能的 markdown 标记
        if corrected.startswith("```sql"):
            corrected = corrected[6:]
        if corrected.startswith("```"):
            corrected = corrected[3:]
        if corrected.endswith("```"):
            corrected = corrected[:-3]
        corrected = corrected.strip()
        
        logger.info("SQL 修正成功: %s...", corrected[:100])

        return corrected

    except Exception as e:
        logger.error("SQL 修正失败: %s", e)
        raise ToolError(
            "SQL 修正失败",
            code="sql_correction_failed",
            cause=e,
        ) from e


async def execute_sql(query: str) -> Dict[str, Any]:
    """
    执行 SQL 查询（需要权限）。

    Args:
        query: SQL 查询语句。
    Returns:
        查询结果字典，包含 rows 与 row_count。

    Raises:
        ToolError: 当执行失败时抛出。
    """
    settings = get_settings()
    
    try:
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row
        
        conn = await AsyncConnection.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password.get_secret_value(),
        )
        
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query)
            results = await cursor.fetchall()
        
        await conn.close()
        
        row_count = len(results) if results else 0
        logger.info("SQL 执行成功，返回 %s 条记录", row_count)

        return {
            "rows": results,
            "row_count": row_count,
        }

    except Exception as e:
        logger.error("SQL 执行失败: %s", e)
        raise ToolError(
            "SQL 执行失败",
            code="sql_execute_failed",
            cause=e,
        ) from e


async def execute_sql_with_retry(
    query: str,
    database_info: str,
    max_retries: int = 3,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    执行 SQL 查询，失败时自动修正并重试。

    Args:
        query: SQL 查询语句。
        database_info: 数据库结构信息（用于修正）。
        max_retries: 最大重试次数。
        llm: LLM 实例。

    Returns:
        查询结果字典，包含 rows、row_count、attempts 与 final_sql。

    Raises:
        ToolError: 当重试失败时抛出。
    """
    current_sql = query
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            result = await execute_sql(current_sql)
        except ToolError as exc:
            last_error = exc.to_dict()
            logger.warning("SQL 执行失败（第 %s 次）: %s", attempt, last_error)
            result = None

        if result:
            return {
                **result,
                "attempts": attempt,
                "final_sql": current_sql,
            }
        
        if attempt < max_retries:
            # 尝试修正 SQL
            try:
                correction = await correct_sql(
                    current_sql,
                    str(last_error),
                    database_info,
                    llm,
                )
            except ToolError as exc:
                last_error = exc.to_dict()
                break
            current_sql = correction
            logger.info("SQL 已修正，准备重试")
    
    raise ToolError(
        f"SQL 执行失败（已重试 {max_retries} 次）",
        code="sql_execute_retry_failed",
        details={
            "attempts": max_retries,
            "final_sql": current_sql,
            "last_error": last_error,
        },
    )


CHECK_DATABASE_CONNECTION_SPEC = ToolSpec(
    name="check_database_connection",
    description="检查 PostgreSQL 数据库连接状态。",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    handler=check_database_connection,
)

GET_DATABASE_SCHEMA_SPEC = ToolSpec(
    name="get_database_schema",
    description="获取数据库结构信息（表名、列名），支持分页。",
    parameters={
        "type": "object",
        "properties": {
            "schemas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "需要查询的 schema 列表，可选。",
            },
            "limit": {
                "type": "integer",
                "description": "每次返回的表数量上限（用于分页）。",
            },
            "offset": {
                "type": "integer",
                "description": "分页起始偏移量。",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    handler=get_database_schema,
)

GENERATE_SQL_SPEC = ToolSpec(
    name="generate_sql",
    description="根据数据库结构信息和任务描述生成 SQL 语句。",
    parameters={
        "type": "object",
        "properties": {
            "database_info": {
                "type": "string",
                "description": "数据库结构信息（JSON 字符串）。",
            },
            "task_description": {
                "type": "string",
                "description": "用户查询需求描述。",
            },
            "llm": {
                "type": "object",
                "description": "可选的 LLM 实例。",
            },
        },
        "required": ["database_info", "task_description"],
        "additionalProperties": False,
    },
    handler=generate_sql,
)

VALIDATE_SQL_SPEC = ToolSpec(
    name="validate_sql",
    description="验证 SQL 语句语法是否正确（使用 EXPLAIN）。",
    parameters={
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "待验证的 SQL 语句。",
            }
        },
        "required": ["sql"],
        "additionalProperties": False,
    },
    handler=validate_sql,
)

CORRECT_SQL_SPEC = ToolSpec(
    name="correct_sql",
    description="根据错误信息和数据库结构修正 SQL 语句。",
    parameters={
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "原始 SQL 语句。"},
            "error_message": {"type": "string", "description": "错误信息。"},
            "database_info": {
                "type": "string",
                "description": "数据库结构信息（JSON 字符串）。",
            },
            "llm": {"type": "object", "description": "可选的 LLM 实例。"},
        },
        "required": ["sql", "error_message", "database_info"],
        "additionalProperties": False,
    },
    handler=correct_sql,
)

EXECUTE_SQL_SPEC = ToolSpec(
    name="execute_sql",
    description="执行 SQL 查询并返回结果。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL 查询语句。"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=execute_sql,
)

EXECUTE_SQL_WITH_RETRY_SPEC = ToolSpec(
    name="execute_sql_with_retry",
    description="执行 SQL 查询，失败时自动修正并重试。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL 查询语句。"},
            "database_info": {
                "type": "string",
                "description": "数据库结构信息（JSON 字符串）。",
            },
            "max_retries": {"type": "integer", "description": "最大重试次数。"},
            "llm": {"type": "object", "description": "可选的 LLM 实例。"},
        },
        "required": ["query", "database_info"],
        "additionalProperties": False,
    },
    handler=execute_sql_with_retry,
)


__all__ = [
    "check_database_connection",
    "get_database_schema",
    "generate_sql",
    "validate_sql",
    "correct_sql",
    "execute_sql",
    "execute_sql_with_retry",
    "CHECK_DATABASE_CONNECTION_SPEC",
    "GET_DATABASE_SCHEMA_SPEC",
    "GENERATE_SQL_SPEC",
    "VALIDATE_SQL_SPEC",
    "CORRECT_SQL_SPEC",
    "EXECUTE_SQL_SPEC",
    "EXECUTE_SQL_WITH_RETRY_SPEC",
]

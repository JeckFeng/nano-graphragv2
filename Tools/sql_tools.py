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
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    获取数据库结构信息
    
    Args:
        schemas: 要查询的 schema 列表，默认查询 hazard_affected_body 和 disaster_risk
        user_id: 用户 ID，用于权限检查
        
    Returns:
        Dict 包含:
            - success: bool 是否成功
            - content: str JSON 格式的数据库结构信息
            - error: str 错误信息（失败时）
            
    """
    settings = get_settings()
    
    if schemas is None:
        schemas = ["rag_document", "public","agent_backend"]
    
    try:
        from psycopg import AsyncConnection
        
        conn = await AsyncConnection.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password.get_secret_value(),
        )
        
        schema_data = {}
        have_schemas = []
        
        async with conn.cursor() as cursor:
            # 获取存在的 schema
            placeholders = ','.join(['%s'] * len(schemas))
            await cursor.execute(f"""
                SELECT nspname 
                FROM pg_namespace 
                WHERE nspname IN ({placeholders})
                ORDER BY nspname;
            """, tuple(schemas))
            existing_schemas = [row[0] for row in await cursor.fetchall()]
            
            for schema in existing_schemas:
                have_schemas.append(schema)
                
                # 获取表信息
                await cursor.execute("""
                    SELECT c.relname AS table_name,
                           COALESCE(obj_description(c.oid, 'pg_class'), '无注释') AS table_comment
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relkind = 'r'
                      AND n.nspname = %s
                    ORDER BY table_name;
                """, (schema,))
                tables = await cursor.fetchall()
                
                schema_tables = []
                for table_name, table_comment in tables:
                    # 获取列信息
                    await cursor.execute("""
                        SELECT a.attname AS column_name,
                               COALESCE(col_description(a.attrelid, a.attnum), '无注释') AS column_comment
                        FROM pg_attribute a
                        JOIN pg_class c ON c.oid = a.attrelid
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relname = %s
                          AND n.nspname = %s
                          AND a.attnum > 0
                          AND NOT a.attisdropped
                        ORDER BY a.attnum;
                    """, (table_name, schema))
                    columns = await cursor.fetchall()
                    
                    column_list = [
                        {"name": col_name, "column_comment": col_comment}
                        for col_name, col_comment in columns
                    ]
                    
                    schema_tables.append({
                        "table_name": table_name,
                        "table_comment": table_comment,
                        "columns": column_list,
                    })
                
                schema_data[schema] = schema_tables
        
        schema_data["have_schemas"] = have_schemas
        
        await conn.close()
        
        logger.info(f"获取数据库结构成功，共 {len(have_schemas)} 个 schema")
        
        return {
            "success": True,
            "content": json.dumps(schema_data, ensure_ascii=False, indent=2),
            "error": None,
        }
        
    except Exception as e:
        logger.error(f"获取数据库结构失败: {e}")
        raise ToolError(
            "数据库信息查询失败",
            code="db_schema_fetch_failed",
            details={"schemas": schemas},
            cause=e,
        ) from e


async def generate_sql(
    database_info: str,
    task_desc: str,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    根据数据库结构和任务描述生成 SQL 语句
    
    Args:
        database_info: 数据库结构信息（JSON 格式）
        task_desc: 用户的查询需求描述
        llm: LLM 实例，可选，默认使用 My_LLM
        
    Returns:
        Dict 包含:
            - success: bool 是否成功
            - content: str 生成的 SQL 语句
            - error: str 错误信息（失败时）
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
任务描述: {task_desc}

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
        input_variables=["database_info", "task_desc", "POSTGIS_GUIDANCE"],
    )
    
    try:
        chain = prompt | llm
        response = await chain.ainvoke({
            "database_info": database_info,
            "task_desc": task_desc,
            "POSTGIS_GUIDANCE": POSTGIS_GUIDANCE,
        })
        
        sql = response.content.strip() if hasattr(response, "content") else str(response).strip()
        
        # 清理可能的 markdown 标记
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()
        
        logger.info(f"SQL 生成成功: {sql[:100]}...")
        
        return {
            "success": True,
            "content": sql,
            "error": None,
        }
        
    except Exception as e:
        logger.error(f"SQL 生成失败: {e}")
        raise ToolError(
            "SQL 生成失败",
            code="sql_generation_failed",
            cause=e,
        ) from e


async def validate_sql(sql: str) -> Dict[str, Any]:
    """
    验证 SQL 语句语法是否正确（使用 EXPLAIN）
    
    Args:
        sql: 待验证的 SQL 语句
        
    Returns:
        Dict 包含:
            - valid: bool 是否有效
            - message: str 验证结果消息
            - errors: List[str] 错误列表
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
        
        async with conn.cursor() as cursor:
            await cursor.execute(f"EXPLAIN {sql}")
        
        await conn.close()
        
        logger.info("SQL 语法验证通过")
        
        return {
            "valid": True,
            "message": "SQL 语法验证通过",
            "errors": [],
        }
        
    except Exception as e:
        logger.warning(f"SQL 语法错误: {e}")
        return {
            "valid": False,
            "message": f"SQL 语法错误: {str(e)}",
            "errors": [str(e)],
        }


async def correct_sql(
    sql: str,
    error_message: str,
    database_info: str,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    修正错误的 SQL 语句
    
    Args:
        sql: 原始 SQL 语句
        error_message: 错误信息
        database_info: 数据库结构信息
        llm: LLM 实例，可选
        
    Returns:
        Dict 包含:
            - success: bool 是否成功
            - content: str 修正后的 SQL 语句
            - error: str 错误信息（失败时）
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
        
        logger.info(f"SQL 修正成功: {corrected[:100]}...")
        
        return {
            "success": True,
            "content": corrected,
            "error": None,
        }
        
    except Exception as e:
        logger.error(f"SQL 修正失败: {e}")
        raise ToolError(
            "SQL 修正失败",
            code="sql_correction_failed",
            cause=e,
        ) from e


async def execute_sql(query: str, user_id: str = "default") -> Dict[str, Any]:
    """
    执行 SQL 查询（需要权限）
    
    Args:
        query: SQL 查询语句
        user_id: 用户 ID，用于权限检查
        
    Returns:
        Dict 包含:
            - success: bool 是否成功
            - content: str JSON 格式的查询结果
            - error: str 错误信息（失败时）
            - row_count: int 返回行数
            
    Raises:
        PermissionError: 用户没有 execute_sql 工具权限
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
        logger.info(f"SQL 执行成功，返回 {row_count} 条记录")
        
        return {
            "success": True,
            "content": json.dumps(results, default=str, ensure_ascii=False),
            "error": None,
            "row_count": row_count,
        }
        
    except Exception as e:
        logger.error(f"SQL 执行失败: {e}")
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
    执行 SQL 查询，失败时自动修正并重试
    
    Args:
        query: SQL 查询语句
        database_info: 数据库结构信息（用于修正）
        max_retries: 最大重试次数
        llm: LLM 实例
        
    Returns:
        Dict 包含:
            - success: bool 是否成功
            - content: str 查询结果
            - error: str 错误信息（失败时）
            - attempts: int 尝试次数
            - final_sql: str 最终执行的 SQL
    """
    current_sql = query
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            result = await execute_sql(current_sql)
        except ToolError as exc:
            last_error = exc.to_dict()
            logger.warning(f"SQL 执行失败（第 {attempt} 次）: {last_error}")
            result = None
        
        if result and result.get("success"):
            return {
                **result,
                "attempts": attempt,
                "final_sql": current_sql,
            }
        
        if result:
            last_error = result.get("error")
            logger.warning(f"SQL 执行失败（第 {attempt} 次）: {last_error}")
        
        if attempt < max_retries:
            # 尝试修正 SQL
            try:
                correction = await correct_sql(current_sql, str(last_error), database_info, llm)
            except ToolError as exc:
                last_error = exc.to_dict()
                break
            if correction.get("success"):
                current_sql = correction["content"]
                logger.info("SQL 已修正，准备重试")
            else:
                last_error = correction.get("error")
                break
    
    raise ToolError(
        f"SQL 执行失败（已重试 {max_retries} 次）",
        code="sql_execute_retry_failed",
        details={
            "attempts": max_retries,
            "final_sql": current_sql,
            "last_error": last_error,
        },
    )

"""PostgreSQL 数据库连接测试工具（独立脚本）"""
from __future__ import annotations

import os
import time
import psycopg
from dotenv import load_dotenv

load_dotenv()


class DbWriteError(RuntimeError):
    """数据库写入异常。"""


def assert_postgres_connection(
    *,
    user: str | None,
    password: str | None,
    host: str | None,
    dbname: str | None,
    port: str | None,
    options: str | None = None,
) -> None:
    """
    测试 PostgreSQL 连通性；失败抛出 DbWriteError。
    - 仅执行简单的连接与 SELECT 1，保持轻量。
    """
    try:
        with psycopg.connect(
            user=user,
            password=password,
            host=host,
            dbname=dbname,
            port=port,
            options=options,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
    except Exception as e:
        raise DbWriteError(f"PostgreSQL 连接失败：{e}") from e


def database_connection():
    """测试 PostgreSQL 数据库连接 - CLI 版，复用库级纯函数。"""
    print("=" * 60)
    print("🚀 PostgreSQL 数据库连接测试程序")
    print("=" * 60)

    start_time = time.time()
    try:
        print("\n📡 尝试连接数据库...")
        assert_postgres_connection(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("LANGGRAPH_MEMORY_DB"),
            port=os.getenv("DB_PORT"),
        )
        end_time = time.time()
        print("✅ 连接成功!")
        print(f"⏱️ 连接耗时: {round((end_time - start_time) * 1000, 2)} 毫秒")
        print(f"🔒 连接参数: user={os.getenv('DB_USER')}, dbname={os.getenv('LANGGRAPH_MEMORY_DB')}")
        print("\n✅ 所有测试通过! 数据库连接正常")
        return True

    except DbWriteError as e:
        end_time = time.time()
        print("\n❌ 连接失败!")
        print(f"⏱️ 尝试耗时: {round((end_time - start_time) * 1000, 2)} 毫秒")
        print(f"⚠️ 错误详情: {e}")
        return False
    except Exception as e:
        end_time = time.time()
        print("\n❌ 连接失败!")
        print(f"⏱️ 尝试耗时: {round((end_time - start_time) * 1000, 2)} 毫秒")
        print(f"⚠️ 异常信息: {e}")
        return False
    finally:
        print("\n" + "=" * 60)


if __name__ == "__main__":
    database_connection()

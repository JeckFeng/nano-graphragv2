"""
应用配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理，支持从环境变量和 .env 文件读取配置。

使用示例:
    from config import get_settings
    
    settings = get_settings()
    print(settings.database_url)  # 动态构建的连接字符串
    print(settings.deepseek_api_key)
"""

from functools import lru_cache
from typing import Literal, Optional
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    应用配置类
    
    所有配置项均可通过环境变量或 .env 文件设置。
    环境变量名称为大写形式（如 DB_HOST）。
    
    Attributes:
        db_host: PostgreSQL 数据库主机
        db_port: PostgreSQL 数据库端口
        deepseek_api_key: DeepSeek API 密钥（默认 LLM）
        environment: 运行环境 (development/staging/production)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略未定义的环境变量
    )
    
    # -------------------------------------------------------------------------
    # 数据库配置（分离字段，不再使用 DATABASE_URL）
    # -------------------------------------------------------------------------
    db_host: str = Field(
        default="localhost",
        description="PostgreSQL 主机地址",
    )
    db_port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="PostgreSQL 端口",
    )
    db_name: str = Field(
        default="langgraph_memory",
        description="PostgreSQL 数据库名",
    )
    db_user: str = Field(
        default="postgres",
        description="PostgreSQL 用户名",
    )
    db_password: SecretStr = Field(
        default=SecretStr(""),
        description="PostgreSQL 密码",
    )
    db_pool_min_size: int = Field(
        default=5,
        ge=1,
        description="数据库连接池最小连接数",
    )
    db_pool_max_size: int = Field(
        default=20,
        ge=1,
        description="数据库连接池最大连接数",
    )
    
    # -------------------------------------------------------------------------
    # DeepSeek 配置（默认 LLM 提供商）
    # -------------------------------------------------------------------------
    deepseek_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="DeepSeek API 密钥",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API 基础 URL",
    )
    
    
    # -------------------------------------------------------------------------
    # DashScope 配置（阿里通义千问）
    # -------------------------------------------------------------------------
    dashscope_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="DashScope API 密钥（阿里通义千问）",
    )
    
    # -------------------------------------------------------------------------
    # 默认 LLM 配置
    # -------------------------------------------------------------------------
    default_llm_model: str = Field(
        default="deepseek-chat",
        description="默认使用的 LLM 模型",
    )
    default_llm_provider: str = Field(
        default="deepseek",
        description="默认使用的 LLM 提供商 (openai/deepseek/dashscope/ollama/vllm)",
    )
    
    
    # -------------------------------------------------------------------------
    # 应用配置
    # -------------------------------------------------------------------------
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="运行环境",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="日志级别",
    )
    
    
    # -------------------------------------------------------------------------
    # 外部服务配置
    # -------------------------------------------------------------------------
    amap_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="高德地图 API 密钥",
    )
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j 数据库连接 URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j 用户名",
    )
    neo4j_password: SecretStr = Field(
        default=SecretStr(""),
        description="Neo4j 密码",
    )
    neo4j_timeout: float = Field(
        default=100.0,
        ge=1.0,
        description="Neo4j 连接超时时间（秒）",
    )

    
    # -------------------------------------------------------------------------
    # GraphRAG 配置
    # -------------------------------------------------------------------------
    graphrag_root: str = Field(
        default="./christmas",
        description="GraphRAG 根目录路径",
    )
    graphrag_config: str = Field(
        default="",
        description="GraphRAG 配置文件路径",
    )
    graphrag_community_level: int = Field(
        default=4,
        ge=0,
        le=10,
        description="GraphRAG 社区层级",
    )
    rag_media_schema: str = Field(
        default="rag_document",
        description="RAG 媒体表所在的数据库 Schema",
    )
    
    # -------------------------------------------------------------------------
    # 人工审核全局配置
    # -------------------------------------------------------------------------
    approval_timeout_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="审核超时时间（秒），默认 3600 秒（1小时）",
    )
    approval_auto_approve_after_timeout: bool = Field(
        default=False,
        description="超时后是否自动批准，默认 False（拒绝）",
    )
    
    # -------------------------------------------------------------------------
    # 计算属性
    # -------------------------------------------------------------------------
    @computed_field
    @property
    def database_url(self) -> str:
        """
        动态构建数据库连接字符串
        
        从分离的配置项构建 PostgreSQL 连接 URL。
        密码会进行 URL 编码以处理特殊字符。
        
        Returns:
            str: PostgreSQL 连接字符串
        """
        password = self.db_password.get_secret_value()
        # URL 编码密码以处理特殊字符
        encoded_password = quote_plus(password) if password else ""
        return f"postgresql://{self.db_user}:{encoded_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    # -------------------------------------------------------------------------
    # 辅助属性
    # -------------------------------------------------------------------------
    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.environment == "production"
    
    @property
    def langsmith_enabled(self) -> bool:
        """LangSmith 是否启用"""
        return bool(self.langsmith_api_key.get_secret_value()) and self.langsmith_tracing_v2
    
    @property
    def has_deepseek(self) -> bool:
        """DeepSeek 是否可用"""
        return bool(self.deepseek_api_key.get_secret_value())
    
    @property
    def has_openai(self) -> bool:
        """OpenAI 是否可用"""
        return bool(self.openai_api_key.get_secret_value())
    
    @property
    def has_dashscope(self) -> bool:
        """DashScope 是否可用"""
        return bool(self.dashscope_api_key.get_secret_value())
    
    @property
    def has_ollama(self) -> bool:
        """Ollama 是否配置（通过 base_url 判断）"""
        return bool(self.ollama_base_url and self.ollama_model_name)
    
    @property
    def has_vllm(self) -> bool:
        """vLLM 是否配置"""
        return bool(self.vllm_base_url and self.vllm_model_name)


@lru_cache
def get_settings() -> Settings:
    """
    获取应用配置单例
    
    使用 lru_cache 确保配置只加载一次，提高性能。
    
    Returns:
        Settings: 应用配置实例
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.database_url)
        postgresql://postgres:***@localhost:5432/nano_graph
    """
    return Settings()


def reset_settings_cache() -> None:
    """
    重置配置缓存
    
    在配置变更后调用此方法清除缓存，使新配置生效。
    """
    get_settings.cache_clear()

"""
LLM 配置模型定义

使用 Pydantic 定义 LLM 和 Embedding 的配置模型。

使用示例:
    from core.llm_config import LLMConfig, EmbeddingConfig
    from core.llm_providers import LLMProvider
    
    config = LLMConfig(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="sk-xxx",
    )
"""

from typing import Optional

from pydantic import BaseModel, Field, SecretStr, field_validator

from .llm_providers import LLMProvider, EmbeddingProvider


class LLMConfig(BaseModel):
    """
    LLM 配置模型
    
    定义创建 LLM 实例所需的所有配置参数。
    
    Attributes:
        provider: LLM 提供商
        model_name: 模型名称
        api_key: API 密钥（部分提供商需要）
        base_url: API 基础 URL（可选，使用默认值）
        temperature: 温度参数，控制输出随机性
        max_tokens: 最大输出 token 数
        timeout: 请求超时时间（秒）
        top_p: 核采样参数
        frequency_penalty: 频率惩罚参数
        presence_penalty: 存在惩罚参数
    """
    
    provider: LLMProvider = Field(
        default=LLMProvider.DEEPSEEK,
        description="LLM 提供商",
    )
    model_name: str = Field(
        default="deepseek-chat",
        description="模型名称",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API 密钥",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="API 基础 URL（留空使用提供商默认值）",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="温度参数（0-2），控制输出随机性",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="最大输出 token 数",
    )
    timeout: Optional[float] = Field(
        default=60.0,
        ge=1.0,
        description="请求超时时间（秒）",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="核采样参数",
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="频率惩罚参数",
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="存在惩罚参数",
    )
    
    @field_validator("provider", mode="before")
    @classmethod
    def validate_provider(cls, v):
        """验证并转换 provider"""
        if isinstance(v, str):
            return LLMProvider.from_string(v)
        return v
    
    def get_api_key_value(self) -> Optional[str]:
        """
        获取 API Key 的明文值
        
        Returns:
            Optional[str]: API Key 明文，如果未设置则返回 None
        """
        if self.api_key is None:
            return None
        return self.api_key.get_secret_value()
    
    def model_dump_safe(self) -> dict:
        """
        安全导出配置（隐藏敏感信息）
        
        Returns:
            dict: 不包含敏感信息的配置字典
        """
        data = self.model_dump(exclude={"api_key"})
        data["has_api_key"] = self.api_key is not None
        return data


class EmbeddingConfig(BaseModel):
    """
    Embedding 模型配置
    
    定义创建 Embedding 模型实例所需的配置参数。
    
    Attributes:
        provider: Embedding 提供商
        model_name: 模型名称
        api_key: API 密钥（部分提供商需要）
        base_url: API 基础 URL
        dimensions: 嵌入向量维度（可选）
        batch_size: 批处理大小
    """
    
    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OLLAMA,
        description="Embedding 提供商",
    )
    model_name: str = Field(
        default="nomic-embed-text",
        description="模型名称",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API 密钥",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="API 基础 URL（留空使用提供商默认值）",
    )
    dimensions: Optional[int] = Field(
        default=None,
        ge=1,
        description="嵌入向量维度（部分模型支持）",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="批处理大小",
    )
    
    @field_validator("provider", mode="before")
    @classmethod
    def validate_provider(cls, v):
        """验证并转换 provider"""
        if isinstance(v, str):
            return EmbeddingProvider.from_string(v)
        return v
    
    def get_api_key_value(self) -> Optional[str]:
        """
        获取 API Key 的明文值
        
        Returns:
            Optional[str]: API Key 明文，如果未设置则返回 None
        """
        if self.api_key is None:
            return None
        return self.api_key.get_secret_value()
    
    def model_dump_safe(self) -> dict:
        """
        安全导出配置（隐藏敏感信息）
        
        Returns:
            dict: 不包含敏感信息的配置字典
        """
        data = self.model_dump(exclude={"api_key"})
        data["has_api_key"] = self.api_key is not None
        return data


# =============================================================================
# 配置构建辅助函数
# =============================================================================

def create_llm_config_from_settings() -> LLMConfig:
    """
    从 Settings 创建默认 LLM 配置
    
    按优先级选择：DeepSeek > Ollama > OpenAI
    
    Returns:
        LLMConfig: 根据 Settings 创建的配置
    """
    from config import get_settings
    
    settings = get_settings()
    
    # 根据默认 provider 设置创建配置
    provider_str = settings.default_llm_provider.lower()
    
    if provider_str == "deepseek" and settings.has_deepseek:
        return LLMConfig(
            provider=LLMProvider.DEEPSEEK,
            model_name=settings.default_llm_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    elif provider_str == "ollama" and settings.has_ollama:
        return LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name=settings.ollama_model_name,
            base_url=settings.ollama_base_url,
        )
    elif provider_str == "dashscope" and settings.has_dashscope:
        return LLMConfig(
            provider=LLMProvider.DASHSCOPE,
            model_name=settings.default_llm_model,
            api_key=settings.dashscope_api_key,
        )
    elif provider_str == "vllm" and settings.has_vllm:
        return LLMConfig(
            provider=LLMProvider.VLLM,
            model_name=settings.vllm_model_name,
            base_url=settings.vllm_base_url,
        )
    elif provider_str == "openai" and settings.has_openai:
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name=settings.default_llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )
    
    # 自动选择可用的 provider
    if settings.has_deepseek:
        return LLMConfig(
            provider=LLMProvider.DEEPSEEK,
            model_name="deepseek-chat",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    elif settings.has_ollama:
        return LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name=settings.ollama_model_name,
            base_url=settings.ollama_base_url,
        )
    elif settings.has_openai:
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name=settings.default_llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )
    
    # 回退：使用 OpenAI 配置（即使没有 API Key）
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=settings.default_llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
    )


def create_embedding_config_from_settings() -> EmbeddingConfig:
    """
    从 Settings 创建默认 Embedding 配置
    
    按优先级选择：Ollama > OpenAI > DashScope
    
    Returns:
        EmbeddingConfig: 根据 Settings 创建的配置
    """
    from config import get_settings
    
    settings = get_settings()
    
    # 优先使用 Ollama 本地 Embedding
    if settings.has_ollama and settings.ollama_embedding_model:
        return EmbeddingConfig(
            provider=EmbeddingProvider.OLLAMA,
            model_name=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
    
    # 其次使用 OpenAI
    if settings.has_openai:
        return EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model_name="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
    
    # 最后使用 DashScope
    if settings.has_dashscope:
        return EmbeddingConfig(
            provider=EmbeddingProvider.DASHSCOPE,
            model_name="text-embedding-v2",
            api_key=settings.dashscope_api_key,
        )
    
    # 回退：使用 Ollama 默认配置
    return EmbeddingConfig(
        provider=EmbeddingProvider.OLLAMA,
        model_name=settings.ollama_embedding_model or "nomic-embed-text",
        base_url=settings.ollama_base_url,
    )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "LLMConfig",
    "EmbeddingConfig",
    "create_llm_config_from_settings",
    "create_embedding_config_from_settings",
]


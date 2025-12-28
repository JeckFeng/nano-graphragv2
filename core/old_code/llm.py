"""
LLM 初始化模块

提供统一的 LLM 实例获取接口。

本模块是对 LLMFactory 的便捷封装，提供简单易用的 API：
- get_llm(): 获取默认 LLM 实例
- get_llm_by_name(): 根据模型名称获取 LLM 实例
- get_embeddings(): 获取 Embedding 模型实例

使用示例:
    from core.llm import get_llm, get_llm_by_name, get_embeddings
    
    # 获取默认 LLM（使用 settings 中的配置）
    llm = get_llm()
    
    # 获取指定提供商和模型的 LLM
    llm = get_llm(provider="ollama", model="llama3")
    
    # 获取 Embedding 模型
    embeddings = get_embeddings()
    
    # 在 Agent 中使用
    agent = create_react_agent(get_llm(), tools=tools)
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from .llm_factory import LLMFactory, create_llm as _factory_create_llm
from .llm_config import LLMConfig, EmbeddingConfig, create_llm_config_from_settings
from .llm_providers import LLMProvider, EmbeddingProvider

logger = logging.getLogger(__name__)


# =============================================================================
# 主要接口
# =============================================================================

def get_llm(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs,
) -> BaseChatModel:
    """
    获取 LLM 实例
    
    如果不指定参数，使用配置中的默认模型和提供商。
    
    Args:
        model: 模型名称，默认使用 settings.default_llm_model
        provider: 提供商名称（openai/deepseek/dashscope/ollama/vllm）
        **kwargs: 其他配置参数（temperature, max_tokens 等）
        
    Returns:
        BaseChatModel: LLM 实例
        
    Example:
        >>> llm = get_llm()  # 使用默认模型
        >>> llm = get_llm("deepseek-chat")  # 指定模型
        >>> llm = get_llm(provider="ollama", model="llama3")  # 指定提供商和模型
    """
    # 如果没有指定任何参数，使用缓存的默认配置
    if model is None and provider is None and not kwargs:
        return LLMFactory.create_llm_cached()
    
    # 构建配置
    from config import get_settings
    settings = get_settings()
    
    # 确定提供商
    if provider:
        llm_provider = LLMProvider.from_string(provider)
    else:
        llm_provider = LLMProvider.from_string(settings.default_llm_provider)
    
    # 确定模型名称
    model_name = model or settings.default_llm_model
    
    # 确定 API Key 和 base_url
    api_key = None
    base_url = None
    
    if llm_provider == LLMProvider.DEEPSEEK:
        api_key = settings.deepseek_api_key
        base_url = settings.deepseek_base_url
    elif llm_provider == LLMProvider.OPENAI:
        api_key = settings.openai_api_key
        base_url = settings.openai_api_base
    elif llm_provider == LLMProvider.DASHSCOPE:
        api_key = settings.dashscope_api_key
    elif llm_provider == LLMProvider.OLLAMA:
        base_url = settings.ollama_base_url
        model_name = model or settings.ollama_model_name
    elif llm_provider == LLMProvider.VLLM:
        base_url = settings.vllm_base_url
        model_name = model or settings.vllm_model_name
    
    config = LLMConfig(
        provider=llm_provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
    
    return LLMFactory.create_llm(config)


@lru_cache(maxsize=8)
def get_llm_by_name(model: str, provider: Optional[str] = None) -> BaseChatModel:
    """
    根据模型名称获取 LLM 实例（带缓存）
    
    使用 LRU 缓存避免重复创建相同模型的实例。
    
    Args:
        model: 模型名称
        provider: 提供商名称（可选，会自动推断）
        
    Returns:
        BaseChatModel: LLM 实例
        
    Example:
        >>> llm = get_llm_by_name("gpt-4o")
        >>> llm = get_llm_by_name("llama3", provider="ollama")
        >>> response = llm.invoke("Hello!")
    """
    # 自动推断 provider
    if provider is None:
        provider = _infer_provider(model)
    
    return get_llm(model=model, provider=provider)


def _infer_provider(model: str) -> str:
    """
    根据模型名称推断提供商
    
    Args:
        model: 模型名称
        
    Returns:
        str: 提供商名称
    """
    model_lower = model.lower()
    
    if model_lower.startswith("gpt-") or model_lower.startswith("o1"):
        return "openai"
    elif model_lower.startswith("deepseek"):
        return "deepseek"
    elif model_lower.startswith("qwen"):
        # 检查是否是 Ollama 的 qwen
        from config import get_settings
        settings = get_settings()
        if settings.has_dashscope:
            return "dashscope"
        elif settings.has_ollama:
            return "ollama"
        return "dashscope"
    elif model_lower.startswith("llama") or model_lower.startswith("mistral"):
        return "ollama"
    elif model_lower.startswith("claude"):
        # Claude 暂时不支持，使用 OpenAI 兼容接口
        return "openai"
    else:
        # 默认使用配置的 provider
        from config import get_settings
        return get_settings().default_llm_provider


def get_embeddings(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs,
) -> Embeddings:
    """
    获取 Embedding 模型实例
    
    Args:
        model: 模型名称
        provider: 提供商名称（openai/ollama/dashscope）
        **kwargs: 其他配置参数
        
    Returns:
        Embeddings: Embedding 模型实例
        
    Example:
        >>> embeddings = get_embeddings()  # 使用默认配置
        >>> embeddings = get_embeddings("nomic-embed-text", provider="ollama")
    """
    if model is None and provider is None and not kwargs:
        return LLMFactory.create_embeddings_cached()
    
    from config import get_settings
    settings = get_settings()
    
    # 确定提供商
    if provider:
        embed_provider = EmbeddingProvider.from_string(provider)
    else:
        # 默认使用 Ollama
        embed_provider = EmbeddingProvider.OLLAMA
    
    # 确定模型名称
    if model is None:
        if embed_provider == EmbeddingProvider.OLLAMA:
            model = settings.ollama_embedding_model
        elif embed_provider == EmbeddingProvider.OPENAI:
            model = "text-embedding-3-small"
        else:
            model = "text-embedding-v2"
    
    # 确定 API Key 和 base_url
    api_key = None
    base_url = None
    
    if embed_provider == EmbeddingProvider.OPENAI:
        api_key = settings.openai_api_key
    elif embed_provider == EmbeddingProvider.OLLAMA:
        base_url = settings.ollama_base_url
    elif embed_provider == EmbeddingProvider.DASHSCOPE:
        api_key = settings.dashscope_api_key
    
    config = EmbeddingConfig(
        provider=embed_provider,
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
    
    return LLMFactory.create_embeddings(config)


# =============================================================================
# 缓存管理
# =============================================================================

def reset_llm_cache() -> None:
    """
    重置 LLM 缓存
    
    在配置变更后调用此方法清除缓存。
    """
    get_llm_by_name.cache_clear()
    LLMFactory.clear_cache()
    logger.info("LLM cache cleared")


# =============================================================================
# 兼容性别名（便于迁移）
# =============================================================================

class _LazyLLM:
    """
    延迟加载的 LLM 包装器
    
    用于兼容旧代码中可能使用的 My_LLM 变量名。
    """
    _instance: Optional[BaseChatModel] = None
    
    def __getattr__(self, name: str):
        if self._instance is None:
            self._instance = get_llm()
        return getattr(self._instance, name)
    
    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = get_llm()
        return self._instance(*args, **kwargs)
    
    def reset(self):
        """重置实例，下次访问时重新创建"""
        self._instance = None


# 兼容旧代码
My_LLM = _LazyLLM()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 主要接口
    "get_llm",
    "get_llm_by_name",
    "get_embeddings",
    "reset_llm_cache",
    # 兼容性导出
    "My_LLM",
    # 重导出工厂类（高级用法）
    "LLMFactory",
    "LLMConfig",
    "EmbeddingConfig",
    "LLMProvider",
    "EmbeddingProvider",
]

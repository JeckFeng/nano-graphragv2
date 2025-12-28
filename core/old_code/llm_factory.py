"""
LLM 工厂模块

提供统一的 LLM 和 Embedding 模型创建接口。

使用示例:
    from core.llm_factory import LLMFactory
    from core.llm_config import LLMConfig
    from core.llm_providers import LLMProvider
    
    # 使用默认配置
    llm = LLMFactory.create_llm()
    
    # 使用自定义配置
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3",
        base_url="http://localhost:11434",
    )
    llm = LLMFactory.create_llm(config)
    
    # 创建 Embedding
    embeddings = LLMFactory.create_embeddings()
"""

import logging
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from .llm_config import (
    LLMConfig,
    EmbeddingConfig,
    create_llm_config_from_settings,
    create_embedding_config_from_settings,
)
from .llm_providers import (
    LLMProvider,
    EmbeddingProvider,
    LLM_PROVIDER_INFO,
    EMBEDDING_PROVIDER_INFO,
)

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    LLM 工厂类
    
    提供统一的 LLM 和 Embedding 模型创建接口，支持多种提供商：
    - OpenAI (GPT-4o, GPT-4, GPT-3.5)
    - DeepSeek (deepseek-chat, deepseek-coder)
    - DashScope (qwen-max, qwen-plus)
    - Ollama (本地模型)
    - vLLM (本地高性能推理)
    
    Example:
        >>> llm = LLMFactory.create_llm()
        >>> response = llm.invoke("Hello!")
    """
    
    # LLM 实例缓存
    _llm_cache: Dict[str, BaseChatModel] = {}
    _embeddings_cache: Dict[str, Embeddings] = {}
    
    @classmethod
    def create_llm(cls, config: Optional[LLMConfig] = None) -> BaseChatModel:
        """
        根据配置创建 LLM 实例
        
        Args:
            config: LLM 配置，如果为 None 则使用默认配置
            
        Returns:
            BaseChatModel: LLM 实例
            
        Raises:
            ValueError: 如果提供商不支持
            ImportError: 如果缺少必要的依赖包
        """
        if config is None:
            config = create_llm_config_from_settings()
        
        logger.info(f"Creating LLM: provider={config.provider.value}, model={config.model_name}")
        
        if config.provider == LLMProvider.OPENAI:
            return cls._create_openai_llm(config)
        elif config.provider == LLMProvider.DEEPSEEK:
            return cls._create_deepseek_llm(config)
        elif config.provider == LLMProvider.DASHSCOPE:
            return cls._create_dashscope_llm(config)
        elif config.provider == LLMProvider.OLLAMA:
            return cls._create_ollama_llm(config)
        elif config.provider == LLMProvider.VLLM:
            return cls._create_vllm_llm(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    @classmethod
    def create_llm_cached(cls, config: Optional[LLMConfig] = None) -> BaseChatModel:
        """
        创建或获取缓存的 LLM 实例
        
        使用配置的哈希值作为缓存键，相同配置返回相同实例。
        
        Args:
            config: LLM 配置
            
        Returns:
            BaseChatModel: LLM 实例
        """
        if config is None:
            config = create_llm_config_from_settings()
        
        # 生成缓存键（不包含敏感信息）
        cache_key = f"{config.provider.value}:{config.model_name}:{config.base_url}:{config.temperature}"
        
        if cache_key not in cls._llm_cache:
            cls._llm_cache[cache_key] = cls.create_llm(config)
        
        return cls._llm_cache[cache_key]
    
    @classmethod
    def _create_openai_llm(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建 OpenAI LLM
        
        Args:
            config: LLM 配置
            
        Returns:
            BaseChatModel: ChatOpenAI 实例
        """
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=config.model_name,
            api_key=config.get_api_key_value(),
            base_url=config.base_url or LLM_PROVIDER_INFO[LLMProvider.OPENAI]["default_base_url"],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
        )
    
    @classmethod
    def _create_deepseek_llm(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建 DeepSeek LLM (OpenAI 兼容接口)
        
        Args:
            config: LLM 配置
            
        Returns:
            BaseChatModel: ChatOpenAI 实例（使用 DeepSeek API）
        """
        from langchain_openai import ChatOpenAI
        
        base_url = config.base_url or LLM_PROVIDER_INFO[LLMProvider.DEEPSEEK]["default_base_url"]
        
        return ChatOpenAI(
            model=config.model_name,
            api_key=config.get_api_key_value(),
            base_url=base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
        )
    
    @classmethod
    def _create_dashscope_llm(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建 DashScope LLM (阿里通义千问) - 使用 OpenAI 兼容接口
        
        Args:
            config: LLM 配置
            
        Returns:
            BaseChatModel: ChatOpenAI 实例（使用 DashScope API）
        """
        from langchain_openai import ChatOpenAI
        
        base_url = config.base_url or LLM_PROVIDER_INFO[LLMProvider.DASHSCOPE]["default_base_url"]
        
        return ChatOpenAI(
            model=config.model_name,
            api_key=config.get_api_key_value(),
            base_url=base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
        )
    
    @classmethod
    def _create_ollama_llm(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建 Ollama LLM (本地部署)
        
        Args:
            config: LLM 配置
            
        Returns:
            BaseChatModel: ChatOllama 实例
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "请安装 langchain-ollama: pip install langchain-ollama"
            )
        
        base_url = config.base_url or LLM_PROVIDER_INFO[LLMProvider.OLLAMA]["default_base_url"]
        
        return ChatOllama(
            model=config.model_name,
            base_url=base_url,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    
    @classmethod
    def _create_vllm_llm(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建 vLLM LLM (本地高性能推理)
        
        vLLM 使用 OpenAI 兼容接口，不需要真实 API Key。
        
        Args:
            config: LLM 配置
            
        Returns:
            BaseChatModel: ChatOpenAI 实例（使用 vLLM API）
        """
        from langchain_openai import ChatOpenAI
        
        base_url = config.base_url or LLM_PROVIDER_INFO[LLMProvider.VLLM]["default_base_url"]
        
        return ChatOpenAI(
            model=config.model_name,
            api_key="EMPTY",  # vLLM 不需要真实 API Key
            base_url=base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
    
    # =========================================================================
    # Embedding 相关
    # =========================================================================
    
    @classmethod
    def create_embeddings(cls, config: Optional[EmbeddingConfig] = None) -> Embeddings:
        """
        根据配置创建 Embedding 模型实例
        
        Args:
            config: Embedding 配置，如果为 None 则使用默认配置
            
        Returns:
            Embeddings: Embedding 模型实例
            
        Raises:
            ValueError: 如果提供商不支持
            ImportError: 如果缺少必要的依赖包
        """
        if config is None:
            config = create_embedding_config_from_settings()
        
        logger.info(f"Creating Embeddings: provider={config.provider.value}, model={config.model_name}")
        
        if config.provider == EmbeddingProvider.OPENAI:
            return cls._create_openai_embeddings(config)
        elif config.provider == EmbeddingProvider.OLLAMA:
            return cls._create_ollama_embeddings(config)
        elif config.provider == EmbeddingProvider.DASHSCOPE:
            return cls._create_dashscope_embeddings(config)
        else:
            raise ValueError(f"Unsupported Embedding provider: {config.provider}")
    
    @classmethod
    def create_embeddings_cached(cls, config: Optional[EmbeddingConfig] = None) -> Embeddings:
        """
        创建或获取缓存的 Embedding 实例
        
        Args:
            config: Embedding 配置
            
        Returns:
            Embeddings: Embedding 模型实例
        """
        if config is None:
            config = create_embedding_config_from_settings()
        
        cache_key = f"{config.provider.value}:{config.model_name}:{config.base_url}"
        
        if cache_key not in cls._embeddings_cache:
            cls._embeddings_cache[cache_key] = cls.create_embeddings(config)
        
        return cls._embeddings_cache[cache_key]
    
    @classmethod
    def _create_openai_embeddings(cls, config: EmbeddingConfig) -> Embeddings:
        """创建 OpenAI Embeddings"""
        from langchain_openai import OpenAIEmbeddings
        
        kwargs: Dict[str, Any] = {
            "model": config.model_name,
            "api_key": config.get_api_key_value(),
        }
        
        if config.base_url:
            kwargs["base_url"] = config.base_url
        
        if config.dimensions:
            kwargs["dimensions"] = config.dimensions
        
        return OpenAIEmbeddings(**kwargs)
    
    @classmethod
    def _create_ollama_embeddings(cls, config: EmbeddingConfig) -> Embeddings:
        """创建 Ollama Embeddings"""
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise ImportError(
                "请安装 langchain-ollama: pip install langchain-ollama"
            )
        
        base_url = config.base_url or EMBEDDING_PROVIDER_INFO[EmbeddingProvider.OLLAMA]["default_base_url"]
        
        return OllamaEmbeddings(
            model=config.model_name,
            base_url=base_url,
        )
    
    @classmethod
    def _create_dashscope_embeddings(cls, config: EmbeddingConfig) -> Embeddings:
        """创建 DashScope Embeddings"""
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
        except ImportError:
            raise ImportError(
                "请安装 langchain-community: pip install langchain-community"
            )
        
        return DashScopeEmbeddings(
            model=config.model_name,
            dashscope_api_key=config.get_api_key_value(),
        )
    
    # =========================================================================
    # 工具方法
    # =========================================================================
    
    @classmethod
    def validate_config(cls, config: LLMConfig) -> Tuple[bool, str]:
        """
        校验 LLM 配置有效性
        
        进行基本的配置校验，不会实际调用 API。
        
        Args:
            config: LLM 配置
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息或 "OK")
        """
        provider_info = LLM_PROVIDER_INFO.get(config.provider)
        
        if provider_info is None:
            return False, f"Unsupported provider: {config.provider}"
        
        if provider_info["requires_api_key"] and not config.api_key:
            return False, f"Provider {config.provider.value} requires an API key"
        
        if not config.model_name:
            return False, "Model name is required"
        
        return True, "OK"
    
    @classmethod
    def validate_embedding_config(cls, config: EmbeddingConfig) -> Tuple[bool, str]:
        """
        校验 Embedding 配置有效性
        
        进行基本的配置校验，不会实际调用 API。
        
        Args:
            config: Embedding 配置
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息或 "OK")
        """
        provider_info = EMBEDDING_PROVIDER_INFO.get(config.provider)
        
        if provider_info is None:
            return False, f"Unsupported Embedding provider: {config.provider}"
        
        if provider_info["requires_api_key"] and not config.api_key:
            return False, f"Embedding provider {config.provider.value} requires an API key"
        
        if not config.model_name:
            return False, "Model name is required"
        
        return True, "OK"
    
    @classmethod
    async def validate_config_with_api(cls, config: LLMConfig) -> Tuple[bool, str]:
        """
        校验 LLM 配置有效性（包含 API 调用）
        
        尝试创建 LLM 实例并发送测试请求。
        
        Args:
            config: LLM 配置
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息或 "OK")
        """
        # 首先进行基本校验
        is_valid, message = cls.validate_config(config)
        if not is_valid:
            return is_valid, message
        
        # 尝试创建并调用 LLM
        try:
            llm = cls.create_llm(config)
            response = await llm.ainvoke("Hi")
            return True, "OK"
        except Exception as e:
            return False, str(e)
    
    @classmethod
    def get_supported_providers(cls) -> Dict[LLMProvider, Dict[str, Any]]:
        """
        获取支持的 LLM 提供商列表
        
        Returns:
            Dict: 提供商信息字典
        """
        return LLM_PROVIDER_INFO
    
    @classmethod
    def get_supported_embedding_providers(cls) -> Dict[EmbeddingProvider, Dict[str, Any]]:
        """
        获取支持的 Embedding 提供商列表
        
        Returns:
            Dict: 提供商信息字典
        """
        return EMBEDDING_PROVIDER_INFO
    
    @classmethod
    def get_provider_models(cls, provider: LLMProvider) -> List[str]:
        """
        获取指定提供商的推荐模型列表
        
        Args:
            provider: LLM 提供商
            
        Returns:
            List[str]: 模型名称列表
        """
        info = LLM_PROVIDER_INFO.get(provider)
        if info:
            return info.get("models", [])
        return []
    
    @classmethod
    def clear_cache(cls) -> None:
        """
        清除所有缓存的 LLM 和 Embedding 实例
        """
        cls._llm_cache.clear()
        cls._embeddings_cache.clear()
        logger.info("LLM and Embeddings cache cleared")


# =============================================================================
# 便捷函数
# =============================================================================

def create_llm(
    provider: Optional[LLMProvider] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> BaseChatModel:
    """
    创建 LLM 实例的便捷函数
    
    Args:
        provider: LLM 提供商，默认使用配置中的默认值
        model_name: 模型名称
        **kwargs: 其他 LLMConfig 参数
        
    Returns:
        BaseChatModel: LLM 实例
        
    Example:
        >>> llm = create_llm(provider=LLMProvider.OLLAMA, model_name="llama3")
        >>> llm = create_llm()  # 使用默认配置
    """
    if provider is None and model_name is None and not kwargs:
        return LLMFactory.create_llm_cached()
    
    config = LLMConfig(
        provider=provider or LLMProvider.DEEPSEEK,
        model_name=model_name or "deepseek-chat",
        **kwargs,
    )
    return LLMFactory.create_llm(config)


def create_embeddings(
    provider: Optional[EmbeddingProvider] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> Embeddings:
    """
    创建 Embedding 模型的便捷函数
    
    Args:
        provider: Embedding 提供商
        model_name: 模型名称
        **kwargs: 其他 EmbeddingConfig 参数
        
    Returns:
        Embeddings: Embedding 模型实例
    """
    if provider is None and model_name is None and not kwargs:
        return LLMFactory.create_embeddings_cached()
    
    config = EmbeddingConfig(
        provider=provider or EmbeddingProvider.OLLAMA,
        model_name=model_name or "nomic-embed-text",
        **kwargs,
    )
    return LLMFactory.create_embeddings(config)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "LLMFactory",
    "create_llm",
    "create_embeddings",
]


"""
LLM 工厂模块

提供统一的 LLM 模型创建接口。

技术栈:
    - Python 3.12
    - LangChain
    - langchain-openai

设计约束:
    - 支持多种 LLM 提供商（OpenAI、DeepSeek、DashScope）
    - 提供缓存机制避免重复创建
    - 统一的配置接口

使用示例:
    from core.llm_factory import LLMFactory
    from core.llm_config import LLMConfig
    from core.llm_providers import LLMProvider
    
    # 使用默认配置
    llm = LLMFactory.create_llm()
    
    # 使用自定义配置
    config = LLMConfig(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="sk-xxx",
    )
    llm = LLMFactory.create_llm(config)
"""

import logging
from typing import Optional, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from core.llm_config import LLMConfig
from core.llm_providers import LLMProvider, LLM_PROVIDER_INFO

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    LLM 工厂类
    
    提供统一的 LLM 模型创建接口，支持多种提供商：
    - OpenAI (GPT-4o, GPT-4, GPT-3.5)
    - DeepSeek (deepseek-chat, deepseek-coder)
    - DashScope (qwen-max, qwen-plus)
    
    Example:
        >>> llm = LLMFactory.create_llm()
        >>> response = llm.invoke("Hello!")
    """
    
    # LLM 实例缓存
    _llm_cache: Dict[str, BaseChatModel] = {}
    
    @classmethod
    def create_llm(cls, config: LLMConfig) -> BaseChatModel:
        """
        根据配置创建 LLM 实例
        
        Args:
            config: LLM 配置
            
        Returns:
            BaseChatModel: LLM 实例
            
        Raises:
            ValueError: 如果提供商不支持
            ImportError: 如果缺少必要的依赖包
        """
        logger.info(f"创建 LLM: provider={config.provider.value}, model={config.model_name}")
        
        if config.provider == LLMProvider.OPENAI:
            return cls._create_openai_llm(config)
        elif config.provider == LLMProvider.DEEPSEEK:
            return cls._create_deepseek_llm(config)
        elif config.provider == LLMProvider.DASHSCOPE:
            return cls._create_dashscope_llm(config)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {config.provider}")
    
    @classmethod
    def create_llm_cached(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建或获取缓存的 LLM 实例
        
        使用配置的哈希值作为缓存键，相同配置返回相同实例。
        
        Args:
            config: LLM 配置
            
        Returns:
            BaseChatModel: LLM 实例
        """
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
        )
    
    @classmethod
    def clear_cache(cls) -> None:
        """清除所有缓存的 LLM 实例"""
        cls._llm_cache.clear()
        logger.info("LLM 缓存已清除")


__all__ = [
    "LLMFactory",
]

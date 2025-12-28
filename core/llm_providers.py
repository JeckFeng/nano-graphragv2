"""
LLM 提供商枚举定义

定义系统支持的 LLM 提供商枚举类型。

技术栈:
    - Python 3.12
    - Enum

设计约束:
    - 支持 OpenAI、DeepSeek、DashScope 三种提供商
    - 提供提供商元数据信息
    - 支持从字符串转换为枚举值

使用示例:
    from core.llm_providers import LLMProvider
    
    provider = LLMProvider.DEEPSEEK
    print(provider.value)  # "deepseek"
"""

from enum import Enum
from typing import Dict, Any


class LLMProvider(str, Enum):
    """
    支持的 LLM 提供商
    
    Attributes:
        OPENAI: OpenAI (GPT-4o, GPT-4, GPT-3.5)
        DEEPSEEK: DeepSeek (deepseek-chat, deepseek-coder)
        DASHSCOPE: 阿里通义千问 (qwen-max, qwen-plus)
    """
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    DASHSCOPE = "dashscope"
    
    @classmethod
    def from_string(cls, value: str) -> "LLMProvider":
        """
        从字符串创建枚举值
        
        Args:
            value: 提供商名称字符串
            
        Returns:
            LLMProvider: 对应的枚举值
            
        Raises:
            ValueError: 如果提供商不支持
        """
        value_lower = value.lower()
        for provider in cls:
            if provider.value == value_lower:
                return provider
        raise ValueError(f"不支持的 LLM 提供商: {value}")


# 提供商元数据
LLM_PROVIDER_INFO: Dict[LLMProvider, Dict[str, Any]] = {
    LLMProvider.OPENAI: {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "requires_api_key": True,
        "default_base_url": "https://api.openai.com/v1",
        "description": "OpenAI GPT 系列模型",
    },
    LLMProvider.DEEPSEEK: {
        "name": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-coder"],
        "requires_api_key": True,
        "default_base_url": "https://api.deepseek.com",
        "description": "DeepSeek 深度求索 AI 模型",
    },
    LLMProvider.DASHSCOPE: {
        "name": "DashScope (阿里通义千问)",
        "models": ["qwen-max", "qwen-plus"],
        "requires_api_key": True,
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "description": "阿里云通义千问系列模型",
    },
}


__all__ = [
    "LLMProvider",
    "LLM_PROVIDER_INFO",
]

"""
Core 模块

提供 LLM 工厂和配置管理功能。

主要组件:
    - LLMFactory: LLM 实例创建工厂
    - LLMConfig: LLM 配置模型
    - LLMProvider: LLM 提供商枚举
    - load_llm_config: 配置加载函数

使用示例:
    from core import LLMFactory, load_llm_config
    
    # 加载配置
    config = load_llm_config()
    
    # 创建 LLM 实例
    llm = LLMFactory.create_llm(config)
"""

from core.llm_factory import LLMFactory
from core.llm_config import LLMConfig
from core.llm_providers import LLMProvider, LLM_PROVIDER_INFO
from core.config_loader import load_llm_config, load_yaml_config

__all__ = [
    "LLMFactory",
    "LLMConfig",
    "LLMProvider",
    "LLM_PROVIDER_INFO",
    "load_llm_config",
    "load_yaml_config",
]

"""
配置加载器模块

从 YAML 配置文件和环境变量加载 LLM 配置。

技术栈:
    - Python 3.12
    - PyYAML
    - python-dotenv

设计约束:
    - 支持从 YAML 文件加载配置
    - 支持从环境变量读取敏感信息（API Key）
    - 提供默认配置

使用示例:
    from core.config_loader import load_llm_config
    
    # 加载默认配置
    config = load_llm_config()
    
    # 加载指定提供商的配置
    config = load_llm_config(provider="deepseek")
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from pydantic import SecretStr
from core.llm_config import LLMConfig
from core.llm_providers import LLMProvider

# 配置文件路径
CONFIG_DIR = Path(__file__).parent.parent / "config"
LLM_CONFIG_FILE = CONFIG_DIR / "llm.yaml"


def load_yaml_config() -> dict:
    """
    从 YAML 文件加载配置
    
    Returns:
        dict: 配置字典
        
    Raises:
        FileNotFoundError: 如果配置文件不存在
    """
    if not LLM_CONFIG_FILE.exists():
        raise FileNotFoundError(f"配置文件不存在: {LLM_CONFIG_FILE}")
    
    with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_llm_config(provider: Optional[str] = None) -> LLMConfig:
    """
    加载 LLM 配置
    
    从 YAML 文件加载基础配置，从环境变量读取 API Key 和 Base URL。
    
    Args:
        provider: 提供商名称（openai/deepseek/dashscope），默认使用配置文件中的 default
        
    Returns:
        LLMConfig: LLM 配置对象
        
    Raises:
        ValueError: 如果提供商不支持或配置无效
    """
    # 加载 YAML 配置
    yaml_config = load_yaml_config()
    
    # 确定使用哪个提供商的配置
    if provider is None:
        config_data = yaml_config.get("default", {})
        provider = config_data.get("provider", "dashscope")
    else:
        config_data = yaml_config.get(provider, {})
        if not config_data:
            raise ValueError(f"配置文件中未找到提供商 '{provider}' 的配置")
    
    # 转换 provider 为枚举
    llm_provider = LLMProvider.from_string(provider)
    
    # 从环境变量读取 API Key 和 Base URL
    api_key = None
    base_url = None
    
    if llm_provider == LLMProvider.OPENAI:
        api_key_str = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
    elif llm_provider == LLMProvider.DEEPSEEK:
        api_key_str = os.environ.get("DEEPSEEK_API_KEY")
        base_url = os.environ.get("DEEPSEEK_BASE_URL")
    elif llm_provider == LLMProvider.DASHSCOPE:
        api_key_str = os.environ.get("DASHSCOPE_API_KEY")
        base_url = os.environ.get("DASHSCOPE_BASE_URL")
    else:
        api_key_str = None
    
    # 转换 API Key 为 SecretStr
    if api_key_str:
        api_key = SecretStr(api_key_str)
    
    # 构建配置对象
    return LLMConfig(
        provider=llm_provider,
        model_name=config_data.get("model_name", "qwen-plus"),
        api_key=api_key,
        base_url=base_url,
        temperature=config_data.get("temperature", 0.7),
        max_tokens=config_data.get("max_tokens"),
        timeout=config_data.get("timeout", 60.0),
    )


__all__ = [
    "load_llm_config",
    "load_yaml_config",
]

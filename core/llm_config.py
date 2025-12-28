"""
LLM 配置模型定义

使用 Pydantic 定义 LLM 的配置模型。

技术栈:
    - Python 3.12
    - Pydantic v2

设计约束:
    - 使用 Pydantic 进行配置验证
    - 支持 API Key 安全存储
    - 提供配置导出功能

使用示例:
    from core.llm_config import LLMConfig
    from core.llm_providers import LLMProvider
    
    config = LLMConfig(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="sk-xxx",
    )
"""

from typing import Optional
from pydantic import BaseModel, Field, SecretStr, field_validator
from core.llm_providers import LLMProvider


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
    """
    
    provider: LLMProvider = Field(
        default=LLMProvider.DASHSCOPE,
        description="LLM 提供商",
    )
    model_name: str = Field(
        default="qwen-plus",
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


__all__ = [
    "LLMConfig",
]

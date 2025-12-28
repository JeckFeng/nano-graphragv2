"""
LLM 提供商枚举定义

定义系统支持的 LLM 和 Embedding 提供商枚举类型。

使用示例:
    from core.llm_providers import LLMProvider, EmbeddingProvider
    
    provider = LLMProvider.DEEPSEEK
    print(provider.value)  # "deepseek"
"""

from enum import Enum


class LLMProvider(str, Enum):
    """
    支持的 LLM 提供商
    
    Attributes:
        OPENAI: OpenAI (GPT-4o, GPT-4, GPT-3.5)
        DEEPSEEK: DeepSeek (deepseek-chat, deepseek-coder)
        DASHSCOPE: 阿里通义千问 (qwen-max, qwen-plus, qwen-turbo)
        OLLAMA: Ollama 本地部署 (llama3, qwen2, mistral)
        VLLM: vLLM 本地部署 (自定义模型)
    """
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    DASHSCOPE = "dashscope"
    OLLAMA = "ollama"
    VLLM = "vllm"
    
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
        raise ValueError(f"Unsupported LLM provider: {value}")


class EmbeddingProvider(str, Enum):
    """
    支持的 Embedding 提供商
    
    Attributes:
        OPENAI: OpenAI Embeddings (text-embedding-3-small, text-embedding-3-large)
        OLLAMA: Ollama 本地 Embedding (nomic-embed-text, mxbai-embed-large)
        DASHSCOPE: 阿里 Embedding (text-embedding-v1, text-embedding-v2)
    """
    OPENAI = "openai"
    OLLAMA = "ollama"
    DASHSCOPE = "dashscope"
    
    @classmethod
    def from_string(cls, value: str) -> "EmbeddingProvider":
        """
        从字符串创建枚举值
        
        Args:
            value: 提供商名称字符串
            
        Returns:
            EmbeddingProvider: 对应的枚举值
            
        Raises:
            ValueError: 如果提供商不支持
        """
        value_lower = value.lower()
        for provider in cls:
            if provider.value == value_lower:
                return provider
        raise ValueError(f"Unsupported Embedding provider: {value}")


# =============================================================================
# 提供商元数据
# =============================================================================

LLM_PROVIDER_INFO = {
    LLMProvider.OPENAI: {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1-preview", "o1-mini"],
        "requires_api_key": True,
        "default_base_url": "https://api.openai.com/v1",
        "description": "OpenAI GPT 系列模型",
    },
    LLMProvider.DEEPSEEK: {
        "name": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "requires_api_key": True,
        "default_base_url": "https://api.deepseek.com",
        "description": "DeepSeek 深度求索 AI 模型",
    },
    LLMProvider.DASHSCOPE: {
        "name": "DashScope (阿里通义千问)",
        "models": ["qwen-max", "qwen-max-latest", "qwen-plus", "qwen-turbo"],
        "requires_api_key": True,
        "default_base_url": None,
        "description": "阿里云通义千问系列模型",
    },
    LLMProvider.OLLAMA: {
        "name": "Ollama (本地部署)",
        "models": ["llama3", "llama3.2", "qwen2.5", "mistral", "codellama", "deepseek-r1"],
        "requires_api_key": False,
        "default_base_url": "http://localhost:11434",
        "description": "Ollama 本地运行开源模型",
    },
    LLMProvider.VLLM: {
        "name": "vLLM (本地部署)",
        "models": [],  # 取决于用户部署的模型
        "requires_api_key": False,
        "default_base_url": "http://localhost:8000/v1",
        "description": "vLLM 高性能推理服务器",
    },
}

EMBEDDING_PROVIDER_INFO = {
    EmbeddingProvider.OPENAI: {
        "name": "OpenAI Embeddings",
        "models": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        "requires_api_key": True,
        "default_base_url": "https://api.openai.com/v1",
        "description": "OpenAI 文本嵌入模型",
    },
    EmbeddingProvider.OLLAMA: {
        "name": "Ollama Embeddings (本地)",
        "models": ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
        "requires_api_key": False,
        "default_base_url": "http://localhost:11434",
        "description": "Ollama 本地嵌入模型",
    },
    EmbeddingProvider.DASHSCOPE: {
        "name": "DashScope Embeddings",
        "models": ["text-embedding-v1", "text-embedding-v2"],
        "requires_api_key": True,
        "default_base_url": None,
        "description": "阿里云文本嵌入模型",
    },
}


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "LLMProvider",
    "EmbeddingProvider",
    "LLM_PROVIDER_INFO",
    "EMBEDDING_PROVIDER_INFO",
]


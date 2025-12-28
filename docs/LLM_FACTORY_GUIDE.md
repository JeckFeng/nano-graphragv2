# LLM 工厂使用指南

## 概述

本项目实现了一个灵活的 LLM 工厂模式，支持多种大语言模型提供商。用户可以通过配置文件轻松切换不同的 LLM 提供商和模型。

## 支持的提供商

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **DeepSeek**: deepseek-chat, deepseek-coder
- **DashScope**: qwen-max, qwen-plus (阿里通义千问)

## 配置方式

### 1. 配置文件 (config/llm.yaml)

编辑 `config/llm.yaml` 文件，设置默认使用的提供商和模型：

```yaml
default:
  provider: dashscope  # 可选: openai, deepseek, dashscope
  model_name: qwen-plus
  temperature: 0.7
  max_tokens: null
  timeout: 60.0
```

### 2. 环境变量 (.env)

在 `.env` 文件中配置 API Key 和 Base URL：

```bash
# DashScope (默认)
DASHSCOPE_API_KEY=sk-your-dashscope-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# OpenAI (可选)
# OPENAI_API_KEY=sk-your-openai-key
# OPENAI_BASE_URL=https://api.openai.com/v1

# DeepSeek (可选)
# DEEPSEEK_API_KEY=sk-your-deepseek-key
# DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## 使用方法

### 方法 1: 使用默认配置

```python
from core import LLMFactory, load_llm_config

# 加载默认配置（从 config/llm.yaml 的 default 部分）
config = load_llm_config()

# 创建 LLM 实例
llm = LLMFactory.create_llm(config)

# 使用 LLM
response = llm.invoke("你好！")
print(response.content)
```

### 方法 2: 指定提供商

```python
from core import LLMFactory, load_llm_config

# 加载 DeepSeek 配置
config = load_llm_config(provider="deepseek")

# 创建 LLM 实例
llm = LLMFactory.create_llm(config)
```

### 方法 3: 自定义配置

```python
from core import LLMFactory, LLMConfig, LLMProvider
from pydantic import SecretStr

# 手动创建配置
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o",
    api_key=SecretStr("sk-your-key"),
    temperature=0.5,
    max_tokens=2000,
)

# 创建 LLM 实例
llm = LLMFactory.create_llm(config)
```

## 切换提供商

### 切换到 OpenAI

1. 在 `.env` 中配置 OpenAI API Key
2. 修改 `config/llm.yaml` 的 default 部分
3. 重启应用

### 切换到 DeepSeek

1. 在 `.env` 中配置 DeepSeek API Key
2. 修改 `config/llm.yaml` 的 default 部分
3. 重启应用

## 配置参数说明

- `provider`: LLM 提供商 (openai/deepseek/dashscope)
- `model_name`: 模型名称
- `api_key`: API 密钥（从环境变量读取）
- `base_url`: API 基础 URL（可选）
- `temperature`: 温度参数 (0.0-2.0)
- `max_tokens`: 最大输出 token 数
- `timeout`: 请求超时时间（秒）

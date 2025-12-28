# 快速开始指南

## 1. 查看当前配置

查看 `config/llm.yaml` 文件，确认默认使用的提供商：

```yaml
default:
  provider: dashscope  # 当前使用 DashScope
  model_name: qwen-plus
```

## 2. 配置 API Key

在 `.env` 文件中设置对应提供商的 API Key：

```bash
# DashScope (默认)
DASHSCOPE_API_KEY=sk-your-key-here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 3. 运行应用

```bash
cd /mnt/data_nvme/code/python/nano-graphragv2
python agents/top_supervisor.py
```

## 4. 切换到其他提供商

### 切换到 OpenAI

1. 编辑 `.env`：
```bash
OPENAI_API_KEY=sk-your-openai-key
```

2. 编辑 `config/llm.yaml`：
```yaml
default:
  provider: openai
  model_name: gpt-4o
```

3. 重启应用

### 切换到 DeepSeek

1. 编辑 `.env`：
```bash
DEEPSEEK_API_KEY=sk-your-deepseek-key
```

2. 编辑 `config/llm.yaml`：
```yaml
default:
  provider: deepseek
  model_name: deepseek-chat
```

3. 重启应用

## 5. 验证配置

运行测试脚本（需要安装依赖）：

```bash
python test_llm_factory.py
```

## 常见问题

**Q: 如何知道使用的是哪个模型？**

A: 查看应用启动日志，会显示：
```
创建 LLM: provider=dashscope, model=qwen-plus
```

**Q: 可以同时使用多个提供商吗？**

A: 可以。在代码中指定不同的 provider：
```python
config1 = load_llm_config(provider="openai")
config2 = load_llm_config(provider="deepseek")
```

**Q: 如何添加新的模型？**

A: 在 `config/llm.yaml` 的 `supported_models` 部分添加即可。

## 更多信息

- 详细使用指南: `docs/LLM_FACTORY_GUIDE.md`
- 改造总结: `LLM_FACTORY_SUMMARY.md`

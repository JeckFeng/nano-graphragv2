"""
LLM 工厂测试脚本

测试 LLM 工厂的基本功能。

使用方法:
    python test_llm_factory.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from core import LLMFactory, load_llm_config, LLMProvider

# 加载环境变量
load_dotenv()


def test_load_config():
    """测试配置加载"""
    print("=" * 60)
    print("测试 1: 加载默认配置")
    print("=" * 60)
    
    try:
        config = load_llm_config()
        print(f"✓ 配置加载成功")
        print(f"  提供商: {config.provider.value}")
        print(f"  模型: {config.model_name}")
        print(f"  温度: {config.temperature}")
        print(f"  有 API Key: {config.api_key is not None}")
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


def test_create_llm():
    """测试 LLM 创建"""
    print("\n" + "=" * 60)
    print("测试 2: 创建 LLM 实例")
    print("=" * 60)
    
    try:
        config = load_llm_config()
        llm = LLMFactory.create_llm(config)
        print(f"✓ LLM 实例创建成功")
        print(f"  类型: {type(llm).__name__}")
        return True
    except Exception as e:
        print(f"✗ LLM 创建失败: {e}")
        return False


def test_llm_invoke():
    """测试 LLM 调用"""
    print("\n" + "=" * 60)
    print("测试 3: 调用 LLM")
    print("=" * 60)
    
    try:
        config = load_llm_config()
        llm = LLMFactory.create_llm(config)
        
        print("发送测试消息: '你好，请用一句话介绍你自己'")
        response = llm.invoke("你好，请用一句话介绍你自己")
        print(f"✓ LLM 调用成功")
        print(f"  响应: {response.content[:100]}...")
        return True
    except Exception as e:
        print(f"✗ LLM 调用失败: {e}")
        return False


def test_cache():
    """测试缓存功能"""
    print("\n" + "=" * 60)
    print("测试 4: 缓存功能")
    print("=" * 60)
    
    try:
        config = load_llm_config()
        llm1 = LLMFactory.create_llm_cached(config)
        llm2 = LLMFactory.create_llm_cached(config)
        
        if llm1 is llm2:
            print(f"✓ 缓存功能正常（返回相同实例）")
            return True
        else:
            print(f"✗ 缓存功能异常（返回不同实例）")
            return False
    except Exception as e:
        print(f"✗ 缓存测试失败: {e}")
        return False


def test_provider_info():
    """测试提供商信息"""
    print("\n" + "=" * 60)
    print("测试 5: 提供商信息")
    print("=" * 60)
    
    try:
        from core.llm_providers import LLM_PROVIDER_INFO
        
        print("支持的提供商:")
        for provider, info in LLM_PROVIDER_INFO.items():
            print(f"\n  {provider.value}:")
            print(f"    名称: {info['name']}")
            print(f"    模型: {', '.join(info['models'][:3])}...")
            print(f"    需要 API Key: {info['requires_api_key']}")
        
        print(f"\n✓ 提供商信息获取成功")
        return True
    except Exception as e:
        print(f"✗ 提供商信息获取失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("LLM 工厂功能测试")
    print("=" * 60)
    
    tests = [
        test_load_config,
        test_create_llm,
        test_provider_info,
        test_cache,
        # test_llm_invoke,  # 需要有效的 API Key，可选
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试异常: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有测试通过")
        return 0
    else:
        print("✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

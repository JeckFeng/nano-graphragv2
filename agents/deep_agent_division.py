"""
Deep Agent 除法工具示例

本模块演示如何使用 LangGraph 的 Deep Agents 框架构建一个带有人工审核功能的智能代理。
该代理使用 DashScope 的 qwen-plus 模型，并实现了 Human-in-the-loop 功能。

技术栈:
    - LangGraph 1.0
    - Deep Agents
    - DashScope (qwen-plus)
    - Python 3.12

设计约束:
    - 必须配置 DASHSCOPE_API_KEY 环境变量
    - 使用 MemorySaver 作为 checkpointer 以支持状态持久化
    - 工具调用需要人工审核批准
    - 除数不能为零
"""

import asyncio
import os
import uuid
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# 加载环境变量
load_dotenv()


@tool
def divide_numbers(a: float, b: float) -> float:
    """
    两数相除工具
    
    Args:
        a: 被除数
        b: 除数
        
    Returns:
        float: 两数之商
        
    Raises:
        ValueError: 当除数为零时抛出
    """
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


def create_math_agent() -> tuple:
    """
    创建数学助手 Deep Agent
    
    该函数初始化一个带有人工审核功能的 Deep Agent，使用 DashScope 的 qwen-plus 模型。
    
    Returns:
        tuple: (agent, checkpointer) - 代理实例和检查点保存器
        
    Raises:
        ValueError: 当 DASHSCOPE_API_KEY 未设置时抛出
    """
    # 验证 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")
    
    # 初始化 DashScope 模型（使用 ChatOpenAI 封装）
    model = ChatOpenAI(
        model="qwen-plus",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key=api_key
    )
    
    # 创建内存检查点保存器（Human-in-the-loop 必需）
    checkpointer = MemorySaver()
    
    # 创建 Deep Agent
    agent = create_deep_agent(
        model=model,
        tools=[divide_numbers],
        interrupt_on={
            "divide_numbers": True,  # 调用除法工具时中断，等待人工审核
        },
        checkpointer=checkpointer,
        system_prompt="""你是一个数学助手。使用 divide_numbers 工具来计算两数之商。注意除数不能为零。

**严格规则**：
1. 禁止自己进行任何计算，无论多简单，必须使用 divide_numbers 工具
2. 如果工具调用被人工审核拒绝（rejected），不要重试，直接报告操作被拒绝
3. 不要尝试用其他方式计算或编造结果"""
    )
    
    return agent, checkpointer


def handle_human_review(result: dict, config: dict, agent) -> dict:
    """
    处理人工审核流程
    
    当代理执行被中断时，显示待审核的工具调用信息，并等待用户决策。
    
    Args:
        result: 代理执行结果，包含中断信息
        config: 配置字典，包含 thread_id 用于状态持久化
        agent: Deep Agent 实例
        
    Returns:
        dict: 恢复执行后的最终结果
    """
    # 提取中断信息
    interrupts = result["__interrupt__"][0].value
    action_requests = interrupts["action_requests"]
    review_configs = interrupts["review_configs"]
    
    # 创建工具名到审核配置的映射
    config_map = {cfg["action_name"]: cfg for cfg in review_configs}
    
    # 显示待审核的工具调用信息
    print("\n" + "="*50)
    print("检测到工具调用，需要人工审核：")
    print("="*50)
    
    for action in action_requests:
        review_config = config_map[action["name"]]
        print(f"\n工具名称: {action['name']}")
        print(f"调用参数: {action['args']}")
        print(f"允许的决策: {review_config['allowed_decisions']}")
    
    # 获取用户决策
    user_input = input("\n请输入决策 (approve/reject/edit): ").strip().lower()
    
    # 构建决策列表
    if user_input == "approve":
        decisions = [{"type": "approve"}]
        print("✓ 已批准工具调用")
    elif user_input == "reject":
        decisions = [{"type": "reject"}]
        print("✗ 已拒绝工具调用")
    else:
        # 默认批准
        decisions = [{"type": "approve"}]
        print("✓ 默认批准工具调用")
    
    # 使用相同的 config 恢复执行
    result = asyncio.run(agent.ainvoke(
        Command(resume={"decisions": decisions}),
        config=config
    ))
    
    return result


def main() -> None:
    """
    主函数：演示 Deep Agent 的 Human-in-the-loop 功能
    
    执行流程:
        1. 创建 Deep Agent
        2. 发送用户请求
        3. 如果触发中断，进行人工审核
        4. 显示最终结果
    """
    try:
        # 创建代理
        agent, checkpointer = create_math_agent()
        
        # 创建配置，包含唯一的 thread_id 用于状态持久化
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        # 发送用户请求
        print("发送请求: What is 20 / 4?")
        result = asyncio.run(agent.ainvoke({
            "messages": [{"role": "user", "content": "What is 20 / 4?"}]
        }, config=config))
        
        # 检查是否触发中断（需要人工审核）
        if result.get("__interrupt__"):
            result = handle_human_review(result, config, agent)
        
        # 显示最终结果
        print("\n" + "="*50)
        print("最终答案:")
        print("="*50)
        print(result["messages"][-1].content)
        
    except ValueError as e:
        print(f"配置错误: {e}")
    except Exception as e:
        print(f"执行错误: {e}")


if __name__ == "__main__":
    main()

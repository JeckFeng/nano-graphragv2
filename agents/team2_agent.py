"""
Team2 Agent - 加减法运算团队代理

本模块实现一个团队代理，它不直接使用工具，而是通过调用子代理来完成任务。
Team2 Agent 管理两个子代理：加法代理和减法代理。

技术栈:
    - LangGraph 1.0
    - Deep Agents (CompiledSubAgent 功能)
    - DashScope (qwen-plus)
    - Python 3.12

设计约束:
    - 必须配置 DASHSCOPE_API_KEY 环境变量
    - 使用 MemorySaver 作为 checkpointer 以支持状态持久化
    - 子代理的工具调用需要人工审核批准
    - Team2 Agent 本身不使用工具，只负责任务分发
    - 复用已有的 deep_agent_add 和 deep_agent_subtraction 实现
"""

import os
import uuid

from deepagents import create_deep_agent, CompiledSubAgent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# 导入已有的代理实现
from agents.deep_agent_add import create_math_agent as create_addition_agent
from agents.deep_agent_subtraction import create_math_agent as create_subtraction_agent

# 加载环境变量
load_dotenv()


def create_team2_agent() -> tuple:
    """
    创建 Team2 Agent（团队代理）
    
    该函数创建一个团队代理，它管理两个子代理：
    - addition-agent: 负责加法运算（复用 deep_agent_add）
    - subtraction-agent: 负责减法运算（复用 deep_agent_subtraction）
    
    团队代理本身不使用工具，而是将任务委派给合适的子代理执行。
    
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
    
    # 创建加法子代理（复用已有实现）
    addition_agent, _ = create_addition_agent()
    addition_subagent = CompiledSubAgent(
        name="addition-agent",
        description="专门负责加法运算的代理",
        runnable=addition_agent
    )
    
    # 创建减法子代理（复用已有实现）
    subtraction_agent, _ = create_subtraction_agent()
    subtraction_subagent = CompiledSubAgent(
        name="subtraction-agent",
        description="专门负责减法运算的代理",
        runnable=subtraction_agent
    )
    
    # 定义子代理列表
    subagents = [addition_subagent, subtraction_subagent]
    
    # 创建 Team2 Agent（团队代理）
    agent = create_deep_agent(
        model=model,
        system_prompt="""你是 Team2 的团队管理者。你不直接执行计算，而是将任务委派给合适的子代理：

- 对于加法运算，调用 addition-agent
- 对于减法运算，调用 subtraction-agent

请根据用户的需求，选择合适的子代理来完成任务。

**严格规则**：
1. 禁止自己进行任何计算，无论多简单，必须调用子代理处理
2. 如果子代理的工具调用被人工审核拒绝（rejected），不要重试该操作
3. 被拒绝的操作不要给出计算结果，直接报告该操作被拒绝
4. 只返回成功执行的操作结果""",
        subagents=subagents,
        checkpointer=checkpointer
    )
    
    return agent, checkpointer


def handle_human_review(result: dict, config: dict, agent) -> dict:
    """
    处理人工审核流程
    
    当子代理执行被中断时，显示待审核的工具调用信息，并等待用户决策。
    
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
    print("检测到子代理工具调用，需要人工审核：")
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
    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config
    )
    
    return result


def main() -> None:
    """
    主函数：演示 Team2 Agent 的子代理功能
    
    执行流程:
        1. 创建 Team2 Agent（包含两个子代理）
        2. 发送用户请求
        3. Team2 Agent 将任务委派给合适的子代理
        4. 如果子代理触发中断，进行人工审核
        5. 显示最终结果
    """
    try:
        # 创建团队代理
        agent, checkpointer = create_team2_agent()
        
        # 创建配置，包含唯一的 thread_id 用于状态持久化
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        # 测试加法任务
        print("="*50)
        print("测试 1: 加法运算")
        print("="*50)
        print("发送请求: Calculate 5 + 3")
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Calculate 5 + 3"}]
        }, config=config)
        
        # 检查是否触发中断（需要人工审核）
        if result.get("__interrupt__"):
            result = handle_human_review(result, config, agent)
        
        # 显示最终结果
        print("\n" + "="*50)
        print("最终答案:")
        print("="*50)
        print(result["messages"][-1].content)
        
        # 测试减法任务
        print("\n\n" + "="*50)
        print("测试 2: 减法运算")
        print("="*50)
        config2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
        print("发送请求: Calculate 10 - 3")
        result2 = agent.invoke({
            "messages": [{"role": "user", "content": "Calculate 10 - 3"}]
        }, config=config2)
        
        # 检查是否触发中断（需要人工审核）
        if result2.get("__interrupt__"):
            result2 = handle_human_review(result2, config2, agent)
        
        # 显示最终结果
        print("\n" + "="*50)
        print("最终答案:")
        print("="*50)
        print(result2["messages"][-1].content)
        
    except ValueError as e:
        print(f"配置错误: {e}")
    except Exception as e:
        print(f"执行错误: {e}")


if __name__ == "__main__":
    main()

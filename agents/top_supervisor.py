"""
Top Supervisor - 总经理代理

本模块实现一个顶层监督代理，它不直接使用工具，而是通过调用团队代理来完成任务。
Top Supervisor 管理两个团队代理：Team1 和 Team2。

技术栈:
    - LangGraph 1.0
    - Deep Agents (CompiledSubAgent 功能)
    - DashScope (qwen-plus)
    - Python 3.12

设计约束:
    - 必须配置 DASHSCOPE_API_KEY 环境变量
    - 使用 MemorySaver 作为 checkpointer 以支持状态持久化
    - 子代理的工具调用需要人工审核批准
    - Top Supervisor 本身不使用工具，只负责任务分发
    - 复用已有的 team1_agent 和 team2_agent 实现
"""

import os
import uuid

from deepagents import create_deep_agent, CompiledSubAgent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# 导入已有的团队代理实现
from agents.team1_agent import create_team1_agent
from agents.team2_agent import create_team2_agent

# 加载环境变量
load_dotenv()


def create_top_supervisor() -> tuple:
    """
    创建 Top Supervisor（总经理代理）
    
    该函数创建一个顶层监督代理，它管理两个团队代理：
    - team1-agent: 负责乘法和除法运算（复用 team1_agent）
    - team2-agent: 负责加法和减法运算（复用 team2_agent）
    
    总经理代理本身不使用工具，而是将任务委派给合适的团队代理执行。
    
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
    
    # 创建 Team1 子代理（复用已有实现）
    team1_agent, _ = create_team1_agent()
    team1_subagent = CompiledSubAgent(
        name="team1-agent",
        description="负责乘法和除法运算的团队代理",
        runnable=team1_agent
    )
    
    # 创建 Team2 子代理（复用已有实现）
    team2_agent, _ = create_team2_agent()
    team2_subagent = CompiledSubAgent(
        name="team2-agent",
        description="负责加法和减法运算的团队代理",
        runnable=team2_agent
    )
    
    # 定义子代理列表
    subagents = [team1_subagent, team2_subagent]
    
    # 创建 Top Supervisor（总经理代理）
    agent = create_deep_agent(
        model=model,
        system_prompt="""你是公司的总经理（Top Supervisor）。你不直接执行计算，而是将任务委派给合适的团队：

- 对于乘法和除法运算，调用 team1-agent
- 对于加法和减法运算，调用 team2-agent

请根据用户的需求，选择合适的团队代理来完成任务。

**严格规则**：
1. 禁止自己进行任何计算，无论多简单，必须调用子代理处理
2. 如果子代理的工具调用被人工审核拒绝（rejected），不要重试该操作
3. 被拒绝的操作应该被视为用户明确不希望执行的，直接跳过
4. 在最终回复中明确说明：
   - 哪些操作成功执行了
   - 哪些操作被拒绝了（不要给出这些操作的结果）""",
        subagents=subagents,
        checkpointer=checkpointer
    )
    
    return agent, checkpointer


def handle_human_review(result: dict, config: dict, agent) -> dict:
    """
    处理人工审核流程
    
    当子代理执行被中断时，显示待审核的工具调用信息，并等待用户决策。
    支持处理多个中断对象（多层级代理场景）。
    
    Args:
        result: 代理执行结果，包含中断信息
        config: 配置字典，包含 thread_id 用于状态持久化
        agent: Deep Agent 实例
        
    Returns:
        dict: 恢复执行后的最终结果
    """
    # 获取所有中断对象
    interrupts_list = result["__interrupt__"]
    
    # 显示待审核的工具调用信息
    print("\n" + "="*50)
    print(f"检测到 {len(interrupts_list)} 个工具调用，需要人工审核：")
    print("="*50)
    
    # 收集决策
    resume_map = {}
    
    for i, interrupt_obj in enumerate(interrupts_list, 1):
        interrupts = interrupt_obj.value
        action_requests = interrupts["action_requests"]
        review_configs = interrupts["review_configs"]
        
        # 每个中断对象通常只有一个 action_request
        for action in action_requests:
            config_map = {cfg["action_name"]: cfg for cfg in review_configs}
            review_config = config_map[action["name"]]
            
            print(f"\n[{i}/{len(interrupts_list)}] 工具名称: {action['name']}")
            print(f"调用参数: {action['args']}")
            print(f"允许的决策: {review_config['allowed_decisions']}")
            
            user_input = input("请输入决策 (approve/reject/edit): ").strip().lower()
            
            # 构建决策
            if user_input == "approve":
                decision = {"type": "approve"}
                print("✓ 已批准")
            elif user_input == "reject":
                decision = {"type": "reject"}
                print("✗ 已拒绝")
            else:
                # 默认批准
                decision = {"type": "approve"}
                print("✓ 默认批准")
            
            # 同时构建 interrupt_id 映射（用于多中断场景）
            resume_map[interrupt_obj.id] = {"decisions":[decision]}
    print("resume_map is : ",resume_map)
    # 恢复中断
    result = agent.invoke(
        Command(resume=resume_map),
        config=config
    )
    
    return result


def main() -> None:
    """
    主函数：演示 Top Supervisor 的多层级代理功能
    
    执行流程:
        1. 创建 Top Supervisor（包含两个团队代理）
        2. 发送用户请求
        3. Top Supervisor 将任务委派给合适的团队代理
        4. 团队代理再将任务委派给具体的子代理
        5. 如果触发中断，进行人工审核
        6. 显示最终结果
    """
    try:
        # 创建总经理代理
        agent, checkpointer = create_top_supervisor()
        # # 测试 4: 减法任务（由 Team2 处理）
        print("\n\n" + "="*50)
        print("测试 4: 减法运算（Team2 负责）")
        print("="*50)
        config4 = {"configurable": {"thread_id": str(uuid.uuid4())}}
        print("发送请求: Calculate 10 - 3")
        result4 = agent.invoke({
            "messages": [{"role": "user", "content": "Calculate 10 - 3"}]
        }, config=config4)
        
        # 检查是否触发中断（需要人工审核）
        if result4.get("__interrupt__"):
            result4 = handle_human_review(result4, config4, agent)
        
        # 显示最终结果
        print("\n" + "="*50)
        print("最终答案:")
        print("="*50)
        print(result4["messages"][-1].content)
        
        # 测试 5: 多步骤复杂任务（跨团队协作）
        print("\n\n" + "="*50)
        print("测试 5: 多步骤复杂任务（需要多个团队协作）")
        print("="*50)
        config5 = {"configurable": {"thread_id": str(uuid.uuid4())}}
        print("发送请求: First calculate 10 × 20, then calculate 100 − 13, and finally calculate 30 ÷ 2.")
        result5 = agent.invoke({
            "messages": [{"role": "user", "content": "First calculate 10 × 20, then calculate 100 − 13, and finally calculate 30 ÷ 2."}]
        }, config=config5)
        
        # 处理可能的多次中断
        while result5.get("__interrupt__"):
            print("\n[检测到中断，需要人工审核]")
            result5 = handle_human_review(result5, config5, agent)
        
        # 显示最终结果
        print("\n" + "="*50)
        print("最终答案:")
        print("="*50)
        print(result5["messages"][-1].content)
        
        # 打印所有消息，查看工具调用情况
        print("\n" + "="*50)
        print("DEBUG: 完整消息历史（查看工具调用）")
        print("="*50)
        for i, msg in enumerate(result5["messages"]):
            print(f"\n--- 消息 {i+1} ---")
            print(f"类型: {msg.__class__.__name__}")
            print(f"内容: {msg.content if hasattr(msg, 'content') else 'N/A'}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"工具调用: {msg.tool_calls}")
            if hasattr(msg, 'name') and msg.name:
                print(f"工具名称: {msg.name}")
        print("="*50)
        
    except ValueError as e:
        print(f"配置错误: {e}")
    except Exception as e:
        print(f"执行错误: {e}")


if __name__ == "__main__":
    main()




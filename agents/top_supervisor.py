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
    - 使用 AsyncPostgresSaver 作为 checkpointer 以支持状态持久化
    - 子代理的工具调用需要人工审核批准
    - Top Supervisor 本身不使用工具，只负责任务分发
    - 复用已有的 team1_agent 和 team2_agent 实现
"""

import asyncio
import os
import uuid

from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from deepagents.backends import FilesystemBackend
# 导入已有的团队代理实现
from agents.team1_agent import create_team1_agent
from agents.team2_agent import create_team2_agent
# 导入 LLM 工厂
from core import LLMFactory, load_llm_config
from core.tool_context import wrap_runnable_with_tool_context

# 加载环境变量
load_dotenv()

# 从环境变量构建数据库 URI
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("LANGGRAPH_MEMORY_DB", "langgraph_memory")
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def create_top_supervisor(checkpointer) -> tuple:
    """
    创建 Top Supervisor（总经理代理）
    
    该函数创建一个顶层监督代理，它管理两个团队代理：
    - team1-agent: 负责乘法和除法运算（复用 team1_agent）
    - team2-agent: 负责加法和减法运算（复用 team2_agent）
    
    总经理代理本身不使用工具，而是将任务委派给合适的团队代理执行。
    
    Returns:
        tuple: (agent, checkpointer) - 代理实例和检查点保存器
        
    Raises:
        ValueError: 当 LLM 配置无效时抛出
    """
    # 使用 LLM 工厂创建模型实例
    # 从配置文件和环境变量加载配置
    llm_config = load_llm_config()
    model = LLMFactory.create_llm(llm_config)
    
    # 配置混合存储后端（官方示例写法）
    composite_backend = lambda rt: CompositeBackend(
        default=StateBackend(rt),
        routes={
            "/memories/": FilesystemBackend(root_dir="./fs",virtual_mode=True),
        }
    )
    
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

**记忆系统**：
- 短期记忆：保存在 /workspace/ 下，仅在当前对话中有效
- 长期记忆：保存在 /memories/ 下，跨对话持久化
- 用户偏好和重要信息应保存到 /memories/user_preferences.txt

**严格规则**：
1. 禁止自己进行任何计算，无论多简单，必须调用子代理处理
2. 如果子代理的工具调用被人工审核拒绝（rejected），不要重试该操作
3. 被拒绝的操作应该被视为用户明确不希望执行的，直接跳过
4. 在最终回复中明确说明：
   - 哪些操作成功执行了
   - 哪些操作被拒绝了（不要给出这些操作的结果）
5. 在对话开始时，尝试读取 /memories/user_preferences.txt 了解用户偏好""",
        subagents=subagents,
        checkpointer=checkpointer,
        backend=composite_backend,
        store=InMemoryStore()
    )
    
    agent = wrap_runnable_with_tool_context(agent)
    return agent, checkpointer


async def handle_human_review(result: dict, config: dict, agent) -> dict:
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
    result = await agent.ainvoke(
        Command(resume=resume_map),
        config=config
    )
    
    return result


async def main() -> None:
    """
    主函数：演示 Top Supervisor 的多轮对话功能
    
    执行流程:
        1. 创建 Top Supervisor（包含两个团队代理）
        2. 进入多轮对话循环
        3. Top Supervisor 将任务委派给合适的团队代理
        4. 如果触发中断，进行人工审核
        5. 显示结果并继续下一轮对话
    """
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            # 创建总经理代理
            agent, checkpointer = create_top_supervisor(checkpointer)
            
            # 创建配置，包含唯一的 thread_id 用于状态持久化
            # 可以将thread_id硬编码，用来测试长期记忆，长期记忆必须要求是相同的thread_id
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            
            print("=" * 60)
            print("Top Supervisor - 总经理代理（带记忆功能）")
            print("=" * 60)
            print(f"Thread ID: {thread_id}")
            print("\n功能说明：")
            print("- 短期记忆：/workspace/ 下的文件仅在当前对话有效")
            print("- 长期记忆：/memories/ 下的文件跨对话持久化")
            print("- 输入 'quit' 或 'exit' 退出")
            print("=" * 60)
            
            # 多轮对话循环
            while True:
                # 获取用户输入
                user_input = input("\n你: ").strip()
                
                # 退出条件
                if user_input.lower() in ["quit", "exit", "退出"]:
                    print("\n再见！")
                    break
                
                if not user_input:
                    continue
                
                # 发送用户请求
                result = await agent.ainvoke({
                    "messages": [{"role": "user", "content": user_input}]
                }, config=config)
                
                # 处理可能的多次中断
                while result.get("__interrupt__"):
                    result = await handle_human_review(result, config, agent)
                
                # 显示助手回复
                print(f"\n助手: {result['messages'][-1].content}")
        
    except ValueError as e:
        print(f"配置错误: {e}")
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"执行错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())

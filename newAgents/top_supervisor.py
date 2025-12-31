"""
Top Supervisor - 总协调代理

本模块实现一个顶层监督代理，它不直接使用工具，而是通过调用团队代理来完成任务。
Top Supervisor 管理四个团队代理：RAG、SQL、地图与 Neo4j。

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
    - 复用已有的 team 实现
"""

import asyncio
import uuid

from deepagents import CompiledSubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from config.settings import get_settings
from core import LLMFactory, load_llm_config
from core.tool_context import wrap_runnable_with_tool_context

# 导入已有的团队代理实现
from newAgents.team.map_team import create_map_team_agent
from newAgents.team.neo4j_team import create_neo4j_team_agent
from newAgents.team.rag_team import create_rag_team_agent
from newAgents.team.sql_team import create_sql_team_agent

# 加载环境变量
load_dotenv()

# 从配置构建 LangGraph 记忆数据库 URI
DB_URI = get_settings().langgraph_memory_database_url

# Top Supervisor 系统提示词
TOP_SUPERVISOR_PROMPT = """你是系统的顶层 Supervisor，负责协调多个专业团队完成用户的复杂任务。

## 可用团队

你可以委托任务给以下专业团队：
- **rag_team**: 知识库检索团队
  - 能力：文档搜索、GraphRAG 检索、图片/表格检索、答案生成
  - 适用场景：需要查询知识库、文档、产品说明等
  
- **sql_team**: SQL 数据库查询团队
  - 能力：表结构分析、SQL 生成与验证、数据库查询
  - 适用场景：需要查询业务数据库、统计分析、数据报表等
  
- **map_team**: 地图服务团队
  - 能力：驾车路线规划、POI 搜索
  - 适用场景：需要规划出行路线、查找地点等
  
- **neo4j_team**: 图数据库查询团队
  - 能力：Cypher 查询生成、图数据查询、关系分析
  - 适用场景：需要查询空间关系、地理实体关联等

## 核心能力

### 1. 意图识别
分析用户输入，识别出用户的真实需求和期望结果。考虑：
- 用户想要获取什么信息？
- 需要执行什么操作？
- 涉及哪些数据源或服务？

### 2. 任务拆解
将复杂任务拆解为可执行的子任务：
- 每个子任务应该明确、具体
- 识别子任务之间的依赖关系
- 确定可以并行执行的任务

### 3. 任务分配
根据任务性质分配给合适的团队：
- 知识查询 → rag_team
- 数据库查询 → sql_team
- 路线规划 → map_team
- 图关系查询 → neo4j_team

### 4. 结果汇总
收集各团队的执行结果，整合生成最终答案：
- 合并各团队返回的信息
- 处理部分失败的情况
- 生成结构化、易理解的最终答案

## 工作流程

1. **分析阶段**: 分析用户请求，识别意图和所需团队
2. **规划阶段**: 制定执行计划，哪些任务应该分配给哪个team
3. **执行阶段**: 将任务分配给对应的team
4. **汇总阶段**: 汇总结果生成最终答案


"""


def create_top_supervisor(checkpointer) -> tuple:
    """
    创建 Top Supervisor（顶层监督代理）。

    该函数创建一个顶层监督代理，它管理四个团队代理：
    - rag_team: 负责知识库检索（复用 rag_team）
    - sql_team: 负责数据库查询（复用 sql_team）
    - map_team: 负责路线规划（复用 map_team）
    - neo4j_team: 负责图数据库查询（复用 neo4j_team）

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
            "/memories/": FilesystemBackend(root_dir="./fs", virtual_mode=True),
        },
    )

    # 创建 RAG 子代理（复用已有实现）
    rag_team_agent, _ = create_rag_team_agent()
    rag_team_subagent = CompiledSubAgent(
        name="rag_team",
        description="知识库检索团队，负责文档与 GraphRAG 检索",
        runnable=rag_team_agent,
    )

    # 创建 SQL 子代理（复用已有实现）
    sql_team_agent, _ = create_sql_team_agent()
    sql_team_subagent = CompiledSubAgent(
        name="sql_team",
        description="SQL 数据库查询团队，负责结构分析与查询执行",
        runnable=sql_team_agent,
    )

    # 创建地图子代理（复用已有实现）
    map_team_agent, _ = create_map_team_agent()
    map_team_subagent = CompiledSubAgent(
        name="map_team",
        description="地图服务团队，负责路线规划与地点查询",
        runnable=map_team_agent,
    )

    # 创建 Neo4j 子代理（复用已有实现）
    neo4j_team_agent, _ = create_neo4j_team_agent()
    neo4j_team_subagent = CompiledSubAgent(
        name="neo4j_team",
        description="图数据库查询团队，负责 Cypher 生成与查询执行",
        runnable=neo4j_team_agent,
    )

    # 定义子代理列表
    subagents = [
        rag_team_subagent,
        sql_team_subagent,
        map_team_subagent,
        neo4j_team_subagent,
    ]

    # 创建 Top Supervisor（顶层监督代理）
    agent = create_deep_agent(
        model=model,
        system_prompt=TOP_SUPERVISOR_PROMPT,
        subagents=subagents,
        checkpointer=checkpointer,
        backend=composite_backend,
        store=InMemoryStore(),
    )

    agent = wrap_runnable_with_tool_context(agent)
    return agent, checkpointer


async def handle_human_review(result: dict, config: dict, agent) -> dict:
    """
    处理人工审核流程。

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
    print("\n" + "=" * 50)
    print(f"检测到 {len(interrupts_list)} 个工具调用，需要人工审核：")
    print("=" * 50)

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
            resume_map[interrupt_obj.id] = {"decisions": [decision]}

    print("resume_map is : ", resume_map)
    # 恢复中断
    result = await agent.ainvoke(
        Command(resume=resume_map),
        config=config,
    )

    return result


async def main() -> None:
    """
    主函数：演示 Top Supervisor 的多轮对话功能。

    执行流程:
        1. 创建 Top Supervisor（包含多个团队代理）
        2. 进入多轮对话循环
        3. Top Supervisor 将任务委派给合适的团队代理
        4. 如果触发中断，进行人工审核
        5. 显示结果并继续下一轮对话
    """
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            # 创建总协调代理
            agent, checkpointer = create_top_supervisor(checkpointer)

            # 创建配置，包含唯一的 thread_id 用于状态持久化
            # 可以将 thread_id 硬编码，用来测试长期记忆，长期记忆必须要求相同 thread_id
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            print("=" * 60)
            print("Top Supervisor - 顶层协调代理（带记忆功能）")
            print("=" * 60)
            print(f"Thread ID: {thread_id}")
            print("\n功能说明：")
            print("- 短期记忆：/workspace/ 下的文件仅在当前对话中有效")
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
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                )

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

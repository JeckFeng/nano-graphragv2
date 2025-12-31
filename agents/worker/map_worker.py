"""
Deep Agent 高德地图路线规划工具示例

本模块演示如何使用 LangGraph 的 Deep Agents 框架构建一个带有人工审核功能的智能代理。
该代理使用 DashScope 的 qwen-plus 模型，并实现了 Human-in-the-loop 功能。

注意：记忆系统已移至 top_supervisor，此 Agent 会自动继承父 Agent 的配置。

技术栈:
    - LangGraph 1.0
    - Deep Agents
    - DashScope (qwen-plus)
    - Python 3.12

设计约束:
    - 必须配置 DASHSCOPE_API_KEY 环境变量
    - 使用 MemorySaver 作为 checkpointer 以支持状态持久化
    - 工具调用需要人工审核批准
"""

import asyncio
import uuid

from deepagents import create_deep_agent
from dotenv import load_dotenv
from core.tool_context import context_tool, wrap_runnable_with_tool_context
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from core import LLMFactory, load_llm_config
from Tools.gaode_map_tool import gaode_driving_route as _gaode_driving_route

# 加载环境变量
load_dotenv()


@context_tool
async def gaode_driving_route(origin: str, destination: str) -> dict:
    """
    高德驾车路线规划工具

    Args:
        origin: 起点经纬度，格式为'经度,纬度'
        destination: 终点经纬度，格式为'经度,纬度'

    Returns:
        dict: 路线规划结果
    """
    return await _gaode_driving_route(origin=origin, destination=destination)


def create_map_agent() -> tuple:
    """
    创建路线规划 Deep Agent

    该函数初始化一个带有人工审核功能的 Deep Agent，使用 DashScope 的 qwen-plus 模型。
    注意：记忆系统已移至 top_supervisor，此 Agent 会自动继承父 Agent 的配置。

    Returns:
        tuple: (agent, checkpointer) - 代理实例和检查点保存器

    Raises:
        ValueError: 当 DASHSCOPE_API_KEY 未设置时抛出
    """
    # 使用 LLM 工厂创建模型实例
    # 从配置文件和环境变量加载配置
    llm_config = load_llm_config()
    model = LLMFactory.create_llm(llm_config)

    # 创建内存检查点保存器（Human-in-the-loop 必需）
    checkpointer = MemorySaver()

    # 创建 Deep Agent（不配置 backend 和 store，会从父 Agent 继承）
    agent = create_deep_agent(
        model=model,
        tools=[gaode_driving_route],
        interrupt_on={
            "gaode_driving_route": True,  # 调用路线规划工具时中断，等待人工审核
        },
        checkpointer=checkpointer,
        system_prompt="""你是一个路线规划助手，负责使用高德地图路线规划工具获取驾车路线。

## 任务目标与成功定义
- 目标：基于起点/终点坐标调用工具并返回可核验的路线结果。
- 成功：输出包含距离与时长的结构化答案；失败时说明原因与建议。

## 背景与上下文
- 当前仅支持坐标输入，格式为“经度,纬度”（如：116.481028,39.989643）。
- 记忆系统由上层代理管理，你只负责检索与结果整理。

## 角色定义
- **你（map_worker）**：执行路线规划工具调用并整理结果。

## 行为边界（Behavior Boundaries）
- 禁止自行计算或编造路线，必须使用 gaode_driving_route 工具。
- 若缺少坐标或格式不合法，不调用工具，直接提示需要补充/纠正坐标。
- 如果工具调用被人工审核拒绝（rejected），不要重试，直接报告操作被拒绝。
- 距离与时长必须输出为原始单位（米/秒），不做换算。

## 可使用工具（Tools）
- **gaode_driving_route**：驾车路线规划工具。

## 流程逻辑
1. 校验起点与终点坐标是否齐全且格式正确。
2. 调用 gaode_driving_route 获取路线结果。
3. 提取 summary 中的 distance_meters 与 duration_seconds。
4. 输出结构化答案。

## 验收标准（Acceptance Criteria）
- 明确给出距离与时长（单位：米/秒）。
- 若失败，给出原因与可执行建议（例如坐标格式）。

## 输出格式规定
按以下格式输出（无内容请填写“无”）：
1. **路线结论**：<简要结论>
2. **距离与时长**：distance_meters=<数字>, duration_seconds=<数字>
3. **来源与说明**：<高德地图/参数说明>
4. **异常或建议**：<原因与建议或“无”>"""
    )

    agent = wrap_runnable_with_tool_context(agent)
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
    主函数：演示 Deep Agent 的多轮对话功能

    执行流程:
        1. 创建 Deep Agent
        2. 进入多轮对话循环
        3. 每次请求如果触发中断，进行人工审核
        4. 显示结果并继续下一轮对话

    注意：记忆系统由 top_supervisor 统一管理
    """
    try:
        # 创建代理
        agent, checkpointer = create_map_agent()

        # 创建配置，包含唯一的 thread_id 用于状态持久化
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        print("=" * 60)
        print("路线规划 Deep Agent")
        print("=" * 60)
        print(f"Thread ID: {thread_id}")
        print("\n功能说明：")
        print("- 记忆系统由 top_supervisor 统一管理")
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
            result = asyncio.run(agent.ainvoke({
                "messages": [{"role": "user", "content": user_input}]
            }, config=config))

            # 检查是否触发中断（需要人工审核）
            if result.get("__interrupt__"):
                result = handle_human_review(result, config, agent)

            # 显示助手回复
            print(f"\n助手: {result['messages'][-1].content}")

    except ValueError as e:
        print(f"配置错误: {e}")
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"执行错误: {e}")


if __name__ == "__main__":
    main()

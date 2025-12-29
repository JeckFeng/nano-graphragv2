from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import AIMessage

import asyncio
import uuid
import os
from dotenv import load_dotenv
from config.settings import get_settings
# 加载环境变量
load_dotenv()

DB_URI = get_settings().langgraph_memory_database_url
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

class State(TypedDict):
    messages:Annotated[list,add_messages]

async def chatBot(state:State):
    return {"messages":[await model.ainvoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatBot",chatBot)
graph_builder.add_edge(START,"chatBot")
graph_builder.add_edge("chatBot",END)

async def stream_graph_updates(graph, user_input:str, config:dict):
    events = graph.astream({"messages":[{"role":"user","content":user_input}]},config=config,stream_mode="values")
    async for event in events:
        if "messages" in event:
            last_message = event["messages"][-1]
            # 只打印 AI 消息，跳过用户消息
            if hasattr(last_message, 'type') and last_message.type == 'ai':
                print("AI : ", last_message.content)

async def main():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        graph = graph_builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "0824fxl"}}
        while True:
            user_input = input("user :")
            if user_input.lower() in ["exit"]:
                break
            await stream_graph_updates(graph=graph, user_input=user_input, config=config)
            print("\n History: ")
            state = await graph.aget_state(config)
            for message in state.values["messages"]:
                if isinstance(message,AIMessage):
                    prefix = "AI"
                else:
                    prefix = "user"
                print(f"{prefix}: {message.content}")

if __name__=="__main__":
    asyncio.run(main())

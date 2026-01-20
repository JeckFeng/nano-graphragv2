"""API 会话流程测试脚本。

该脚本通过 REST + WebSocket 完成一次完整会话流程：
1) 创建会话
2) map_worker 路线规划
3) sql_worker 表结构查询
4) rag_worker 知识检索
5) 关闭会话（断开 WebSocket）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse, urlunparse

try:
    import httpx
except ImportError as exc:
    raise SystemExit("缺少依赖 httpx，请先安装：pip install httpx") from exc

try:
    import websockets
except ImportError as exc:
    raise SystemExit("缺少依赖 websockets，请先安装：pip install websockets") from exc


DEFAULT_BASE_URL = os.environ.get("NANO_GRAPHRAG_API_URL", "http://localhost:8000")
DEFAULT_USER_ID = os.environ.get("NANO_GRAPHRAG_USER_ID", "api-test-user")
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_TITLE = "API 会话流程测试"

QUESTION_FLOW = [
    ("sql_worker", "查询数据库中有哪些表格"),
    ("rag_worker", "土壤含水量状态信息采用什么方法获得？"),
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 命令行参数集合。
    """
    parser = argparse.ArgumentParser(description="完整会话流程 API 测试脚本")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API 基础地址，例如 http://localhost:8000",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help="外部用户标识（用于绑定会话）",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="会话标题（可选）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP 请求与 WebSocket 等待/心跳超时时间（秒）",
    )
    return parser.parse_args()


def build_ws_url(base_url: str, path: str, params: dict[str, str]) -> str:
    """构建 WebSocket URL。

    Args:
        base_url: API 基础地址。
        path: WebSocket 路径。
        params: 查询参数。

    Returns:
        str: 完整的 WebSocket URL。
    """
    http_url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    parsed = urlparse(http_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode(params)
    ws_url = urlunparse(
        (
            scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            query,
            parsed.fragment,
        )
    )
    return ws_url


async def create_conversation(
    client: httpx.AsyncClient,
    base_url: str,
    user_id: str,
    title: str | None,
) -> dict[str, Any]:
    """创建新会话。

    Args:
        client: httpx 异步客户端。
        base_url: API 基础地址。
        user_id: 外部用户标识。
        title: 会话标题。

    Returns:
        dict[str, Any]: 创建会话响应 JSON。

    Raises:
        RuntimeError: 当 API 返回错误状态码。
    """
    payload = {"user_id": user_id, "title": title}
    response = await client.post(urljoin(base_url.rstrip("/") + "/", "v1/conversations"), json=payload)
    if response.status_code != 201:
        raise RuntimeError(
            f"创建会话失败: status={response.status_code}, body={response.text}"
        )
    return response.json()


async def receive_event(websocket: Any, timeout: float) -> dict[str, Any]:
    """等待 WebSocket 返回事件。

    Args:
        websocket: WebSocket 连接对象。
        timeout: 超时时间（秒）。

    Returns:
        dict[str, Any]: 解析后的事件 payload。

    Raises:
        asyncio.TimeoutError: 等待事件超时。
        ValueError: 返回内容不是合法 JSON。
    """
    raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
    return json.loads(raw_message)


async def send_user_message(
    websocket: Any,
    content: str,
    timeout: float,
) -> dict[str, Any]:
    """通过 WebSocket 发送用户消息并等待最终响应。

    Args:
        websocket: WebSocket 连接对象。
        content: 用户输入内容。
        timeout: 超时时间（秒）。

    Returns:
        dict[str, Any]: 最终响应事件。

    Raises:
        RuntimeError: 当收到错误事件。
    """
    await websocket.send(json.dumps({"type": "user_message", "content": content}))
    while True:
        event = await receive_event(websocket, timeout=timeout)
        event_type = event.get("type")
        if event_type == "final":
            return event
        if event_type == "error":
            raise RuntimeError(
                f"WebSocket 返回错误: code={event.get('code')}, message={event.get('message')}"
            )


async def run_flow(
    base_url: str,
    user_id: str,
    title: str | None,
    timeout: float,
) -> None:
    """执行完整会话流程。

    Args:
        base_url: API 基础地址。
        user_id: 外部用户标识。
        title: 会话标题。
        timeout: HTTP 请求与 WebSocket 等待/心跳超时时间。
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        conversation = await create_conversation(client, base_url, user_id, title)
    thread_id = conversation.get("thread_id")
    if not thread_id:
        raise RuntimeError("创建会话返回数据缺少 thread_id")

    print("创建会话成功:", thread_id)
    ws_url = build_ws_url(
        base_url,
        "/v1/ws/chat",
        {"user_id": user_id, "thread_id": thread_id},
    )

    async with websockets.connect(
        ws_url,
        ping_interval=timeout,
        ping_timeout=timeout,
        open_timeout=timeout,
    ) as websocket:
        for step_name, question in QUESTION_FLOW:
            print(f"\n[{step_name}] 用户问题: {question}")
            event = await send_user_message(websocket, question, timeout)
            print(f"[{step_name}] 助手回复: {event.get('content', '')}")

    print("\n会话流程结束，WebSocket 已关闭。")


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    asyncio.run(run_flow(args.base_url, args.user_id, args.title, args.timeout))


if __name__ == "__main__":
    main()

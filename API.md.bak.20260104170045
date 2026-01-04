# 多智能体最小可用 API 设计方案（MVP）

本文档面向“先跑通”的最小可用 API。目标是：
- 提供新建对话与流式问答能力；
- 在服务层完成 user_id 与 thread_id 绑定；
- 应用层（top_supervisor + team + worker）只接收 thread_id；
- 数据持久化使用 PostgreSQL；
- WebSocket 实现流式输出。

## 1. 架构与分层职责

- 接口层（API）
  - 提供 REST + WebSocket 路由
  - 负责参数校验、鉴权、连接管理、消息序列化
  - 不直接访问数据库

- 服务层（Service）
  - 负责 user_id ↔ thread_id 绑定
  - 负责对话、消息记录的持久化
  - 提供业务接口给 API 层

- 应用层（Application）
  - 封装多智能体系统，仅接收 thread_id 与用户输入
  - 产出可流式输出的事件序列

- 基础设施层（Infra）
  - 数据库连接、配置、日志、健康检查

## 2. 最小可用 API 端点

- `POST /v1/conversations`
  - 请求：`{ "user_id": "u_123", "title": "可选" }`
  - 响应：`{ "thread_id": "uuid", "created_at": "..." }`

- `GET /v1/conversations?user_id=...`
  - 返回该用户的对话列表

- `GET /v1/conversations/{thread_id}/messages?user_id=...`
  - 拉取历史消息（便于前端回显）

- `WS /v1/ws/chat?user_id=...&thread_id=...`
  - 进行流式问答

- `GET /healthz`
  - 健康检查

## 3. WebSocket 消息格式（JSON）

### 3.1 基本字段（统一包裹）
- `type`: 消息类型
- `thread_id`: 对话线程 ID
- `message_id`: 单次对话消息 ID
- `sequence`: 流式序号（从 1 开始）
- `timestamp`: 服务器时间戳

### 3.2 客户端 -> 服务端

```json
{
  "type": "user_message",
  "thread_id": "uuid",
  "content": "从北京到上海路线",
  "metadata": {
    "locale": "zh-CN",
    "frontend_message_id": "client-123"
  }
}
```

### 3.3 服务端 -> 客户端（流式 token）

```json
{
  "type": "token",
  "thread_id": "uuid",
  "message_id": "m-001",
  "sequence": 12,
  "delta": "预计耗时 5 小时",
  "timestamp": "2026-01-04T13:00:00Z"
}
```

### 3.4 服务端 -> 客户端（最终结果）

```json
{
  "type": "final",
  "thread_id": "uuid",
  "message_id": "m-001",
  "content": "路线规划已完成。",
  "artifacts": [
    {
      "kind": "tool_result",
      "path": "/workspace/map_route_result.txt",
      "media_type": "text/plain"
    }
  ],
  "usage": {
    "input_tokens": 1234,
    "output_tokens": 567
  },
  "timestamp": "2026-01-04T13:00:02Z"
}
```

### 3.5 服务端 -> 客户端（错误）

```json
{
  "type": "error",
  "thread_id": "uuid",
  "message_id": "m-001",
  "code": "AGENT_FAILURE",
  "message": "模型调用失败",
  "retryable": true,
  "timestamp": "2026-01-04T13:00:02Z"
}
```

### 3.6 服务端 -> 客户端（心跳）

```json
{ "type": "ping" }
```

## 4. WebSocket 状态机（MVP）

状态定义：
- `CONNECTING`: 建立连接
- `AUTHED`: 完成 user_id/thread_id 校验
- `IDLE`: 等待用户消息
- `RUNNING`: 运行多智能体（流式输出）
- `DONE`: 一轮对话结束
- `ERROR`: 出错
- `CLOSED`: 连接关闭

状态流转（简化）：

```
CONNECTING -> AUTHED -> IDLE
IDLE --(user_message)--> RUNNING
RUNNING --(token stream)--> RUNNING
RUNNING --(final)--> DONE -> IDLE
RUNNING --(error)--> ERROR -> IDLE
* 任意状态 --(close)--> CLOSED
```

说明：
- MVP 直接在服务层自动批准工具调用（或采用默认批准策略）。
- 如果后续要做人工审核，可在 RUNNING 中间增加 `AWAIT_APPROVAL` 状态。

## 5. 服务层接口签名设计（建议）

以下为 Python 风格接口定义（不要求实现细节）：

### 5.1 UserService
```python
class UserService:
    async def get_or_create_user(self, external_user_id: str) -> User: ...
    async def get_user(self, external_user_id: str) -> User | None: ...
```

### 5.2 ThreadService
```python
class ThreadService:
    async def create_thread(self, user_id: int, title: str | None = None) -> Thread: ...
    async def list_threads(self, user_id: int, limit: int = 20, offset: int = 0) -> list[Thread]: ...
    async def get_thread(self, thread_id: str) -> Thread | None: ...
    async def verify_thread_owner(self, user_id: int, thread_id: str) -> bool: ...
```

### 5.3 MessageService
```python
class MessageService:
    async def append_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        tool_payload: dict | None = None,
    ) -> Message: ...

    async def list_messages(
        self,
        thread_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Message]: ...

    async def update_message_content(self, message_id: int, content: str) -> None: ...
```

### 5.4 ChatService（核心）
```python
class ChatService:
    async def stream_chat(
        self,
        user_external_id: str,
        thread_id: str,
        user_content: str,
        metadata: dict | None = None,
    ) -> AsyncIterator[dict]: ...
```

说明：
- `ChatService` 内部会：验证 user/thread 绑定 → 写入用户消息 → 调用应用层流式接口 → 逐步落库 → 产出 WS 事件。

## 6. 应用层：封装 top_supervisor 为“流式输出”

### 6.1 统一的应用层接口
```python
class AgentRunner:
    async def astream(
        self,
        thread_id: str,
        user_content: str,
    ) -> AsyncIterator[dict]: ...
```

### 6.2 事件建议结构
```json
{
  "type": "token" | "final" | "tool_result",
  "content": "...",
  "tool_path": "/workspace/..."  
}
```

### 6.3 包装思路（MVP）
- 使用现有 `create_top_supervisor()` 创建代理。
- 使用 `thread_id` 作为 LangGraph 的 `configurable.thread_id`。
- 若底层支持流式事件（例如 `agent.astream_events(...)` 或 `agent.astream(...)`），将事件转换为标准的 `token/final`。
- 若不支持流式事件，则：
  - 直接 `await agent.ainvoke(...)` 获取最终结果；
  - 在 WS 中只发送 `final`（流式退化）。

### 6.4 伪代码示例
```python
class TopSupervisorRunner(AgentRunner):
    def __init__(self, agent):
        self._agent = agent

    async def astream(self, thread_id: str, user_content: str):
        config = {"configurable": {"thread_id": thread_id}}
        # 优先使用事件流（如果可用）
        if hasattr(self._agent, "astream_events"):
            async for event in self._agent.astream_events(
                {"messages": [{"role": "user", "content": user_content}]},
                config=config,
            ):
                # 按事件类型映射为 token / tool_result / final
                yield self._map_event(event)
        else:
            result = await self._agent.ainvoke(
                {"messages": [{"role": "user", "content": user_content}]},
                config=config,
            )
            yield {"type": "final", "content": result["messages"][-1].content}
```

说明：
- 上述只展示统一入口，具体事件映射需要根据 LangGraph/DeepAgents 实际返回结构微调。
- 对于大型工具结果，可由 worker 写入 `/workspace/`，在 `tool_result` 事件中返回文件路径。

## 7. 数据库表设计（PostgreSQL）

### 7.1 users
- `id` BIGSERIAL PK
- `external_user_id` TEXT UNIQUE NOT NULL
- `created_at` TIMESTAMPTZ

### 7.2 threads
- `id` UUID PK  (thread_id)
- `user_id` BIGINT FK(users.id)
- `title` TEXT NULL
- `created_at` TIMESTAMPTZ

索引：`(user_id, created_at)`

### 7.3 messages
- `id` BIGSERIAL PK
- `thread_id` UUID FK(threads.id)
- `role` TEXT CHECK IN ('user','assistant','tool')
- `content` TEXT
- `tool_payload` JSONB NULL
- `created_at` TIMESTAMPTZ

索引：`(thread_id, created_at)`

## 8. 项目结构规划（MVP）

```
app/
  api/
    routes.py           # REST + WS 路由
  services/
    user_service.py
    thread_service.py
    message_service.py
    chat_service.py
  application/
    agent_runner.py     # 封装 top_supervisor
  infra/
    db.py               # 连接池/Session
    config.py
    logging.py
  schemas/
    requests.py
    responses.py
migrations/
```

## 9. 分阶段实施计划

### 阶段 0：最小骨架
- 建 FastAPI 应用
- 配置 PostgreSQL 连接
- 建表迁移（users/threads/messages）
- 提供 `POST /v1/conversations` 与 `/healthz`

### 阶段 1：用户绑定与历史查询
- 实现 `GET /v1/conversations`
- 实现 `GET /v1/conversations/{thread_id}/messages`
- 服务层完成 user_id/thread_id 绑定校验

### 阶段 2：WS 流式问答
- 实现 `WS /v1/ws/chat`
- 接入应用层流式接口
- 结果与 token 流式回写消息表

### 阶段 3：稳定性与可观测性
- 错误码与重试策略
- 连接心跳、断线重连
- 日志与指标


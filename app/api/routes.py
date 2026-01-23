"""API routes for the MVP service."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import UUID
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.agent_runner import TopSupervisorRunner
from app.infra.db import get_session
from app.schemas.requests import ApprovalDecisionRequest, CreateConversationRequest
from app.schemas.responses import (
    ApprovalListResponse,
    ApprovalRecordResponse,
    ApprovalResolutionResponse,
    ConversationListResponse,
    ConversationSummaryResponse,
    CreateConversationResponse,
    HealthResponse,
    MessageListResponse,
    MessageResponse,
)
from app.services.approval_service import ApprovalService
from app.services.artifact_service import ArtifactService
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from config.settings import get_settings
from app.observability import log_event, set_ctx_ws, clear_ctx, set_message_id
from app.observability.summarize import summarize_payload
from app.api.routes_trace import router as trace_router

router = APIRouter()
v1_router = APIRouter(prefix="/v1")


@router.get("/healthz", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
    """Return basic health status for the service.

    Returns:
        HealthResponse: Health response payload.
    """
    return HealthResponse(status="ok")


@v1_router.post(
    "/conversations",
    response_model=CreateConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    payload: CreateConversationRequest,
    session: AsyncSession = Depends(get_session),
) -> CreateConversationResponse:
    """Create a new conversation thread for a user.

    Args:
        payload: Conversation creation request.
        session: Database session dependency.

    Returns:
        CreateConversationResponse: Response payload.
    """
    service = ConversationService(session)
    thread = await service.create_conversation(payload.user_id, payload.title)
    return CreateConversationResponse(thread_id=str(thread.id), created_at=thread.created_at)


@v1_router.get(
    "/conversations",
    response_model=ConversationListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_conversations(
    user_id: str = Query(..., min_length=1, max_length=128),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> ConversationListResponse:
    """List conversations for a user.

    Args:
        user_id: External user identifier.
        limit: Maximum number of threads to return.
        offset: Offset for pagination.
        session: Database session dependency.

    Returns:
        ConversationListResponse: Response payload.
    """
    service = ConversationService(session)
    threads = await service.list_conversations(user_id, limit=limit, offset=offset)
    summaries: List[ConversationSummaryResponse] = [
        ConversationSummaryResponse(
            thread_id=str(thread.id),
            title=thread.title,
            created_at=thread.created_at,
        )
        for thread in threads
    ]
    return ConversationListResponse(conversations=summaries)


@v1_router.delete(
    "/conversations/{thread_id}",
    status_code=status.HTTP_200_OK,
)
async def delete_conversation(
    thread_id: UUID,
    user_id: str = Query(..., min_length=1, max_length=128),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Delete a conversation thread.

    Args:
        thread_id: Thread identifier.
        user_id: External user identifier.
        session: Database session dependency.

    Returns:
        dict: Deletion result.
    """
    service = ConversationService(session)
    deleted = await service.delete_conversation(user_id, thread_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"deleted": True, "thread_id": str(thread_id)}


@v1_router.get(
    "/conversations/{thread_id}/messages",
    response_model=MessageListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_messages(
    thread_id: UUID,
    user_id: str = Query(..., min_length=1, max_length=128),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> MessageListResponse:
    """List messages for a conversation thread.

    Args:
        thread_id: Thread identifier.
        user_id: External user identifier.
        limit: Maximum number of messages to return.
        offset: Offset for pagination.
        session: Database session dependency.

    Returns:
        MessageListResponse: Response payload.
    """
    service = ConversationService(session)
    messages = await service.list_messages(user_id, thread_id, limit=limit, offset=offset)
    if messages is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    payload = [
        MessageResponse(
            id=message.id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
        )
        for message in messages
    ]
    return MessageListResponse(messages=payload)


@v1_router.get(
    "/artifacts/{artifact_id}",
    status_code=status.HTTP_200_OK,
)
async def get_artifact(
    artifact_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """Fetch an artifact by identifier after user ownership validation.

    Args:
        artifact_id: Artifact identifier.
        user_id: External user identifier.
        session: Database session dependency.

    Returns:
        FileResponse: Artifact file response.
    """
    service = ArtifactService(session)
    artifact = await service.get_artifact_for_user(user_id, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    artifact_path = Path(artifact.storage_path)
    if not artifact_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file missing")
    return FileResponse(
        path=str(artifact_path),
        media_type=artifact.media_type,
        filename=artifact_path.name,
    )


@v1_router.get(
    "/approvals",
    response_model=ApprovalListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_approvals(
    user_id: str = Query(..., min_length=1, max_length=128),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    thread_id: Optional[str] = Query(default=None),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> ApprovalListResponse:
    """List approvals for a user.

    Args:
        user_id: External user identifier.
        status_filter: Optional approval status filter.
        thread_id: Optional thread filter.
        limit: Maximum number of approvals to return.
        offset: Offset for pagination.

    Returns:
        ApprovalListResponse: Response payload.
    """
    service = ApprovalService()
    approvals = await service.list_approvals(
        user_id=user_id,
        status=status_filter,
        thread_id=thread_id,
        limit=limit,
        offset=offset,
    )
    return ApprovalListResponse(approvals=[_map_approval_record(record) for record in approvals])


@v1_router.get(
    "/approvals/{approval_id}",
    response_model=ApprovalRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def get_approval(
    approval_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
) -> ApprovalRecordResponse:
    """Get approval details for a user.

    Args:
        approval_id: Approval identifier.
        user_id: External user identifier.

    Returns:
        ApprovalRecordResponse: Approval record.
    """
    service = ApprovalService()
    record = await service.get_approval(user_id, approval_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval not found")
    return _map_approval_record(record)


@v1_router.get(
    "/approvals/{approval_id}/status",
    status_code=status.HTTP_200_OK,
)
async def get_approval_status(
    approval_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
) -> dict:
    """Query approval execution status (for polling when WebSocket disconnects).

    Args:
        approval_id: Approval identifier.
        user_id: External user identifier.

    Returns:
        dict: Approval status payload.
    """
    service = ApprovalService()
    record = await service.get_approval(user_id, approval_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval not found")

    return {
        "approval_id": record.approval_id,
        "status": record.status,
        "result_content": record.result_content,
    }


@v1_router.post(
    "/approvals/{approval_id}",
    response_model=ApprovalResolutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def resolve_approval(
    approval_id: str,
    payload: ApprovalDecisionRequest,
    user_id: str = Query(..., min_length=1, max_length=128),
) -> ApprovalResolutionResponse:
    """Submit approval decision.

    Returns status="processing" to indicate execution has started.
    Results will be pushed via WebSocket.

    Args:
        approval_id: Approval identifier.
        payload: Approval decision payload.
        user_id: External user identifier.

    Returns:
        ApprovalResolutionResponse: Resolution result.
    """
    decision = (payload.decision or "").strip().lower()
    if decision not in {"approve", "reject", "edit"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid decision")

    service = ApprovalService()
    resolution = await service.resolve_approval(
        user_id=user_id,
        approval_id=approval_id,
        decision=decision,
        edited_args=payload.edited_args,
    )
    if resolution is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval not found")

    # 注意：不再在这里保存消息，由后台任务完成后保存

    return ApprovalResolutionResponse(
        approval_id=resolution.approval_id,
        status=resolution.status,
        result_content=resolution.result_content,
        next_approval=None,
    )


def build_ws_payload(
    event: dict,
    thread_id: str,
    sequence: int,
) -> dict:
    """Build a WebSocket payload by enriching event fields.

    Args:
        event: Event payload from the chat service.
        thread_id: Thread identifier.
        sequence: Sequence number for streaming order.

    Returns:
        dict: WebSocket payload.
    """
    payload = {
        "type": event.get("type", "final"),
        "thread_id": thread_id,
        "message_id": event.get("message_id"),
        "sequence": sequence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if "content" in event:
        payload["content"] = event["content"]
    if "delta" in event:
        payload["delta"] = event["delta"]
    if "artifacts" in event:
        payload["artifacts"] = event["artifacts"]
    if "approval_id" in event:
        payload["approval_id"] = event["approval_id"]
    if "interrupts" in event:
        payload["interrupts"] = event["interrupts"]
    if "status" in event:
        payload["status"] = event["status"]
    if "code" in event:
        payload["code"] = event["code"]
    if "message" in event:
        payload["message"] = event["message"]
    return payload


def _map_approval_record(record) -> ApprovalRecordResponse:
    """Map approval record to response payload.

    Args:
        record: Approval record instance.

    Returns:
        ApprovalRecordResponse: Mapped response.
    """
    interrupts = [
        {
            "interrupt_id": interrupt.interrupt_id,
            "action_requests": interrupt.action_requests,
            "review_configs": interrupt.review_configs,
        }
        for interrupt in record.interrupts
    ]
    return ApprovalRecordResponse(
        approval_id=record.approval_id,
        thread_id=record.thread_id,
        user_id=record.user_id,
        status=record.status,
        created_at=record.created_at,
        resolved_at=record.resolved_at,
        interrupts=interrupts,
        decision=record.decision,
        result_content=record.result_content,
    )


@v1_router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Handle WebSocket chat connections for streaming responses.

    Args:
        websocket: WebSocket connection.
    """
    await websocket.accept()
    user_id = websocket.query_params.get("user_id")
    thread_id = websocket.query_params.get("thread_id")
    streaming_flag = (websocket.query_params.get("streaming") or "1").lower()
    enable_streaming = streaming_flag not in {"0", "false", "off"}
    if not user_id or not thread_id:
        await websocket.close(code=1008)
        return
    try:
        thread_uuid = UUID(thread_id)
    except ValueError:
        await websocket.close(code=1008)
        return

    conn_id = uuid.uuid4().hex
    request_id = uuid.uuid4().hex
    set_ctx_ws(request_id=request_id, user_id=user_id, thread_id=thread_id, conn_id=conn_id)

    in_bytes = 0
    out_bytes = 0

    log_event(
        "ws",
        {
            "source": "backend",
            "component": "ws",
            "event_type": "system",
            "event_name": "ws_connect",
            "ws_event_type": "connect",
            "payload_bytes": 0,
        },
    )

    settings = get_settings()
    runner = TopSupervisorRunner(settings.langgraph_memory_database_url)

    # 注册 WebSocket 连接
    from app.services.ws_manager import ws_manager

    await ws_manager.register(thread_id, websocket)

    try:
        async for session in get_session():
            service = ChatService(session, runner)
        async def send_with_log(payload: dict) -> None:
            nonlocal out_bytes
            summary = summarize_payload(payload)
            out_bytes += int(summary.get("bytes", 0))
            log_event(
                "ws",
                {
                    "source": "backend",
                    "component": "ws",
                    "event_type": "exchange",
                    "event_name": "ws_message_out",
                    "ws_event_type": payload.get("type"),
                    "payload_keys": summary.get("keys"),
                    "payload_bytes": summary.get("bytes"),
                },
            )
            await websocket.send_json(payload)

        try:
            while True:
                payload = await websocket.receive_json()
                summary = summarize_payload(payload)
                in_bytes += int(summary.get("bytes", 0))
                log_event(
                    "ws",
                    {
                        "source": "backend",
                        "component": "ws",
                        "event_type": "exchange",
                        "event_name": "ws_message_in",
                        "ws_event_type": payload.get("type"),
                        "payload_keys": summary.get("keys"),
                        "payload_bytes": summary.get("bytes"),
                    },
                )
                if payload.get("type") != "user_message":
                    await send_with_log(
                        build_ws_payload(
                            {
                                "type": "error",
                                "code": "INVALID_MESSAGE",
                                "message": "Unsupported message type",
                            },
                            thread_id,
                            sequence=0,
                        )
                    )
                    continue
                content = payload.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    await send_with_log(
                        build_ws_payload(
                            {
                                "type": "error",
                                "code": "EMPTY_CONTENT",
                                "message": "Message content is empty",
                            },
                            thread_id,
                            sequence=0,
                        )
                    )
                    continue
                set_message_id(None)

                sequence = 0
                async for event in service.stream_chat(
                    external_user_id=user_id,
                    thread_id=thread_uuid,
                    user_content=content,
                    enable_streaming=enable_streaming,
                ):
                    sequence += 1
                    await send_with_log(build_ws_payload(event, thread_id, sequence=sequence))
        except WebSocketDisconnect:
            log_event(
                "ws",
                {
                    "source": "backend",
                    "component": "ws",
                    "event_type": "system",
                    "event_name": "ws_disconnect",
                    "ws_event_type": "disconnect",
                    "payload_bytes": out_bytes + in_bytes,
                    "in_bytes": in_bytes,
                    "out_bytes": out_bytes,
                },
            )
            clear_ctx()
            return
        except PermissionError:
            await send_with_log(
                build_ws_payload(
                    {
                        "type": "error",
                        "code": "NOT_AUTHORIZED",
                        "message": "Thread not found",
                    },
                    thread_id,
                    sequence=0,
                )
            )
            await websocket.close(code=1008)
            log_event(
                "ws",
                {
                    "source": "backend",
                    "component": "ws",
                    "event_type": "system",
                    "event_name": "ws_disconnect",
                    "ws_event_type": "disconnect",
                    "payload_bytes": out_bytes + in_bytes,
                    "in_bytes": in_bytes,
                    "out_bytes": out_bytes,
                },
            )
            clear_ctx()
            return
        except Exception:
            await send_with_log(
                build_ws_payload(
                    {
                        "type": "error",
                        "code": "SERVER_ERROR",
                        "message": "Unexpected server error",
                    },
                    thread_id,
                    sequence=0,
                )
            )
            await websocket.close(code=1011)
            log_event(
                "ws",
                {
                    "source": "backend",
                    "component": "ws",
                    "event_type": "system",
                    "event_name": "ws_disconnect",
                    "ws_event_type": "disconnect",
                    "payload_bytes": out_bytes + in_bytes,
                    "in_bytes": in_bytes,
                    "out_bytes": out_bytes,
                },
            )
            clear_ctx()
            return
    finally:
        # 确保断开时注销连接
        await ws_manager.unregister(thread_id)


router.include_router(v1_router)
router.include_router(trace_router)

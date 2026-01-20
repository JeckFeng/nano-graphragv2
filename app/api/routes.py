"""API routes for the MVP service."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.agent_runner import TopSupervisorRunner
from app.infra.db import get_session
from app.schemas.requests import CreateConversationRequest
from app.schemas.responses import (
    ConversationListResponse,
    ConversationSummaryResponse,
    CreateConversationResponse,
    HealthResponse,
    MessageListResponse,
    MessageResponse,
)
from app.services.artifact_service import ArtifactService
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from config.settings import get_settings

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
    if "code" in event:
        payload["code"] = event["code"]
    if "message" in event:
        payload["message"] = event["message"]
    return payload


@v1_router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Handle WebSocket chat connections for streaming responses.

    Args:
        websocket: WebSocket connection.
    """
    await websocket.accept()
    user_id = websocket.query_params.get("user_id")
    thread_id = websocket.query_params.get("thread_id")
    if not user_id or not thread_id:
        await websocket.close(code=1008)
        return
    try:
        thread_uuid = UUID(thread_id)
    except ValueError:
        await websocket.close(code=1008)
        return

    settings = get_settings()
    runner = TopSupervisorRunner(settings.langgraph_memory_database_url)

    async for session in get_session():
        service = ChatService(session, runner)
        try:
            while True:
                payload = await websocket.receive_json()
                if payload.get("type") != "user_message":
                    await websocket.send_json(
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
                    await websocket.send_json(
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

                sequence = 0
                async for event in service.stream_chat(
                    external_user_id=user_id,
                    thread_id=thread_uuid,
                    user_content=content,
                ):
                    sequence += 1
                    await websocket.send_json(
                        build_ws_payload(event, thread_id, sequence=sequence)
                    )
        except WebSocketDisconnect:
            return
        except PermissionError:
            await websocket.send_json(
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
            return
        except Exception:
            await websocket.send_json(
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
            return


router.include_router(v1_router)

"""Response schemas for the MVP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response payload."""

    status: str = Field(..., examples=["ok"])


class CreateConversationResponse(BaseModel):
    """Response payload for conversation creation."""

    thread_id: str = Field(..., examples=["uuid"])
    created_at: datetime


class ConversationSummaryResponse(BaseModel):
    """Summary response for a conversation thread."""

    thread_id: str = Field(..., examples=["uuid"])
    title: Optional[str]
    created_at: datetime


class ConversationListResponse(BaseModel):
    """Response payload for listing conversations."""

    conversations: List[ConversationSummaryResponse]


class MessageResponse(BaseModel):
    """Response payload for a conversation message."""

    id: int
    role: str
    content: str
    run_id: Optional[str]
    tool_payload: Optional[dict]
    created_at: datetime


class MessageListResponse(BaseModel):
    """Response payload for listing messages."""

    messages: List[MessageResponse]


class ApprovalInterruptResponse(BaseModel):
    """Interrupt payload for approval responses."""

    interrupt_id: str
    action_requests: List[dict]
    review_configs: List[dict]


class ApprovalRecordResponse(BaseModel):
    """Approval record response payload."""

    approval_id: str
    thread_id: str
    user_id: str
    status: str
    created_at: datetime
    resolved_at: Optional[datetime]
    interrupts: List[ApprovalInterruptResponse]
    decision: Optional[str]
    result_content: Optional[str]


class ApprovalListResponse(BaseModel):
    """Response payload for listing approvals."""

    approvals: List[ApprovalRecordResponse]


class ApprovalResolutionResponse(BaseModel):
    """Response payload for approval resolution."""

    approval_id: str
    status: str
    result_content: Optional[str]
    next_approval: Optional[ApprovalRecordResponse]


class TraceEventResponse(BaseModel):
    """Response payload for a trace event."""

    id: int
    event_time: datetime
    request_id: Optional[str]
    thread_id: str
    run_id: Optional[str]
    message_id: Optional[int]
    user_id: Optional[str]
    event_type: str
    event_name: str
    trace_kind: str
    phase: str
    source: str
    component: str
    tool_name: Optional[str]
    subagent_type: Optional[str]
    ok: Optional[bool]
    latency_ms: Optional[int]
    error: Optional[str]
    payload: Dict[str, Any]
    seq: Optional[int]


class TraceListResponse(BaseModel):
    """Response payload for listing trace events."""

    thread_id: Optional[str]
    run_id: Optional[str]
    limit: int
    offset: int
    traces: List[TraceEventResponse]

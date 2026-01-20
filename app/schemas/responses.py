"""Response schemas for the MVP API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

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
    created_at: datetime


class MessageListResponse(BaseModel):
    """Response payload for listing messages."""

    messages: List[MessageResponse]

"""Request schemas for the MVP API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    """Request payload for creating a conversation."""

    user_id: str = Field(..., min_length=1, max_length=128)
    title: Optional[str] = Field(default=None, max_length=256)

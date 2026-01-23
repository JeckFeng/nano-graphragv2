"""Database models for the MVP API service."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config.settings import get_settings

def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp.

    Returns:
        datetime: Current UTC time with timezone info.
    """
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


BUSINESS_SCHEMA = get_settings().business_schema


class User(Base):
    """User record.

    Invariants:
        - external_user_id is unique and non-empty.
    """

    __tablename__ = "users"
    __table_args__ = {"schema": BUSINESS_SCHEMA}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    external_user_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    threads: Mapped[List[Thread]] = relationship(
        "Thread",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Thread(Base):
    """Conversation thread record.

    Invariants:
        - user_id must reference a valid user.
    """

    __tablename__ = "threads"
    __table_args__ = {"schema": BUSINESS_SCHEMA}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey(f"{BUSINESS_SCHEMA}.users.id"), nullable=False
    )
    title: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    user: Mapped[User] = relationship("User", back_populates="threads")
    messages: Mapped[List[Message]] = relationship(
        "Message",
        back_populates="thread",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """Conversation message record.

    Invariants:
        - thread_id must reference a valid thread.
        - role should be one of: user, assistant, tool.
    """

    __tablename__ = "messages"
    __table_args__ = {"schema": BUSINESS_SCHEMA}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    thread_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(f"{BUSINESS_SCHEMA}.threads.id"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_payload: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    thread: Mapped[Thread] = relationship("Thread", back_populates="messages")
    artifacts: Mapped[List["Artifact"]] = relationship(
        "Artifact",
        back_populates="message",
        cascade="all, delete-orphan",
    )


class Artifact(Base):
    """Artifact record for large payloads.

    Invariants:
        - thread_id must reference a valid thread.
        - message_id must reference a valid message.
        - storage_path must point to an accessible location.
    """

    __tablename__ = "artifacts"
    __table_args__ = {"schema": BUSINESS_SCHEMA}

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thread_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(f"{BUSINESS_SCHEMA}.threads.id"),
        nullable=False,
    )
    message_id: Mapped[int] = mapped_column(
        ForeignKey(f"{BUSINESS_SCHEMA}.messages.id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    media_type: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    message: Mapped[Message] = relationship("Message", back_populates="artifacts")


class TraceEvent(Base):
    """Trace event record for agent reasoning visualization.

    Invariants:
        - trace_kind is within the supported enum set.
        - phase is within the supported enum set.
        - thread_id references a valid conversation thread.
    """

    __tablename__ = "trace_events"
    __table_args__ = (
        CheckConstraint(
            "trace_kind IN ('todo_update','tool_span','subagent_dispatch','hitl_interrupt','hitl_resume')",
            name="trace_events_kind_chk",
        ),
        CheckConstraint(
            "phase IN ('start','end','update','error','pending','resume')",
            name="trace_events_phase_chk",
        ),
        Index("trace_events_thread_time_idx", "thread_id", "event_time"),
        Index("trace_events_run_seq_idx", "run_id", "seq"),
        Index("trace_events_user_time_idx", "user_id", "event_time"),
        {"schema": BUSINESS_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    thread_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    message_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    event_type: Mapped[str] = mapped_column(String(32), default="trace", nullable=False)
    event_name: Mapped[str] = mapped_column(String(64), default="trace_event", nullable=False)
    trace_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    phase: Mapped[str] = mapped_column(String(16), nullable=False)

    source: Mapped[str] = mapped_column(String(32), default="backend", nullable=False)
    component: Mapped[str] = mapped_column(String(32), default="agent", nullable=False)

    tool_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    subagent_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    ok: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    seq: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

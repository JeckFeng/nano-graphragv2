"""Trace API routes for history replay."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.db import get_session
from app.infra.models import TraceEvent
from app.schemas.responses import TraceEventResponse, TraceListResponse
from app.services.trace_service import TraceService

router = APIRouter(prefix="/v1", tags=["traces"])


def _build_trace_response(trace: TraceEvent) -> TraceEventResponse:
    """Convert a TraceEvent model into a response payload.

    Args:
        trace: Trace event ORM instance.

    Returns:
        TraceEventResponse: Response payload.
    """
    return TraceEventResponse(
        id=trace.id,
        event_time=trace.event_time,
        request_id=trace.request_id,
        thread_id=str(trace.thread_id),
        run_id=trace.run_id,
        message_id=trace.message_id,
        user_id=trace.user_id,
        event_type=trace.event_type,
        event_name=trace.event_name,
        trace_kind=trace.trace_kind,
        phase=trace.phase,
        source=trace.source,
        component=trace.component,
        tool_name=trace.tool_name,
        subagent_type=trace.subagent_type,
        ok=trace.ok,
        latency_ms=trace.latency_ms,
        error=trace.error,
        payload=trace.payload or {},
        seq=trace.seq,
    )


@router.get(
    "/conversations/{thread_id}/traces",
    response_model=TraceListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_traces_by_thread(
    thread_id: UUID,
    user_id: str = Query(..., min_length=1, max_length=128),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    run_id: Optional[str] = Query(default=None),
    message_id: Optional[int] = Query(default=None),
    trace_kind: Optional[str] = Query(default=None),
    phase: Optional[str] = Query(default=None),
    order: str = Query(default="asc", pattern="^(asc|desc)$"),
    session: AsyncSession = Depends(get_session),
) -> TraceListResponse:
    """List trace events for a conversation thread.

    Args:
        thread_id: Thread identifier.
        user_id: External user identifier.
        limit: Maximum number of records.
        offset: Pagination offset.
        run_id: Optional run filter.
        message_id: Optional message filter.
        trace_kind: Optional trace kind filter.
        phase: Optional phase filter.
        order: Sorting order.
        session: Database session dependency.

    Returns:
        TraceListResponse: Response payload.
    """
    service = TraceService(session)
    traces = await service.list_by_thread(
        user_id=user_id,
        thread_id=thread_id,
        limit=limit,
        offset=offset,
        run_id=run_id,
        message_id=message_id,
        trace_kind=trace_kind,
        phase=phase,
        order=order,
    )
    if traces is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    return TraceListResponse(
        thread_id=str(thread_id),
        run_id=run_id,
        limit=limit,
        offset=offset,
        traces=[_build_trace_response(trace) for trace in traces],
    )


@router.get(
    "/runs/{run_id}/traces",
    response_model=TraceListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_traces_by_run(
    run_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    order: str = Query(default="asc", pattern="^(asc|desc)$"),
    session: AsyncSession = Depends(get_session),
) -> TraceListResponse:
    """List trace events for a run.

    Args:
        run_id: Run identifier.
        user_id: External user identifier.
        limit: Maximum number of records.
        offset: Pagination offset.
        order: Sorting order.
        session: Database session dependency.

    Returns:
        TraceListResponse: Response payload.
    """
    service = TraceService(session)
    traces = await service.list_by_run(
        user_id=user_id,
        run_id=run_id,
        limit=limit,
        offset=offset,
        order=order,
    )
    if traces is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return TraceListResponse(
        thread_id=None,
        run_id=run_id,
        limit=limit,
        offset=offset,
        traces=[_build_trace_response(trace) for trace in traces],
    )

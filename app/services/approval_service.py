"""Approval service providing human-in-the-loop review support."""

from __future__ import annotations

import asyncio
import copy
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from agents.top_supervisor import create_top_supervisor
from config.settings import get_settings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command


@dataclass
class ApprovalInterrupt:
    """Interrupt payload for an approval record."""

    interrupt_id: str
    action_requests: List[Dict[str, Any]]
    review_configs: List[Dict[str, Any]]


@dataclass
class ApprovalRecord:
    """Approval record for human review.

    Invariants:
        - approval_id is unique.
        - status is one of pending/approved/rejected/edited.
    """

    approval_id: str
    thread_id: str
    user_id: str
    status: str
    created_at: datetime
    resolved_at: Optional[datetime]
    interrupts: List[ApprovalInterrupt]
    decision: Optional[str]
    result_content: Optional[str]


@dataclass
class ApprovalResolution:
    """Resolution result returned after an approval decision."""

    approval_id: str
    status: str
    result_content: Optional[str]
    next_approval: Optional[ApprovalRecord]


class ApprovalStore:
    """In-memory approval store with resolve locking.

    Invariants:
        - Store is process-local; data is not persisted across restarts.
        - Only one resolve operation per approval_id at a time.
    """

    def __init__(self) -> None:
        """Initialize the approval store."""
        self._records: Dict[str, ApprovalRecord] = {}
        self._lock = asyncio.Lock()
        self._resolving_ids: Set[str] = set()

    async def try_start_resolve(self, approval_id: str) -> bool:
        """Try to acquire resolve lock for an approval.

        Args:
            approval_id: Approval identifier.

        Returns:
            bool: True if lock acquired, False if already resolving.
        """
        async with self._lock:
            if approval_id in self._resolving_ids:
                return False
            self._resolving_ids.add(approval_id)
            return True

    async def finish_resolve(self, approval_id: str) -> None:
        """Release resolve lock for an approval.

        Args:
            approval_id: Approval identifier.
        """
        async with self._lock:
            self._resolving_ids.discard(approval_id)

    async def create(self, record: ApprovalRecord) -> ApprovalRecord:
        """Create and store an approval record.

        Args:
            record: Approval record to store.

        Returns:
            ApprovalRecord: Stored record.
        """
        async with self._lock:
            self._records[record.approval_id] = record
        return record

    async def get(self, approval_id: str) -> Optional[ApprovalRecord]:
        """Fetch an approval record by id.

        Args:
            approval_id: Approval identifier.

        Returns:
            Optional[ApprovalRecord]: Record if found.
        """
        async with self._lock:
            return self._records.get(approval_id)

    async def list(
        self,
        user_id: str,
        status: Optional[str],
        thread_id: Optional[str],
        limit: int,
        offset: int,
    ) -> List[ApprovalRecord]:
        """List approvals for a user.

        Args:
            user_id: External user identifier.
            status: Optional status filter.
            thread_id: Optional thread filter.
            limit: Max number of records.
            offset: Pagination offset.

        Returns:
            List[ApprovalRecord]: Matching approvals.
        """
        async with self._lock:
            records = [record for record in self._records.values() if record.user_id == user_id]

        if status:
            records = [record for record in records if record.status == status]
        if thread_id:
            records = [record for record in records if record.thread_id == thread_id]

        records.sort(key=lambda item: item.created_at, reverse=True)
        return records[offset : offset + limit]

    async def update(self, record: ApprovalRecord) -> None:
        """Update an approval record in the store.

        Args:
            record: Updated approval record.
        """
        async with self._lock:
            self._records[record.approval_id] = record


_STORE = ApprovalStore()


class ApprovalService:
    """Service for managing human-in-the-loop approvals."""

    def __init__(self) -> None:
        """Initialize approval service."""
        settings = get_settings()
        self._db_uri = settings.langgraph_memory_database_url

    async def create_approval(
        self,
        user_id: str,
        thread_id: str,
        interrupts_list: List[Any],
    ) -> ApprovalRecord:
        """Create an approval record from LangGraph interrupts.

        Args:
            user_id: External user identifier.
            thread_id: Thread identifier.
            interrupts_list: Interrupt objects returned by LangGraph.

        Returns:
            ApprovalRecord: Created approval record.
        """
        interrupt_records: List[ApprovalInterrupt] = []
        for interrupt_obj in interrupts_list:
            interrupts = interrupt_obj.value
            action_requests = copy.deepcopy(interrupts.get("action_requests", []))
            review_configs = copy.deepcopy(interrupts.get("review_configs", []))
            interrupt_records.append(
                ApprovalInterrupt(
                    interrupt_id=str(interrupt_obj.id),
                    action_requests=action_requests,
                    review_configs=review_configs,
                )
            )

        record = ApprovalRecord(
            approval_id=str(uuid.uuid4()),
            thread_id=thread_id,
            user_id=user_id,
            status="pending",
            created_at=datetime.now(timezone.utc),
            resolved_at=None,
            interrupts=interrupt_records,
            decision=None,
            result_content=None,
        )
        return await _STORE.create(record)

    async def list_approvals(
        self,
        user_id: str,
        status: Optional[str],
        thread_id: Optional[str],
        limit: int,
        offset: int,
    ) -> List[ApprovalRecord]:
        """List approvals for a user.

        Args:
            user_id: External user identifier.
            status: Optional status filter.
            thread_id: Optional thread filter.
            limit: Max number of records.
            offset: Pagination offset.

        Returns:
            List[ApprovalRecord]: Approval records.
        """
        return await _STORE.list(user_id, status, thread_id, limit, offset)

    async def get_approval(self, user_id: str, approval_id: str) -> Optional[ApprovalRecord]:
        """Fetch an approval record if owned by the user.

        Args:
            user_id: External user identifier.
            approval_id: Approval identifier.

        Returns:
            Optional[ApprovalRecord]: Approval record if found.
        """
        record = await _STORE.get(approval_id)
        if record and record.user_id == user_id:
            return record
        return None

    async def resolve_approval(
        self,
        user_id: str,
        approval_id: str,
        decision: str,
        edited_args: Optional[Dict[str, Any]] = None,
    ) -> Optional[ApprovalResolution]:
        """Resolve an approval by resuming the agent execution.

        Args:
            user_id: External user identifier.
            approval_id: Approval identifier.
            decision: Decision type (approve/reject/edit).
            edited_args: Optional edited tool args for edit decision.

        Returns:
            Optional[ApprovalResolution]: Resolution result if approval found.
        """
        record = await self.get_approval(user_id, approval_id)
        if record is None:
            return None
        if record.status != "pending":
            return ApprovalResolution(
                approval_id=record.approval_id,
                status=record.status,
                result_content=record.result_content,
                next_approval=None,
            )

        # 尝试获取处理锁
        if not await _STORE.try_start_resolve(approval_id):
            return ApprovalResolution(
                approval_id=record.approval_id,
                status="processing",
                result_content=None,
                next_approval=None,
            )

        try:
            resume_map = self._build_resume_map(record, decision, edited_args)
            result = await self._resume_agent(record.thread_id, resume_map)
            record.decision = decision
            record.resolved_at = datetime.now(timezone.utc)

            if result.get("__interrupt__"):
                record.status = self._map_decision_status(decision)
                await _STORE.update(record)
                next_record = await self.create_approval(
                    user_id,
                    record.thread_id,
                    result["__interrupt__"],
                )
                return ApprovalResolution(
                    approval_id=record.approval_id,
                    status=record.status,
                    result_content=None,
                    next_approval=next_record,
                )

            content = ""
            if result.get("messages"):
                content = result["messages"][-1].content
            record.status = self._map_decision_status(decision)
            record.result_content = content
            await _STORE.update(record)
            return ApprovalResolution(
                approval_id=record.approval_id,
                status=record.status,
                result_content=content,
                next_approval=None,
            )
        finally:
            await _STORE.finish_resolve(approval_id)

    async def _resume_agent(self, thread_id: str, resume_map: Dict[str, Any]) -> dict:
        """Resume the agent execution using LangGraph Command.

        Args:
            thread_id: Thread identifier.
            resume_map: Resume map payload for Command.

        Returns:
            dict: Agent execution result.
        """
        config = {"configurable": {"thread_id": thread_id}}
        async with AsyncPostgresSaver.from_conn_string(self._db_uri) as checkpointer:
            await checkpointer.setup()
            agent, _ = create_top_supervisor(checkpointer)
            return await agent.ainvoke(Command(resume=resume_map), config=config)

    @staticmethod
    def _build_resume_map(
        record: ApprovalRecord,
        decision: str,
        edited_args: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build resume map for LangGraph Command.

        Args:
            record: Approval record.
            decision: Decision type.
            edited_args: Edited tool args for edit decision.

        Returns:
            Dict[str, Any]: Resume map payload.
        """
        resume_map: Dict[str, Any] = {}
        for interrupt in record.interrupts:
            decisions: List[Dict[str, Any]] = []
            for action in interrupt.action_requests:
                if decision == "edit":
                    decisions.append(
                        {
                            "type": "edit",
                            "args": edited_args or action.get("args", {}),
                        }
                    )
                else:
                    decisions.append({"type": decision})
            resume_map[interrupt.interrupt_id] = {"decisions": decisions}
        return resume_map

    @staticmethod
    def _map_decision_status(decision: str) -> str:
        """Map decision type to status.

        Args:
            decision: Decision type.

        Returns:
            str: Status string.
        """
        if decision == "reject":
            return "rejected"
        if decision == "edit":
            return "edited"
        return "approved"

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
        """Submit approval decision and start async execution.

        Args:
            user_id: External user identifier.
            approval_id: Approval identifier.
            decision: Decision type (approve/reject/edit).
            edited_args: Optional edited tool args for edit decision.

        Returns:
            Optional[ApprovalResolution]: status="processing" indicates execution started.
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

        # 构建 resume_map
        resume_map = self._build_resume_map(record, decision, edited_args)

        # 更新状态为处理中
        record.decision = decision
        record.status = "processing"
        await _STORE.update(record)

        # 启动后台任务（不等待）
        asyncio.create_task(self._execute_resume(record, resume_map, user_id))

        return ApprovalResolution(
            approval_id=record.approval_id,
            status="processing",
            result_content=None,
            next_approval=None,
        )

    async def _execute_resume(
        self,
        record: ApprovalRecord,
        resume_map: Dict[str, Any],
        user_id: str,
    ) -> None:
        """Execute agent resume in background, push results via WebSocket.

        Args:
            record: Approval record being resolved.
            resume_map: Resume map payload for Command.
            user_id: External user identifier.
        """
        from app.application.agent_runner import TopSupervisorRunner
        from app.observability import clear_ctx, get_ctx, set_ctx_ws, set_message_id, set_run_id
        from app.services.ws_manager import ws_manager

        thread_id = record.thread_id
        runner = TopSupervisorRunner(self._db_uri)

        # 注入 trace 上下文，确保后续 trace 事件有完整的 thread_id/user_id
        set_ctx_ws(
            request_id=uuid.uuid4().hex,
            user_id=user_id,
            thread_id=thread_id,
            conn_id=None,
        )
        set_message_id(None)
        set_run_id(uuid.uuid4().hex)
        run_id = get_ctx().get("run_id")

        try:
            config = {"configurable": {"thread_id": thread_id}}
            async with AsyncPostgresSaver.from_conn_string(self._db_uri) as checkpointer:
                await checkpointer.setup()
                agent, _ = create_top_supervisor(checkpointer)

                buffered_tokens: list[str] = []
                sequence = 0

                async for event in agent.astream_events(
                    Command(resume=resume_map),
                    config=config,
                ):
                    # 复用现有的中断检测逻辑（支持 GraphInterrupt/on_tool_error）
                    interrupts = TopSupervisorRunner._extract_interrupts(event)
                    if interrupts:
                        await self._handle_new_interrupt(
                            record, interrupts, user_id, sequence
                        )
                        return

                    # 复用现有的事件映射逻辑（过滤 tools: 命名空间）
                    mapped = runner._map_event(event)
                    if mapped:
                        sequence += 1
                        mapped["sequence"] = sequence
                        mapped["thread_id"] = thread_id
                        if run_id:
                            mapped["run_id"] = run_id

                        delta = mapped.get("delta", "")
                        if delta:
                            buffered_tokens.append(delta)

                        await ws_manager.send_to_thread(thread_id, mapped)

                # 执行完成
                content = "".join(buffered_tokens)
                record.status = self._map_decision_status(record.decision or "approve")
                record.result_content = content
                record.resolved_at = datetime.now(timezone.utc)
                await _STORE.update(record)

                # 保存消息到数据库
                await self._save_message_to_db(thread_id, content)

                sequence += 1
                await ws_manager.send_to_thread(thread_id, {
                    "type": "final",
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "content": content,
                    "sequence": sequence,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        except Exception as e:
            await ws_manager.send_to_thread(thread_id, {
                "type": "error",
                "thread_id": thread_id,
                "code": "RESUME_ERROR",
                "message": str(e),
            })
        finally:
            await _STORE.finish_resolve(record.approval_id)
            clear_ctx()

    async def _handle_new_interrupt(
        self,
        record: ApprovalRecord,
        interrupts: list,
        user_id: str,
        sequence: int,
    ) -> None:
        """Handle new interrupt generated during resume execution.

        Args:
            record: Current approval record.
            interrupts: New interrupt objects.
            user_id: External user identifier.
            sequence: Current event sequence number.
        """
        from app.services.ws_manager import ws_manager

        # 更新当前记录状态
        record.status = self._map_decision_status(record.decision or "approve")
        record.resolved_at = datetime.now(timezone.utc)
        await _STORE.update(record)

        # 创建新的审批记录
        new_approval = await self.create_approval(
            user_id=user_id,
            thread_id=record.thread_id,
            interrupts_list=interrupts,
        )

        # 推送 approval_required 事件
        await ws_manager.send_to_thread(record.thread_id, {
            "type": "approval_required",
            "thread_id": record.thread_id,
            "approval_id": new_approval.approval_id,
            "status": new_approval.status,
            "sequence": sequence + 1,
            "interrupts": [
                {
                    "interrupt_id": i.interrupt_id,
                    "action_requests": i.action_requests,
                    "review_configs": i.review_configs,
                }
                for i in new_approval.interrupts
            ],
        })

    async def _save_message_to_db(self, thread_id: str, content: str) -> None:
        """Save assistant message to database.

        Args:
            thread_id: Thread identifier.
            content: Message content.
        """
        if not content:
            return

        from uuid import UUID

        from app.infra.db import ASYNC_SESSION_FACTORY
        from app.observability import get_ctx
        from app.services.message_service import MessageService

        async with ASYNC_SESSION_FACTORY() as session:
            message_service = MessageService(session)
            run_id = get_ctx().get("run_id")
            await message_service.append_message(
                UUID(thread_id),
                "assistant",
                content,
                run_id=run_id,
            )

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

"""WebSocket connection manager for thread-based message routing."""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

from fastapi import WebSocket


class WebSocketManager:
    """Manage WebSocket connections by thread_id.

    Invariants:
        - One connection per thread_id at a time.
        - Thread-safe for concurrent access.
    """

    def __init__(self) -> None:
        """Initialize the WebSocket manager."""
        self._connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def register(self, thread_id: str, websocket: WebSocket) -> None:
        """Register a WebSocket connection for a thread.

        Args:
            thread_id: Thread identifier.
            websocket: WebSocket connection instance.
        """
        async with self._lock:
            self._connections[thread_id] = websocket

    async def unregister(self, thread_id: str) -> None:
        """Unregister a WebSocket connection.

        Args:
            thread_id: Thread identifier.
        """
        async with self._lock:
            self._connections.pop(thread_id, None)

    async def send_to_thread(self, thread_id: str, event: dict) -> bool:
        """Send event to a specific thread's WebSocket.

        Args:
            thread_id: Thread identifier.
            event: Event payload to send.

        Returns:
            bool: True if sent successfully, False if no connection.
        """
        async with self._lock:
            ws = self._connections.get(thread_id)

        if ws is None:
            return False

        try:
            await ws.send_json(event)
            return True
        except Exception:
            return False

    def get_connection(self, thread_id: str) -> Optional[WebSocket]:
        """Get WebSocket connection for a thread (non-async).

        Args:
            thread_id: Thread identifier.

        Returns:
            Optional[WebSocket]: WebSocket connection if exists.
        """
        return self._connections.get(thread_id)


# 全局单例
ws_manager = WebSocketManager()

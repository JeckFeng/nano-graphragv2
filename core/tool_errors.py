"""
工具错误类型定义模块。

提供 ToolError 作为工具层统一异常类型，用于向上层传递结构化错误信息。
设计约束：
- ToolError 的 message 必须可读且非空；
- cause 仅用于日志记录，不应直接暴露给 Agent。
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ToolError(Exception):
    """
    工具层统一异常类型。

    职责：
    - 表达工具执行过程中可预期的失败；
    - 提供可序列化的错误结构，便于日志记录与 Agent 决策。

    重要不变量（invariants）：
    - message 为非空字符串；
    - details 始终为字典对象（无信息时为空字典）。
    """

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        初始化 ToolError。

        Args:
            message: 面向 Agent 的错误说明（简明可读）
            code: 稳定的机器可识别错误码（用于分支逻辑）
            details: 结构化附加信息（用于日志或排查）
            cause: 原始异常（仅用于日志，不应直接返回给 Agent）
        """
        safe_message = message.strip() if isinstance(message, str) else ""
        if not safe_message:
            safe_message = "Unknown tool error"
        super().__init__(safe_message)
        self.message = safe_message
        self.code = code
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为可序列化的错误结构（不包含 cause）。

        Returns:
            Dict[str, Any]: 适合返回给 Agent 的错误结构
        """
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details,
            "error_type": self.__class__.__name__,
        }

    def to_log_dict(self) -> Dict[str, Any]:
        """
        转换为日志用错误结构（包含 cause）。

        Returns:
            Dict[str, Any]: 适合写入日志的错误结构
        """
        payload = self.to_dict()
        if self.cause is not None:
            payload["cause"] = repr(self.cause)
        return payload

    def __str__(self) -> str:
        """
        获取字符串表示。

        Returns:
            str: 错误说明（必要时附带错误码）
        """
        return f"{self.message} (code={self.code})" if self.code else self.message

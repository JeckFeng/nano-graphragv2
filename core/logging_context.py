"""
日志上下文管理模块

提供线程/协程安全的上下文变量，用于在日志中记录 session_id 和 user_id
"""

from contextvars import ContextVar
from typing import Optional

# 上下文变量定义（线程/协程安全）
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


def set_session_id(session_id: str) -> None:
    """设置当前会话 ID"""
    session_id_var.set(session_id)


def set_user_id(user_id: str) -> None:
    """设置当前用户 ID"""
    user_id_var.set(user_id)


def get_session_id() -> Optional[str]:
    """获取当前会话 ID"""
    return session_id_var.get()


def get_user_id() -> Optional[str]:
    """获取当前用户 ID"""
    return user_id_var.get()


__all__ = [
    "session_id_var",
    "user_id_var",
    "set_session_id",
    "set_user_id",
    "get_session_id",
    "get_user_id",
]

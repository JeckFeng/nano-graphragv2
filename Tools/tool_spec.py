"""
工具规范定义模块。

用途：
- 提供轻量的 ToolSpec 结构，用于描述工具元数据与处理函数。

设计约束：
- 仅承载元数据，不做注册、发现或执行逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class ToolSpec:
    """
    工具元数据规范。

    职责：
    - 统一描述工具名称、说明、参数与处理函数。

    重要不变量：
    - name/description 为非空字符串；
    - parameters 为字典；
    - handler 可调用。
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTool(ABC):
    """工具基类"""
    name: str  # 工具名、类名
    description: str
    parameters: Dict[str, Any]  # 如果需要参数，需要进行参数描述。不需要参数，定义为{}
    error: Optional[str] = None  # 如果没有成功，则返回错误信息，否则不用显示
    isSuccess: bool  # 工具是否调用成功
    message: str  # 返回成功调用语句

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """工具调用接口"""
        raise NotImplementedError("子类必须实现__call__方法")

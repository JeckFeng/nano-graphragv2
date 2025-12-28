"""
统一的工具定义和管理模块

提供 @context_tool 装饰器，用于统一管理工具的超时控制、调用次数限制和日志记录。
兼容 LangChain 的 @tool 装饰器。

使用示例:
    from core.tool_context import context_tool
    
    @context_tool
    async def my_tool(query: str) -> str:
        '''工具描述'''
        return "result"
    
    @context_tool("custom_name", timeout=30.0, max_calls=10)
    async def another_tool(param: str) -> str:
        '''另一个工具'''
        return "result"
"""

import asyncio
import concurrent.futures
import json
import logging
import time
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.tools import tool

from config.tool_config import get_tool_config
from core.logger_config import get_logger_config

logger = logging.getLogger(__name__)

# ================================
# 上下文变量定义（线程/协程安全）
# ================================
_call_logs: ContextVar[List[Dict[str, Any]]] = ContextVar("call_logs", default=[])
_current_agent_name: ContextVar[str] = ContextVar("current_agent_name", default="unknown")
_current_worker_name: ContextVar[Optional[str]] = ContextVar("current_worker_name", default=None)

# 从 logger.yaml 读取 tool 日志配置
_tool_log_config = None

def _get_tool_log_config() -> Dict[str, Any]:
    """获取 tool 日志配置（懒加载）"""
    global _tool_log_config
    if _tool_log_config is None:
        try:
            _tool_log_config = get_logger_config("tool")
        except Exception as e:
            logger.warning(f"读取 tool 日志配置失败: {e}，使用默认配置")
            _tool_log_config = {
                "log_dir": "Logs/tool_logs",
                "file_pattern": "tool_{tool_name}_{date}.log",
            }
    return _tool_log_config

# 日志保存路径（从配置读取）
def _get_log_dir() -> Path:
    """获取日志目录路径"""
    config = _get_tool_log_config()
    log_dir = config.get("log_dir", "Logs/tool_logs")
    return Path(log_dir)

LOG_DIR = _get_log_dir()  # 保持向后兼容

# 默认配置（如果配置文件不存在时使用）
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_CALLS = 5


# =================================
# 公共日志工具函数
# =================================


def set_current_agent_name(agent_name: str) -> None:
    """
    在调用工具前设置当前 Agent 名称
    
    Args:
        agent_name: Agent 名称
    """
    _current_agent_name.set(agent_name)


def set_current_worker_name(worker_name: Optional[str]) -> None:
    """
    在调用工具前设置当前 Worker 名称
    
    Args:
        worker_name: Worker 名称，如果为 None 则清除
    """
    _current_worker_name.set(worker_name)


def get_tool_logs() -> List[Dict[str, Any]]:
    """
    获取当前上下文下的所有工具调用日志记录
    
    Returns:
        List[Dict[str, Any]]: 日志记录列表
    """
    return _call_logs.get()


def _json_default(obj: Any) -> Any:
    """
    JSON 序列化兜底函数
    
    - UUID -> str
    - 其他类型 -> repr(obj)
    
    Args:
        obj: 待序列化的对象
    
    Returns:
        可序列化的对象
    """
    if isinstance(obj, UUID):
        return str(obj)
    # 可以继续扩展，例如 BaseModel 等
    return repr(obj)


def _persist_log_sync(log_entry: Dict[str, Any]) -> None:
    """
    同步持久化日志到文件（在后台线程中执行）
    
    日志保存到 {log_dir}/{tool_name}/log_{timestamp}.json
    使用 NDJSON 格式（每行一个 JSON 对象）
    
    Args:
        log_entry: 日志条目字典
    """
    try:
        config = _get_tool_log_config()
        log_dir = Path(config.get("log_dir", "Logs/tool_logs"))
        tool_name = log_entry.get("tool_name", "unknown")
        tool_log_dir = log_dir / tool_name
        tool_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用时间戳命名文件（按小时分组，便于管理）
        timestamp = int(time.time())
        hour_timestamp = timestamp - (timestamp % 3600)  # 整点时间戳
        log_file = tool_log_dir / f"log_{hour_timestamp}.json"
        
        # 追加模式写入（NDJSON 格式）
        log_line = json.dumps(log_entry, ensure_ascii=False, default=_json_default) + "\n"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line)
    
    except Exception as e:
        logger.error(f"持久化工具调用日志失败: {e}", exc_info=True)


def _record_log(
    tool_name: str,
    agent_name: str,
    function_type: str,
    args: Any,
    kwargs: Any,
    result: Any,
    is_success: bool,
    duration: Optional[float] = None,
) -> None:
    """
    统一追加一条工具调用日志
    
    同时记录到内存上下文和持久化到文件。
    
    Args:
        tool_name: 工具名称
        agent_name: Agent 名称
        function_type: 函数类型（"async" 或 "sync"）
        args: 位置参数
        kwargs: 关键字参数
        result: 返回结果
        is_success: 是否成功
        duration: 执行时长（秒）
    """
    worker_name = _current_worker_name.get()
    
    # 构建输入参数（合并 args 和 kwargs）
    input_params: Dict[str, Any] = {}
    if args:
        input_params["args"] = args
    if kwargs:
        input_params["kwargs"] = kwargs
    
    # 构建日志条目
    log_entry: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": "DEBUG" if is_success else "ERROR",
        "logger": "tool",
        "tool_name": tool_name,
        "agent_name": agent_name,
        "worker_name": worker_name or "unknown",
        "function_type": function_type,
        "input_params": input_params,
        "output_result": result,
        "is_success": is_success,
        "duration": round(duration, 6) if duration is not None else None,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # 记录到内存上下文
    logs = _call_logs.get()
    logs.append(log_entry)
    _call_logs.set(logs)
    
    # 异步持久化到文件（不阻塞，使用线程池）
    try:
        # 使用线程池执行文件写入，避免阻塞主线程
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(_persist_log_sync, log_entry)
        # 不等待完成，让它在后台执行
    except Exception as e:
        logger.error(f"调度日志持久化任务失败: {e}", exc_info=True)


# =================================
# 核心封装器逻辑
# =================================


def _run_sync_with_timeout(func: Callable, timeout: float, *args, **kwargs):
    """
    同步函数超时执行
    
    Args:
        func: 要执行的函数
        timeout: 超时时间（秒）
        *args: 位置参数
        **kwargs: 关键字参数
    
    Returns:
        函数执行结果
    
    Raises:
        concurrent.futures.TimeoutError: 超时异常
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)


def _inject_context(
    func: Callable,
    tool_name: str,
    timeout: Optional[float] = None,
    max_calls: Optional[int] = None,
    worker_name: Optional[str] = None,
) -> Callable:
    """
    为工具函数注入上下文
    
    注入功能：
    - 调用次数限制（异常/超时也计数）
    - 日志记录（包含 time / duration / error / is_success / agent / function_type）
    - 超时保护（超时不抛异常，而是返回错误字典，让 Agent 自己判断）
    - 日志持久化到 /Logs/tool_logs
    - 从 tool.yaml 读取配置（支持多层级优先级）
    
    Args:
        func: 原始工具函数
        tool_name: 工具名称
        timeout: 超时时间（秒），如果为 None 则从配置读取
        max_calls: 最大调用次数，如果为 None 则从配置读取
        worker_name: Worker 名称，用于读取 Worker 级别配置
    
    Returns:
        Callable: 包装后的函数
    """
    # 从配置读取默认值（如果装饰器参数未指定）
    # 注意：这里不能获取 agent_name，因为装饰器在模块加载时执行
    # agent_name 需要在运行时从上下文变量获取
    config = get_tool_config()
    decorator_timeout = timeout  # 保存装饰器参数
    decorator_max_calls = max_calls
    
    is_async = asyncio.iscoroutinefunction(func)
    function_type = "async" if is_async else "sync"
    
    # 每个包装函数自己的调用计数（按 tool_name 维度）
    call_counts: Dict[str, int] = {}
    
    if is_async:
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            agent_name = _current_agent_name.get()
            
            # 运行时确定配置（优先级：装饰器参数 > tool.yaml）
            actual_timeout = decorator_timeout if decorator_timeout is not None else config.get_tool_timeout(tool_name, agent_name)
            actual_max_calls = decorator_max_calls if decorator_max_calls is not None else config.get_tool_max_calls(tool_name, agent_name)
            
            count = call_counts.get(tool_name, 0)
            
            if actual_max_calls is not None and count >= actual_max_calls:
                result = {
                    "error": f"工具 `{tool_name}` 超出最大调用次数：{actual_max_calls}",
                    "call_exhausted": True,
                }
                _record_log(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    function_type=function_type,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    is_success=False,
                )
                # 不抛异常，返回错误对象给 Agent
                return result
            
            start = time.perf_counter()
            try:
                # 有超时保护的实际调用
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=actual_timeout)
                duration = time.perf_counter() - start
                _record_log(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    function_type=function_type,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    is_success=True,
                    duration=duration,
                )
                return result
            except asyncio.TimeoutError:
                result = {
                    "error": f"Timeout after {actual_timeout}s",
                    "timeout": actual_timeout,
                }
                _record_log(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    function_type=function_type,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    is_success=False,
                )
                # 不抛异常，交给上层 Agent 自行处理
                return result
            except Exception as e:
                result = {
                    "error": repr(e),
                }
                _record_log(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    function_type=function_type,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    is_success=False,
                )
                # 同样不抛异常，避免整个图直接崩
                return result
            finally:
                # 无论成功/失败/超时，都计数 +1，防止无限调用
                call_counts[tool_name] = count + 1
        
        wrapper.__annotations__ = getattr(func, "__annotations__", {})
        return wrapper
    
    else:
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            agent_name = _current_agent_name.get()
            
            # 运行时确定配置（优先级：装饰器参数 > tool.yaml）
            actual_timeout = decorator_timeout if decorator_timeout is not None else config.get_tool_timeout(tool_name, agent_name)
            actual_max_calls = decorator_max_calls if decorator_max_calls is not None else config.get_tool_max_calls(tool_name, agent_name)
            
            count = call_counts.get(tool_name, 0)
            
            if actual_max_calls is not None and count >= actual_max_calls:
                result = {
                    "error": f"工具 `{tool_name}` 超出最大调用次数：{actual_max_calls}",
                    "call_exhausted": True,
                }
                _record_log(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    function_type=function_type,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    is_success=False,
                )
                return result
            
            start = time.perf_counter()
            try:
                result = _run_sync_with_timeout(func, actual_timeout, *args, **kwargs)
                duration = time.perf_counter() - start
                _record_log(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    function_type=function_type,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    is_success=True,
                    duration=duration,
                )
                return result
            except concurrent.futures.TimeoutError:
                result = {
                    "error": f"Timeout after {actual_timeout}s",
                    "timeout": actual_timeout,
                }
                _record_log(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    function_type=function_type,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    is_success=False,
                )
                return result
            except Exception as e:
                result = {
                    "error": repr(e),
                }
                _record_log(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    function_type=function_type,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    is_success=False,
                )
                return result
            finally:
                call_counts[tool_name] = count + 1
        
        wrapper.__annotations__ = getattr(func, "__annotations__", {})
        return wrapper


# =================================
# 外部接口：装饰器
# =================================


def context_tool(
    name=None,
    *,
    description: Optional[str] = None,
    timeout: Optional[float] = None,
    max_calls: Optional[int] = None,
    worker_name: Optional[str] = None,
):
    """
    统一的工具定义装饰器
    
    兼容 LangChain 的 @tool 装饰器，同时提供：
    - 超时控制
    - 调用次数限制
    - 工具调用日志记录
    - 日志持久化到 /Logs/tool_logs
    
    配置优先级：
    1. 装饰器参数（timeout, max_calls）
    2. tool.yaml 中的工具级别配置
    3. tool.yaml 中的 Worker 级别配置
    4. tool.yaml 中的全局默认配置
    
    使用示例:
        # 方式1：直接装饰
        @context_tool
        async def my_tool(query: str) -> str:
            '''工具描述'''
            return "result"
        
        # 方式2：指定名称和参数
        @context_tool("custom_name", timeout=30.0, max_calls=10)
        async def another_tool(param: str) -> str:
            '''另一个工具'''
            return "result"
        
        # 方式3：指定 Worker 名称（用于读取 Worker 级别配置）
        @context_tool(worker_name="rag_worker")
        async def search_tool(query: str) -> str:
            '''搜索工具'''
            return "result"
    
    Args:
        name: 工具名称，如果为 None 则使用函数名
        description: 工具描述
        timeout: 超时时间（秒），如果为 None 则从配置读取
        max_calls: 最大调用次数，如果为 None 则从配置读取
        worker_name: Worker 名称，用于读取 Worker 级别配置
    
    Returns:
        Callable: 装饰后的工具函数
    """
    
    def wrap_func(func: Callable):
        # 确定工具名称
        tool_name = name if isinstance(name, str) else func.__name__
        
        # 获取 Worker 名称（优先使用装饰器参数，其次从上下文获取）
        final_worker_name = worker_name or _current_worker_name.get()
        
        # 注入上下文（超时、调用次数、日志）
        wrapped = _inject_context(
            func,
            tool_name=tool_name,
            timeout=timeout,
            max_calls=max_calls,
            worker_name=final_worker_name,
        )
        
        # 兼容 LangChain @tool 装饰器
        if isinstance(name, str):
            return tool(name, description=description)(wrapped)
        else:
            return tool(description=description)(wrapped)
    
    if callable(name):
        # 形如 @context_tool 直接装饰函数的情况
        return wrap_func(name)
    
    # 形如 @context_tool("name", timeout=...) 的情况
    return wrap_func


# =================================
# 导出
# =================================

__all__ = [
    "context_tool",
    "set_current_agent_name",
    "set_current_worker_name",
    "get_tool_logs",
    "LOG_DIR",
    "_call_logs",  # 导出用于 worker 日志记录
]

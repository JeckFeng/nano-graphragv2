"""
高德地图工具模块。

用途：
- 提供驾车路线规划工具，调用高德地图 API 获取行车路线数据。

设计约束：
- 仅负责路线规划，不内置缓存或复杂重试策略。
- API Key 通过参数或环境变量 GAODE_API_KEY 获取。
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import logging
import os
import re

import aiohttp
from dotenv import load_dotenv

from Tools.tool_spec import ToolSpec
from core.tool_errors import ToolError

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_BASE_URL = "https://restapi.amap.com/v5/direction/driving"
_GET_MAX_LENGTH = 2048
_COORD_PATTERN = re.compile(r"^-?\d+\.\d{1,6},-?\d+\.\d{1,6}$")


def _load_api_key(api_key: Optional[str]) -> str:
    """获取高德地图 API Key。

    Args:
        api_key: 显式传入的 API Key，可选。

    Returns:
        可用的 API Key。

    Raises:
        ToolError: 当 API Key 不存在时抛出。
    """
    key = api_key or os.getenv("GAODE_API_KEY")
    if not key:
        raise ToolError(
            "高德 API Key 未配置",
            code="gaode_missing_api_key",
        )
    return key


def _validate_coordinates(coord: str) -> bool:
    """验证经纬度格式。

    Args:
        coord: 经纬度字符串，格式为“经度,纬度”。

    Returns:
        是否符合格式要求。
    """
    return bool(_COORD_PATTERN.match(coord))


def _should_use_post(params: Dict[str, str]) -> bool:
    """判断是否应使用 POST 请求。

    Args:
        params: 请求参数字典。

    Returns:
        是否应使用 POST。
    """
    query = "&".join([f"{key}={value}" for key, value in params.items()])
    return len(f"{_BASE_URL}?{query}") > _GET_MAX_LENGTH


def _safe_int(value: Any) -> int:
    """安全转换为整数。

    Args:
        value: 任意输入值。

    Returns:
        转换后的整数，失败则返回 0。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


async def _fetch_driving_route(
    origin: str,
    destination: str,
    api_key: str,
) -> Dict[str, Any]:
    """调用高德地图 API 获取驾车路线数据。

    Args:
        origin: 起点经纬度。
        destination: 终点经纬度。
        api_key: 高德地图 API Key。

    Returns:
        API 返回的完整路线数据。

    Raises:
        ToolError: 当参数不合法、API 调用失败或网络异常时抛出。
    """
    if not _validate_coordinates(origin):
        raise ToolError(
            "起点经纬度格式错误",
            code="gaode_invalid_origin",
            details={"origin": origin},
        )

    if not _validate_coordinates(destination):
        raise ToolError(
            "终点经纬度格式错误",
            code="gaode_invalid_destination",
            details={"destination": destination},
        )

    params = {
        "key": api_key,
        "origin": origin,
        "destination": destination,
        "show_fields": "polyline,cost",
    }

    try:
        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=20)
            if _should_use_post(params):
                logger.info("参数过长，使用 POST 请求获取路线")
                async with session.post(_BASE_URL, data=params, timeout=timeout) as response:
                    response.raise_for_status()
                    result = await response.json()
            else:
                logger.info("使用 GET 请求获取路线")
                async with session.get(_BASE_URL, params=params, timeout=timeout) as response:
                    response.raise_for_status()
                    result = await response.json()
    except aiohttp.ClientError as exc:
        raise ToolError(
            "高德 API 网络请求失败",
            code="gaode_network_error",
            details={"origin": origin, "destination": destination},
            cause=exc,
        ) from exc
    except Exception as exc:
        raise ToolError(
            "高德 API 响应处理失败",
            code="gaode_response_error",
            details={"origin": origin, "destination": destination},
            cause=exc,
        ) from exc

    if result.get("status") != "1":
        raise ToolError(
            "高德 API 调用失败",
            code="gaode_api_failed",
            details={
                "origin": origin,
                "destination": destination,
                "info": result.get("info", "未知错误"),
                "infocode": result.get("infocode", "未知错误码"),
            },
        )

    return result


async def gaode_driving_route(
    origin: str,
    destination: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """高德驾车路线规划工具。

    Args:
        origin: 起点经纬度，格式为“经度,纬度”。
        destination: 终点经纬度，格式为“经度,纬度”。
        api_key: 高德地图 API Key，可选，默认读取环境变量。

    Returns:
        路线规划结果，包含摘要与完整路线数据。

    Raises:
        ToolError: 当参数不合法、API 调用失败或网络异常时抛出。
    """
    key = _load_api_key(api_key)
    route_data = await _fetch_driving_route(origin, destination, key)

    path = route_data.get("route", {}).get("paths", [{}])[0]
    distance = _safe_int(path.get("distance", 0))
    duration = _safe_int(path.get("duration", 0))

    return {
        "summary": {
            "distance_meters": distance,
            "duration_seconds": duration,
        },
        "route": route_data,
    }


GAODE_DRIVING_ROUTE_SPEC = ToolSpec(
    name="gaode_driving_route",
    description="高德地图驾车路线规划工具，根据起点和终点经纬度获取最优驾车路线。",
    parameters={
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "起点经纬度，格式为'经度,纬度'，小数点后不超过6位。",
                "pattern": r"^-?\d+\.\d{1,6},-?\d+\.\d{1,6}$",
                "examples": ["116.321384,39.904317"],
            },
            "destination": {
                "type": "string",
                "description": "终点经纬度，格式为'经度,纬度'，小数点后不超过6位。",
                "pattern": r"^-?\d+\.\d{1,6},-?\d+\.\d{1,6}$",
                "examples": ["116.587927,40.080102"],
            },
            "api_key": {
                "type": "string",
                "description": "高德地图 API Key，可选。默认读取 GAODE_API_KEY。",
            },
        },
        "required": ["origin", "destination"],
        "additionalProperties": False,
    },
    handler=gaode_driving_route,
)


__all__ = [
    "gaode_driving_route",
    "GAODE_DRIVING_ROUTE_SPEC",
]

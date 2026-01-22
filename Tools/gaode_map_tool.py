"""
高德地图工具模块。

用途：
- 提供驾车路线规划工具，调用高德地图 API 获取行车路线数据。

设计约束：
- 仅负责路线规划，不内置缓存或复杂重试策略。
- API Key 通过参数或环境变量 GAODE_API_KEY 获取。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
import os
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

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
_COORD_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?$")
_COORD_PRECISION = Decimal("0.000001")


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


def _normalize_coordinates(
    coord: str,
    *,
    label: str,
    code: str,
    detail_key: str,
) -> str:
    """规范化经纬度坐标并四舍五入到小数点后 6 位。

    Args:
        coord: 经纬度字符串，格式为“经度,纬度”。
        label: 坐标名称（用于错误提示，如“起点”/“终点”）。
        code: 错误码。
        detail_key: details 字段中的键名。

    Returns:
        规范化后的经纬度字符串，格式为“经度,纬度”。

    Raises:
        ToolError: 当坐标格式错误时抛出。
    """
    text = coord.strip() if coord is not None else ""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ToolError(
            f"{label}经纬度格式错误",
            code=code,
            details={detail_key: coord},
        )

    normalized_text = f"{parts[0]},{parts[1]}"
    if not _COORD_PATTERN.match(normalized_text):
        raise ToolError(
            f"{label}经纬度格式错误",
            code=code,
            details={detail_key: coord},
        )

    try:
        longitude = Decimal(parts[0])
        latitude = Decimal(parts[1])
    except InvalidOperation as exc:
        raise ToolError(
            f"{label}经纬度格式错误",
            code=code,
            details={detail_key: coord},
            cause=exc,
        ) from exc

    longitude = longitude.quantize(_COORD_PRECISION, rounding=ROUND_HALF_UP)
    latitude = latitude.quantize(_COORD_PRECISION, rounding=ROUND_HALF_UP)
    return f"{longitude:.6f},{latitude:.6f}"


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
    origin = _normalize_coordinates(
        origin,
        label="起点",
        code="gaode_invalid_origin",
        detail_key="origin",
    )
    destination = _normalize_coordinates(
        destination,
        label="终点",
        code="gaode_invalid_destination",
        detail_key="destination",
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

    返回结构分为两部分：
    - 摘要和步骤信息：供 LLM 推理使用（不含 polyline）
    - __artifact__：供前端使用的完整数据（含 polyline）

    Args:
        origin: 起点经纬度，格式为"经度,纬度"，超出 6 位小数会自动四舍五入。
        destination: 终点经纬度，格式为"经度,纬度"，超出 6 位小数会自动四舍五入。
        api_key: 高德地图 API Key，可选，默认读取环境变量。

    Returns:
        路线规划结果，包含摘要、步骤和 artifact。

    Raises:
        ToolError: 当参数不合法、API 调用失败或网络异常时抛出。
    """

    key = _load_api_key(api_key)
    route_data = await _fetch_driving_route(origin, destination, key)

    path = route_data.get("route", {}).get("paths", [{}])[0]
    distance = _safe_int(path.get("distance", 0))
    duration = _safe_int(path.get("duration", 0))

    # 提取 polyline 数据（供前端绘制，不进入 LLM 上下文）
    polylines: List[str] = []
    # 构建不含 polyline 的步骤摘要（供 LLM 推理）
    steps_summary: List[Dict[str, Any]] = []

    for step in path.get("steps", []):
        if step.get("polyline"):
            polylines.append(step["polyline"])
        steps_summary.append({
            "instruction": step.get("instruction", ""),
            "road_name": step.get("road_name", ""),
            "distance": _safe_int(step.get("distance", 0)),
            "duration": _safe_int(step.get("duration", 0)),
        })

    return {
        "summary": {
            "distance_meters": distance,
            "duration_seconds": duration,
            "distance_km": round(distance / 1000, 2),
            "duration_minutes": round(duration / 60, 1),
            "steps_count": len(steps_summary),
        },
        "steps": steps_summary,
        "__artifact__": {
            "type": "route_polyline",
            "polylines": polylines,
            "origin": origin,
            "destination": destination,
        },
    }


GAODE_DRIVING_ROUTE_SPEC = ToolSpec(
    name="gaode_driving_route",
    description="高德地图驾车路线规划工具，根据起点和终点经纬度获取最优驾车路线。",
    parameters={
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "起点经纬度，格式为'经度,纬度'，超出6位小数会自动四舍五入。",
                "pattern": r"^-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?$",
                "examples": ["116.321384,39.904317"],
            },
            "destination": {
                "type": "string",
                "description": "终点经纬度，格式为'经度,纬度'，超出6位小数会自动四舍五入。",
                "pattern": r"^-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?$",
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

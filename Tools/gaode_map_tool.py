from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import aiohttp
import os
import re
import logging
from dotenv import load_dotenv

from Tools.baseTool import BaseTool

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GaodeDrivingTool(BaseTool):
    """高德驾车路线规划工具类（继承BaseTool）"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化高德驾车路线规划工具

        :param api_key: 高德地图API密钥，若不提供则从环境变量GAODE_API_KEY获取
        """
        # 初始化BaseTool属性
        self.name: str = "gaode_driving_route"
        self.description: str = "高德地图驾车路线规划工具，根据起点和终点经纬度获取最优驾车路线"
        self.isSuccess: bool = False
        self.message: str = ""
        self.error: Optional[str] = None

        # 参数描述（符合OpenAPI规范格式）
        self.parameters: Dict[str, Any] = {
            "origin": {
                "type": "string",
                "description": "起点经纬度，格式为'经度,纬度'，经纬度小数点后不得超过6位",
                "pattern": r"^-?\d+\.\d{1,6},-?\d+\.\d{1,6}$",
                "examples": ["116.321384,39.904317"]
            },
            "destination": {
                "type": "string",
                "description": "目的地经纬度，格式为'经度,纬度'，经纬度小数点后不得超过6位",
                "pattern": r"^-?\d+\.\d{1,6},-?\d+\.\d{1,6}$",
                "examples": ["116.587927,40.080102"]
            },
            "required": ["origin", "destination"],
            "additionalProperties": False
        }

        # 初始化高德API客户端属性
        self.api_key = api_key or os.getenv("GAODE_API_KEY")
        self.base_url = "https://restapi.amap.com/v5/direction/driving"
        self.get_max_length = 2048  # GET请求URL最大长度阈值

        # 经纬度验证正则表达式
        self.coord_pattern = re.compile(r'^-?\d+\.\d{1,6},-?\d+\.\d{1,6}$')

        # 验证API Key
        if not self.api_key:
            raise ValueError("高德API Key未提供，请通过参数或环境变量GAODE_API_KEY设置")

    def _validate_coordinates(self, coord: str) -> bool:
        """验证经纬度格式是否正确"""
        return bool(self.coord_pattern.match(coord))

    def _should_use_post(self, params: Dict[str, str]) -> bool:
        """判断是否应该使用POST请求"""
        get_url = f"{self.base_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        return len(get_url) > self.get_max_length

    async def get_driving_route(self, origin: str, destination: str) -> Dict:
        """获取驾车路线规划（核心业务逻辑）"""
        # 验证经纬度格式
        if not self._validate_coordinates(origin):
            raise ValueError(f"起点经纬度格式错误: {origin}，正确格式为'经度,纬度'（小数点后不超过6位）")

        if not self._validate_coordinates(destination):
            raise ValueError(f"目的地经纬度格式错误: {destination}，正确格式为'经度,纬度'（小数点后不超过6位）")

        # 构建请求参数
        params = {
            "key": self.api_key,
            "origin": origin,
            "destination": destination,
            "show_fields": "polyline"  # 返回分路段坐标点串
        }

        try:
            # 使用aiohttp进行异步请求
            async with aiohttp.ClientSession() as session:
                if self._should_use_post(params):
                    logger.info(f"参数过长，使用POST请求获取路线（起点: {origin}, 终点: {destination}）")
                    async with session.post(self.base_url, data=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                        response.raise_for_status()
                        result = await response.json()
                else:
                    logger.info(f"使用GET请求获取路线（起点: {origin}, 终点: {destination}）")
                    async with session.get(self.base_url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                        response.raise_for_status()
                        result = await response.json()

            # 检查API返回状态
            if result.get("status") != "1":
                error_msg = result.get("info", "未知错误")
                error_code = result.get("infocode", "未知错误码")
                raise RuntimeError(f"API调用失败: {error_code} - {error_msg}")

            logger.info("路线规划请求成功")
            return result

        except aiohttp.ClientError as e:
            raise RuntimeError(f"网络请求异常: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"处理响应异常: {str(e)}") from e

    async def __call__(self, origin: str, destination: str) -> Dict[str, Any]:
        """工具调用接口实现（符合BaseTool规范）"""
        try:
            # 调用核心方法
            route_data = await self.get_driving_route(origin, destination)

            # 解析关键结果
            distance = route_data.get("route", {}).get("paths", [{}])[0].get("distance", "0")
            duration = route_data.get("route", {}).get("paths", [{}])[0].get("duration", "0")

            # 设置成功状态
            self.isSuccess = True
            self.message = f"驾车路线规划成功（距离：{distance}米，预计时间：{duration}秒）"

            return {
                "success": self.isSuccess,
                "message": self.message,
                "content": route_data
            }
        except Exception as e:
            # 设置错误状态
            self.isSuccess = False
            self.error = str(e)
            self.message = f"驾车路线规划失败{e}"
            return {
                "success": self.isSuccess,
                "message": self.message,
                "error": self.error
            }


if __name__ == "__main__":
    import asyncio


    async def main():
        try:
            # 创建工具实例
            driving_tool = GaodeDrivingTool()

            # 调用工具（两种方式）
            # 方式1：关键字参数
            result = await driving_tool(
                origin="116.321384,39.904317",
                destination="116.587927,40.080102"
            )

            print(f"调用结果: {'成功' if result['success'] else '失败'}")
            print(f"消息: {result['message']}")
            # if result["success"]:
            #     # 保存完整结果到文件
            #     with open("driving_route_result.json", "w", encoding="utf-8") as f:
            #         import json
            #         json.dump(result["content"], f, ensure_ascii=False, indent=2)
            #     print("完整路线数据已保存至 driving_route_result.json")

        except Exception as e:
            print(f"初始化工具失败: {str(e)}")


    asyncio.run(main())

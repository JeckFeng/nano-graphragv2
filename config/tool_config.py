"""
工具配置管理模块

从 config/tool.yaml 读取工具的超时时间和最大调用次数配置。
支持多层级配置优先级：
1. 装饰器参数（在 tool_context.py 中处理）
2. 工具级别配置（agents.<agent_name>.tools.<tool_name>）
3. Agent 级别默认配置（agents.<agent_name>.defaults）
4. 全局默认配置（global_defaults）
5. 硬编码默认值（timeout=60.0, max_calls=5）

特性：
- 支持通过工具名称自动查找所属 Agent
- 支持热重载配置
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# 硬编码默认值（最后的兜底）
HARDCODED_TIMEOUT = 60.0
HARDCODED_MAX_CALLS = 5

# 配置文件路径
CONFIG_FILE = Path(__file__).parent / "tool.yaml"


class ToolConfig:
    """工具配置管理类"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化工具配置
        
        Args:
            config_path: 配置文件路径，如果为 None 则使用默认路径
        """
        self.config_path = config_path or CONFIG_FILE
        self._config: Dict[str, Any] = {}
        self._tool_to_agent_map: Dict[str, str] = {}  # 工具名 -> Agent 名映射
        self._load_config()
    
    def _load_config(self) -> None:
        """从 YAML 文件加载配置"""
        try:
            if not self.config_path.exists():
                logger.warning(f"工具配置文件不存在: {self.config_path}，使用硬编码默认值")
                self._config = {
                    "global_defaults": {
                        "timeout": HARDCODED_TIMEOUT,
                        "max_calls": HARDCODED_MAX_CALLS,
                    },
                    "agents": {}
                }
                return
            
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            
            # 构建工具名到 Agent 名的映射
            self._build_tool_to_agent_map()
            
            logger.info(f"成功加载工具配置: {self.config_path}")
        
        except Exception as e:
            logger.error(f"加载工具配置失败: {e}，使用硬编码默认值", exc_info=True)
            self._config = {
                "global_defaults": {
                    "timeout": HARDCODED_TIMEOUT,
                    "max_calls": HARDCODED_MAX_CALLS,
                },
                "agents": {}
            }
    
    def _build_tool_to_agent_map(self) -> None:
        """构建工具名到 Agent 名的映射"""
        self._tool_to_agent_map = {}
        agents = self._config.get("agents", {})
        
        for agent_name, agent_config in agents.items():
            tools = agent_config.get("tools", {})
            for tool_name in tools.keys():
                self._tool_to_agent_map[tool_name] = agent_name
    
    def _find_agent_for_tool(self, tool_name: str) -> Optional[str]:
        """
        根据工具名查找所属的 Agent
        
        Args:
            tool_name: 工具名称
        
        Returns:
            Optional[str]: Agent 名称，如果找不到则返回 None
        """
        return self._tool_to_agent_map.get(tool_name)
    
    def get_tool_timeout(self, tool_name: str, agent_name: Optional[str] = None) -> float:
        """
        获取工具的超时时间
        
        查找优先级：
        1. agents.<agent_name>.tools.<tool_name>.timeout
        2. agents.<agent_name>.defaults.timeout
        3. global_defaults.timeout
        4. 硬编码默认值
        
        如果 agent_name 为 None，会尝试通过工具名自动查找所属 Agent。
        
        Args:
            tool_name: 工具名称
            agent_name: Agent 名称（可选，如果为 None 则自动查找）
        
        Returns:
            float: 超时时间（秒）
        """
        # 如果没有提供 agent_name，尝试自动查找
        if agent_name is None:
            agent_name = self._find_agent_for_tool(tool_name)
        
        # 1. 工具级别配置
        if agent_name:
            tool_config = self._config.get("agents", {}).get(agent_name, {}).get("tools", {}).get(tool_name, {})
            if "timeout" in tool_config:
                return float(tool_config["timeout"])
            
            # 2. Agent 级别默认配置
            agent_defaults = self._config.get("agents", {}).get(agent_name, {}).get("defaults", {})
            if "timeout" in agent_defaults:
                return float(agent_defaults["timeout"])
        
        # 3. 全局默认配置
        global_defaults = self._config.get("global_defaults", {})
        if "timeout" in global_defaults:
            return float(global_defaults["timeout"])
        
        # 4. 硬编码默认值
        return HARDCODED_TIMEOUT
    
    def get_tool_max_calls(self, tool_name: str, agent_name: Optional[str] = None) -> int:
        """
        获取工具的最大调用次数
        
        查找优先级：
        1. agents.<agent_name>.tools.<tool_name>.max_calls
        2. agents.<agent_name>.defaults.max_calls
        3. global_defaults.max_calls
        4. 硬编码默认值
        
        如果 agent_name 为 None，会尝试通过工具名自动查找所属 Agent。
        
        Args:
            tool_name: 工具名称
            agent_name: Agent 名称（可选，如果为 None 则自动查找）
        
        Returns:
            int: 最大调用次数
        """
        # 如果没有提供 agent_name，尝试自动查找
        if agent_name is None:
            agent_name = self._find_agent_for_tool(tool_name)
        
        # 1. 工具级别配置
        if agent_name:
            tool_config = self._config.get("agents", {}).get(agent_name, {}).get("tools", {}).get(tool_name, {})
            if "max_calls" in tool_config:
                return int(tool_config["max_calls"])
            
            # 2. Agent 级别默认配置
            agent_defaults = self._config.get("agents", {}).get(agent_name, {}).get("defaults", {})
            if "max_calls" in agent_defaults:
                return int(agent_defaults["max_calls"])
        
        # 3. 全局默认配置
        global_defaults = self._config.get("global_defaults", {})
        if "max_calls" in global_defaults:
            return int(global_defaults["max_calls"])
        
        # 4. 硬编码默认值
        return HARDCODED_MAX_CALLS
    
    def reload(self) -> None:
        """重新加载配置文件（支持热更新）"""
        logger.info("重新加载工具配置...")
        self._load_config()


# 全局单例
_tool_config_instance: Optional[ToolConfig] = None


def get_tool_config() -> ToolConfig:
    """
    获取全局工具配置实例（单例模式）
    
    Returns:
        ToolConfig: 工具配置实例
    """
    global _tool_config_instance
    if _tool_config_instance is None:
        _tool_config_instance = ToolConfig()
    return _tool_config_instance


def reload_tool_config() -> None:
    """重新加载工具配置（用于热更新）"""
    config = get_tool_config()
    config.reload()


__all__ = [
    "ToolConfig",
    "get_tool_config",
    "reload_tool_config",
]

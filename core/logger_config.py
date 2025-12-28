"""
日志配置管理模块

读取和解析 config/logger.yaml 配置文件，提供配置访问接口。

使用示例:
    from core.logger_config import LoggerConfig
    
    # 获取配置
    config = LoggerConfig.get_logger_config("top_supervisor")
    print(config["level"])  # "INFO"
    
    # 重新加载配置
    LoggerConfig.reload_config()
"""

import logging
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LoggerConfig:
    """
    日志配置管理器
    
    单例模式，负责读取和解析 logger.yaml 配置文件。
    """
    
    _instance: Optional["LoggerConfig"] = None
    _config: Dict[str, Any] = {}
    _config_path: Path = Path(__file__).parent.parent / "config" / "logger.yaml"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """
        加载配置文件
        """
        if not self._config_path.exists():
            logger.warning(f"日志配置文件不存在: {self._config_path}，使用默认配置")
            self._config = self._get_default_config()
            return
        
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"成功加载日志配置: {self._config_path}")
        except Exception as e:
            logger.error(f"加载日志配置失败: {e}，使用默认配置", exc_info=True)
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        """
        return {
            "global": {
                "log_dir": "Logs",
                "default_format": "json",
                "default_level": "INFO",
                "enable_console": True,
                "enable_file": True,
                "file_mode": "append",
                "max_file_size_mb": 100,
                "backup_count": 10,
                "async_write": True,
                "buffer_size": 1000,
            },
            "top_supervisor": {
                "enabled": True,
                "level": "INFO",
                "format": "json",
                "log_dir": "Logs/top_supervisor_logs",
                "file_pattern": "top_supervisor_{date}.log",
                "file_mode": "append",
                "enable_console": True,
                "enable_file": True,
            },
            "team": {
                "enabled": True,
                "level": "INFO",
                "format": "json",
                "log_dir": "Logs/team_logs",
                "file_pattern": "team_{team_name}_{date}.log",
                "file_mode": "append",
                "enable_console": True,
                "enable_file": True,
            },
            "worker": {
                "enabled": True,
                "level": "INFO",
                "format": "json",
                "log_dir": "Logs/worker_logs",
                "file_pattern": "worker_{worker_name}_{date}.log",
                "file_mode": "append",
                "enable_console": True,
                "enable_file": True,
            },
            "tool": {
                "enabled": True,
                "level": "DEBUG",
                "format": "json",
                "log_dir": "Logs/tool_logs",
                "file_pattern": "tool_{tool_name}_{date}.log",
                "file_mode": "append",
                "enable_console": False,
                "enable_file": True,
            },
        }
    
    def reload_config(self) -> None:
        """
        重新加载配置
        
        清除缓存并重新读取配置文件。
        """
        self._load_config()
        # 清除所有缓存
        self.get_logger_config.cache_clear()
        self.get_global_config.cache_clear()
        logger.info("日志配置已重新加载")
    
    @classmethod
    @lru_cache(maxsize=4)
    def get_logger_config(cls, logger_type: str) -> Dict[str, Any]:
        """
        获取指定类型的日志器配置
        
        Args:
            logger_type: 日志器类型 ("top_supervisor" | "team" | "worker" | "tool")
        
        Returns:
            Dict[str, Any]: 日志器配置字典
        
        Raises:
            ValueError: 如果 logger_type 无效
        """
        instance = cls()
        
        if logger_type not in ["top_supervisor", "team", "worker", "tool"]:
            raise ValueError(f"无效的日志器类型: {logger_type}")
        
        # 获取该类型的配置
        logger_config = instance._config.get(logger_type, {})
        
        # 合并全局配置作为默认值
        global_config = instance.get_global_config()
        
        # 合并配置（logger_config 优先级更高）
        merged_config = {
            "enabled": logger_config.get("enabled", True),
            "level": logger_config.get("level", global_config.get("default_level", "INFO")),
            "format": logger_config.get("format", global_config.get("default_format", "json")),
            "log_dir": logger_config.get("log_dir", global_config.get("log_dir", "Logs")),
            "file_pattern": logger_config.get("file_pattern", f"{logger_type}_{{date}}.log"),
            "file_mode": logger_config.get("file_mode", global_config.get("file_mode", "append")),
            "enable_console": logger_config.get("enable_console", global_config.get("enable_console", True)),
            "enable_file": logger_config.get("enable_file", global_config.get("enable_file", True)),
            "fields": logger_config.get("fields", []),
            "max_file_size_mb": logger_config.get("max_file_size_mb", global_config.get("max_file_size_mb", 100)),
            "backup_count": logger_config.get("backup_count", global_config.get("backup_count", 10)),
            "async_write": logger_config.get("async_write", global_config.get("async_write", True)),
            "buffer_size": logger_config.get("buffer_size", global_config.get("buffer_size", 1000)),
        }
        
        return merged_config
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_global_config(cls) -> Dict[str, Any]:
        """
        获取全局配置
        
        Returns:
            Dict[str, Any]: 全局配置字典
        """
        instance = cls()
        return instance._config.get("global", {})


def get_logger_config(logger_type: str) -> Dict[str, Any]:
    """
    便捷函数：获取日志器配置
    
    Args:
        logger_type: 日志器类型
    
    Returns:
        Dict[str, Any]: 日志器配置字典
    """
    return LoggerConfig.get_logger_config(logger_type)


def get_global_config() -> Dict[str, Any]:
    """
    便捷函数：获取全局配置
    
    Returns:
        Dict[str, Any]: 全局配置字典
    """
    return LoggerConfig.get_global_config()


__all__ = [
    "LoggerConfig",
    "get_logger_config",
    "get_global_config",
]


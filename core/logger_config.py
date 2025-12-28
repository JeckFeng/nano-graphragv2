"""
日志配置管理模块

读取和解析 config/logger.yaml 配置文件，仅支持 tool 日志配置。
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_config: Dict[str, Any] = {}
_config_path = Path(__file__).parent.parent / "config" / "logger.yaml"


def _load_config() -> None:
    """加载配置文件"""
    global _config
    
    if not _config_path.exists():
        logger.warning(f"日志配置文件不存在: {_config_path}，使用默认配置")
        _config = {
            "tool": {
                "enabled": True,
                "level": "DEBUG",
                "log_dir": "Logs/tool_logs",
                "file_pattern": "tool_{tool_name}_{date}.log",
            }
        }
        return
    
    try:
        with open(_config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f) or {}
        logger.info(f"成功加载日志配置: {_config_path}")
    except Exception as e:
        logger.error(f"加载日志配置失败: {e}，使用默认配置", exc_info=True)
        _config = {
            "tool": {
                "enabled": True,
                "level": "DEBUG",
                "log_dir": "Logs/tool_logs",
                "file_pattern": "tool_{tool_name}_{date}.log",
            }
        }


def get_logger_config(logger_type: str) -> Dict[str, Any]:
    """
    获取日志器配置
    
    Args:
        logger_type: 日志器类型（目前仅支持 "tool"）
    
    Returns:
        Dict[str, Any]: 日志器配置字典
    """
    if not _config:
        _load_config()
    
    return _config.get(logger_type, {})


__all__ = ["get_logger_config"]

"""JSONL logging utilities for observability."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.observability.config import get_obs_config
from app.observability.context import get_ctx


_LOGGERS: Dict[str, logging.Logger] = {}


def _log_file_path(name: str) -> Path:
    today = datetime.utcnow().strftime("%Y%m%d")
    base_dir = Path("Logs") / name
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{name}_{today}.jsonl"


def _get_logger(name: str) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]

    config = get_obs_config()
    if name == "ws":
        enabled = config.enabled_ws
    elif name == "agent":
        enabled = config.enabled_agent
    else:
        enabled = config.enabled_api
    level = logging.INFO if enabled else logging.WARNING
    logger = logging.getLogger(f"obs.{name}")
    logger.setLevel(level)
    logger.propagate = False

    handler = logging.FileHandler(_log_file_path(name), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    _LOGGERS[name] = logger
    return logger


def log_event(logger_name: str, event: Dict[str, Any]) -> None:
    """Write a JSONL log event with context merged.

    Args:
        logger_name: Logger category (app/ws/agent).
        event: Event payload.
    """
    merged = {**get_ctx(), **event}
    merged.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    logger = _get_logger(logger_name)
    logger.info(json.dumps(merged, ensure_ascii=False))

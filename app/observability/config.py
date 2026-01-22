"""Observability configuration utilities.

This module provides runtime configuration for the logging system without
introducing new dependencies. Configuration is sourced from environment
variables with safe defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


def _env_bool(key: str, default: bool) -> bool:
    """Parse a boolean environment variable.

    Args:
        key: Environment variable name.
        default: Fallback value if not set.

    Returns:
        bool: Parsed boolean value.
    """
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(key: str, default: float) -> float:
    """Parse a float environment variable."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    """Parse an integer environment variable."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(key: str, default: List[str]) -> List[str]:
    """Parse a comma-separated list from environment."""
    raw = os.getenv(key)
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or default


@dataclass(frozen=True)
class ObservabilityConfig:
    """Runtime config for observability.

    Invariants:
        - body_max_bytes is positive.
        - sample_rate_fullbody is in [0.0, 1.0].
    """

    enabled_api: bool
    enabled_ws: bool
    enabled_agent: bool
    enabled_tool: bool
    enabled_llm: bool
    capture_req_body: bool
    capture_resp_body: bool
    body_max_bytes: int
    sample_rate_fullbody: float
    redact_keys: List[str]
    debug_thread_ids: List[str]
    debug_user_ids: List[str]


_CONFIG: ObservabilityConfig | None = None


def get_obs_config() -> ObservabilityConfig:
    """Return the singleton observability config."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    sample_rate = _env_float("OBS_SAMPLE_RATE_FULLBODY", 0.0)
    if sample_rate < 0.0:
        sample_rate = 0.0
    if sample_rate > 1.0:
        sample_rate = 1.0

    _CONFIG = ObservabilityConfig(
        enabled_api=_env_bool("OBS_ENABLED_API", True),
        enabled_ws=_env_bool("OBS_ENABLED_WS", True),
        enabled_agent=_env_bool("OBS_ENABLED_AGENT", True),
        enabled_tool=_env_bool("OBS_ENABLED_TOOL", True),
        enabled_llm=_env_bool("OBS_ENABLED_LLM", True),
        capture_req_body=_env_bool("OBS_CAPTURE_REQ_BODY", False),
        capture_resp_body=_env_bool("OBS_CAPTURE_RESP_BODY", False),
        body_max_bytes=_env_int("OBS_BODY_MAX_BYTES", 65536),
        sample_rate_fullbody=sample_rate,
        redact_keys=_env_list(
            "OBS_REDACT_KEYS",
            ["authorization", "cookie", "token", "password", "api_key"],
        ),
        debug_thread_ids=_env_list("OBS_DEBUG_THREAD_IDS", []),
        debug_user_ids=_env_list("OBS_DEBUG_USER_IDS", []),
    )
    return _CONFIG

"""Redaction utilities for observability payloads."""

from __future__ import annotations

from typing import Any, Iterable


def redact_value(value: Any, redact_keys: Iterable[str]) -> Any:
    """Redact sensitive fields in nested structures.

    Args:
        value: Input data (dict/list/scalar).
        redact_keys: Iterable of sensitive key names (case-insensitive).

    Returns:
        Any: Redacted data.
    """
    key_set = {key.lower() for key in redact_keys}
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key.lower() in key_set:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_value(item, redact_keys)
        return redacted
    if isinstance(value, list):
        return [redact_value(item, redact_keys) for item in value]
    return value


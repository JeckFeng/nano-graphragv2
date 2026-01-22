"""Summary helpers for request/response payloads."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Optional

from app.observability.redact import redact_value


def _top_keys(value: Any) -> Optional[list[str]]:
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, list):
        return ["<array>"]
    return None


def summarize_json_bytes(raw: bytes, redact_keys: Iterable[str]) -> Dict[str, Any]:
    """Summarize raw JSON bytes.

    Args:
        raw: Raw bytes (possibly truncated).
        redact_keys: Keys to redact in parsed JSON.

    Returns:
        Dict[str, Any]: Summary with bytes, keys, hash, json_ok.
    """
    summary: Dict[str, Any] = {"bytes": len(raw)}
    if not raw:
        summary.update({"keys": None, "hash": None, "json_ok": False})
        return summary

    summary["hash"] = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        summary.update({"keys": None, "json_ok": False})
        return summary

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        summary.update({"keys": None, "json_ok": False})
        return summary

    redacted = redact_value(data, redact_keys)
    summary["keys"] = _top_keys(redacted)
    summary["json_ok"] = True
    return summary


def summarize_payload(payload: Any) -> Dict[str, Any]:
    """Summarize a Python payload for WS logging."""
    try:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except TypeError:
        encoded = repr(payload).encode("utf-8")
    keys = _top_keys(payload)
    return {"bytes": len(encoded), "keys": keys}


def redact_json_bytes(raw: bytes, redact_keys: Iterable[str], max_bytes: int) -> Optional[str]:
    """Return redacted JSON text for payload capture.

    Args:
        raw: Raw payload bytes (possibly truncated).
        redact_keys: Keys to redact.
        max_bytes: Maximum bytes to keep for output.

    Returns:
        Optional[str]: Redacted JSON text or None if not JSON.
    """
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
        data = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    redacted = redact_value(data, redact_keys)
    output = json.dumps(redacted, ensure_ascii=False, separators=(",", ":"))
    encoded = output.encode("utf-8")
    if len(encoded) > max_bytes:
        output = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return output

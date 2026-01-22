"""LangChain callback handler for LLM usage logging."""

from __future__ import annotations

import time
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler

from app.observability.logger import log_event


class ObservabilityLLMCallback(BaseCallbackHandler):
    """Capture LLM usage and latency for observability."""

    def __init__(self) -> None:
        self._start_times: Dict[str, float] = {}

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        run_id = kwargs.get("run_id")
        if run_id:
            self._start_times[run_id] = time.perf_counter()
        return None

    def on_llm_end(self, response, **kwargs: Any) -> Any:
        run_id = kwargs.get("run_id")
        latency_ms = None
        if run_id and run_id in self._start_times:
            latency_ms = int((time.perf_counter() - self._start_times.pop(run_id)) * 1000)

        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        model_name = llm_output.get("model_name") or llm_output.get("model")

        log_event(
            "agent",
            {
                "source": "backend",
                "component": "llm",
                "event_type": "exchange",
                "event_name": "llm_usage",
                "model": model_name,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "llm_latency_ms": latency_ms,
            },
        )
        return None


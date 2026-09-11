from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from langchain_core.callbacks import BaseCallbackHandler

from logging_config import get_logger

logger = get_logger("observability")

_MAX_FIELD_CHARS = 500

# Defense in depth: only these field names are ever allowed through log_event(), no
# matter what a future change to this module or a future caller passes in. A new field
# must be deliberately added here before it can appear in a log line -- this is a
# fail-closed allowlist, not a denylist, specifically so a future accidental
# `log_event(..., prompt=...)` / `api_key=...` / `authorization=...` / raw document
# content gets silently dropped instead of logged.
_ALLOWED_LOG_FIELDS = frozenset(
    {
        "status",
        "duration_ms",
        "error_type",
        "node",
        "run_id",
        "parent_run_id",
        "model",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "session_id",
        "user_id",
    }
)


def _observability_enabled() -> bool:
    return os.getenv("OBSERVABILITY_ENABLED", "true").strip().lower() not in ("false", "0", "no")


def _sanitize_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Allowlist-filter and length-cap fields before they reach the logger. The
    allowlist is the primary control; length-capping is a secondary backstop for the
    allowed fields themselves."""
    sanitized: Dict[str, Any] = {}
    for key, value in fields.items():
        if key not in _ALLOWED_LOG_FIELDS or value is None:
            continue
        if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
            value = value[:_MAX_FIELD_CHARS] + "...(truncated)"
        sanitized[key] = value
    return sanitized


def log_event(event: str, **fields: Any) -> None:
    """Emit one structured observability log line via the existing JSON logger. Never raises."""
    if not _observability_enabled():
        return
    try:
        logger.info(event, extra=_sanitize_fields(fields))
    except Exception:
        pass


F = TypeVar("F", bound=Callable[..., Any])


def observe_duration(event_type: str) -> Callable[[F], F]:
    """Decorator: measure wall-clock duration around a function call and log it as a
    structured event. Never changes the wrapped function's return value, and always
    re-raises exceptions unchanged after logging -- purely observational."""

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _observability_enabled():
                return fn(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                log_event(event_type, status="error", duration_ms=duration_ms, error_type=type(exc).__name__)
                raise
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log_event(event_type, status="success", duration_ms=duration_ms)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _hash_identifier(value: Optional[str]) -> Optional[str]:
    """Hash an identifier before it's ever logged, so raw session/user ids never reach
    a log line. Opportunistically uses a keyed HMAC when OBSERVABILITY_HASH_SECRET
    happens to be set in the process environment (entirely optional -- not part of any
    config file, not required for this feature to work); otherwise falls back to plain
    SHA-256. Either way, equal inputs still produce equal outputs, so events from the
    same session/user remain correlate-able. Blank values are omitted, not hashed."""
    if not value:
        return None
    try:
        secret = os.getenv("OBSERVABILITY_HASH_SECRET")
        if secret:
            digest = hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()
        else:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return digest[:16]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM cost estimation
#
# Best-effort cost ESTIMATE only -- not an authoritative billing figure.
# Source: published Groq on-demand per-token pricing, https://groq.com/pricing
# Last verified against that page: 2026-07-15.
# Re-check and update this table if Groq's pricing changes; this module makes no
# live pricing API call (that would be a new external integration).
# Values are USD per 1,000,000 tokens.
# ---------------------------------------------------------------------------
_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input_per_million": 0.59, "output_per_million": 0.79},
}


def estimate_cost(
    model: Optional[str], input_tokens: Optional[int], output_tokens: Optional[int]
) -> Optional[float]:
    """Best-effort estimated USD cost for one LLM call. Returns None (never a guessed
    number) when the model isn't in _MODEL_PRICING or token counts are unavailable --
    callers must omit the cost field entirely in that case, never log 0 or a default."""
    if not model or input_tokens is None or output_tokens is None:
        return None
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        return None
    try:
        cost = (input_tokens / 1_000_000) * pricing["input_per_million"] + (
            output_tokens / 1_000_000
        ) * pricing["output_per_million"]
        return round(cost, 8)
    except Exception:
        return None


def _extract_usage(response: Any) -> Optional[Dict[str, Optional[int]]]:
    """Defensively extract token usage from an LLMResult. Returns None if unavailable."""
    try:
        generations = getattr(response, "generations", None)
        if generations:
            message = getattr(generations[0][0], "message", None)
            usage = getattr(message, "usage_metadata", None)
            if usage:
                return {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
    except Exception:
        pass
    try:
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage")
        if token_usage:
            return {
                "input_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens"),
                "output_tokens": token_usage.get("completion_tokens") or token_usage.get("output_tokens"),
                "total_tokens": token_usage.get("total_tokens"),
            }
    except Exception:
        pass
    return None


def _extract_model_name(serialized: Optional[dict], metadata: Optional[dict]) -> Optional[str]:
    try:
        meta = metadata or {}
        for key in ("ls_model_name", "model_name", "model"):
            if meta.get(key):
                return str(meta[key])
        if serialized:
            kwargs = serialized.get("kwargs") or {}
            for key in ("model", "model_name"):
                if kwargs.get(key):
                    return str(kwargs[key])
    except Exception:
        pass
    return None


class ObservabilityCallbackHandler(BaseCallbackHandler):
    """Structured logging of LangGraph node execution and LLM calls via the
    langchain_core callback mechanism -- no changes to node logic, prompts, or graph
    topology required. Every method is defensive: a bug here must never break a run."""

    def __init__(self) -> None:
        super().__init__()
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _begin(self, run_id: Any, **info: Any) -> None:
        with self._lock:
            self._runs[str(run_id)] = {"start": time.perf_counter(), **info}

    def _end(self, run_id: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._runs.pop(str(run_id), None)
        if entry is None:
            return None
        duration_ms = round((time.perf_counter() - entry["start"]) * 1000, 2)
        result = {k: v for k, v in entry.items() if k != "start"}
        result["duration_ms"] = duration_ms
        return result

    # --- Node-level (LangGraph chain) events ---
    # Only genuine LangGraph node boundaries are logged (metadata["langgraph_node"]
    # present) -- internal LangChain composition (e.g. structured-output wrapper
    # chains) is deliberately not logged as a "node" to avoid noise.

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        try:
            meta = metadata or {}
            node = meta.get("langgraph_node")
            if not node:
                return
            self._begin(run_id, node=node)
            log_event(
                "node_start",
                run_id=str(run_id),
                node=node,
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                session_id=_hash_identifier(meta.get("session_id")),
                user_id=_hash_identifier(meta.get("user_id")),
            )
        except Exception:
            pass

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        try:
            info = self._end(run_id)
            if info is None:
                return
            log_event(
                "node_end", run_id=str(run_id), status="success", node=info.get("node"),
                duration_ms=info.get("duration_ms"),
            )
        except Exception:
            pass

    def on_chain_error(self, error, *, run_id, **kwargs):
        try:
            info = self._end(run_id)
            if info is None:
                return
            log_event(
                "node_error", run_id=str(run_id), status="error", node=info.get("node"),
                duration_ms=info.get("duration_ms"), error_type=type(error).__name__,
            )
        except Exception:
            pass

    # --- LLM-level events ---

    def _llm_start(self, serialized, *, run_id, metadata=None, **kwargs):
        meta = metadata or {}
        model = _extract_model_name(serialized, meta)
        session_id = _hash_identifier(meta.get("session_id"))
        user_id = _hash_identifier(meta.get("user_id"))
        self._begin(run_id, model=model, session_id=session_id, user_id=user_id)
        log_event("llm_start", run_id=str(run_id), model=model, session_id=session_id, user_id=user_id)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        try:
            self._llm_start(serialized, run_id=run_id, metadata=metadata)
        except Exception:
            pass

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        # Chat models (e.g. ChatGroq) fire this instead of on_llm_start.
        try:
            self._llm_start(serialized, run_id=run_id, metadata=metadata)
        except Exception:
            pass

    def on_llm_end(self, response, *, run_id, **kwargs):
        try:
            info = self._end(run_id)
            if info is None:
                return
            model = info.get("model")
            fields: Dict[str, Any] = {
                "run_id": str(run_id),
                "status": "success",
                "duration_ms": info.get("duration_ms"),
                "model": model,
                "session_id": info.get("session_id"),
                "user_id": info.get("user_id"),
            }
            usage = _extract_usage(response)
            if usage:
                fields.update(usage)
                cost = estimate_cost(model, usage.get("input_tokens"), usage.get("output_tokens"))
                if cost is not None:
                    fields["estimated_cost_usd"] = cost
            log_event("llm_end", **fields)
        except Exception:
            pass

    def on_llm_error(self, error, *, run_id, **kwargs):
        try:
            info = self._end(run_id)
            if info is None:
                return
            log_event(
                "llm_error", run_id=str(run_id), status="error", model=info.get("model"),
                duration_ms=info.get("duration_ms"), error_type=type(error).__name__,
            )
        except Exception:
            pass


_handler: Optional[ObservabilityCallbackHandler] = None
_handler_lock = threading.Lock()


def get_observability_handler() -> ObservabilityCallbackHandler:
    """Lazy singleton (same pattern as get_vector_store()/get_web_search_provider())."""
    global _handler
    if _handler is None:
        with _handler_lock:
            if _handler is None:
                _handler = ObservabilityCallbackHandler()
    return _handler

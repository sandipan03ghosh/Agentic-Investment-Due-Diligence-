from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, TypedDict

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

# Static, sanitized messages surfaced to the UI on a streaming failure. These never
# include the underlying exception, a traceback, or the run's inputs — those are
# logged server-side only (see the `logger.warning(..., exc_info=True)` calls below).
MSG_STREAMING_DEGRADED = "Live progress streaming is unavailable; continuing with standard generation."
MSG_STREAMING_FAILED = "Streaming is unavailable; generating the report without live progress."


class StreamEvent(TypedDict, total=False):
    """Normalized event shape yielded by run_graph_streaming(). Consumers should
    branch on `kind` and treat the other fields as optional depending on it:

    - "node_end":    a node finished. `node` is set.
    - "token":       an LLM content delta. `node` (may be None) and `text` are set.
    - "tool_start"/"tool_end": a sub-node step (web search, KB search, image
      generation, ...) started/finished. `node` (tool name) and `data` are set.
    - "status":      a minor, non-actionable progress note. `text` and/or `data`.
    - "final":        the run finished. `state` holds the completed graph state.
    - "error":        every streaming tier failed; `text` holds a sanitized message.
    """

    kind: str
    node: Optional[str]
    text: Optional[str]
    data: Optional[Dict[str, Any]]
    state: Optional[Dict[str, Any]]


def _tool_event_kind(payload: Dict[str, Any]) -> str:
    phase = payload.get("phase")
    if phase == "start":
        return "tool_start"
    if phase == "end":
        return "tool_end"
    return "status"


def _stream_full(graph_app: Any, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Iterator[StreamEvent]:
    """Primary path: one graph execution, multi-mode streaming.

    - "values" gives the fully-reduced state snapshot after every super-step (LangGraph
      applies each field's own reducer, e.g. the `sections` list's operator.add across
      parallel `worker` branches) — the last one is used as the final state. This avoids
      re-implementing reducer semantics by hand and avoids a second graph_app.invoke().
    - "updates" gives the node name for each step, used only to emit a "node_end"
      progress event (LangGraph's "updates" mode fires once a node *finishes*, so there
      is no separate "node_start" signal available here).
    - "messages" gives LLM content deltas, tagged with the originating node.
    - "custom" carries the tool_start/tool_end events emitted by backend.py.
    """
    final_state: Dict[str, Any] = dict(inputs)
    seen_node: Optional[str] = None

    for mode, payload in graph_app.stream(
        inputs, stream_mode=["updates", "values", "messages", "custom"], config=config
    ):
        if mode == "values":
            if isinstance(payload, dict):
                final_state = payload
                yield StreamEvent(kind="status", state=final_state)

        elif mode == "updates":
            if isinstance(payload, dict):
                for node_name in payload:
                    if node_name != seen_node:
                        yield StreamEvent(kind="node_end", node=node_name)
                        seen_node = node_name

        elif mode == "messages":
            message_chunk, metadata = payload
            text = getattr(message_chunk, "content", None)
            if isinstance(text, str) and text:
                node_name = (metadata or {}).get("langgraph_node")
                yield StreamEvent(kind="token", node=node_name, text=text)

        elif mode == "custom":
            if isinstance(payload, dict):
                yield StreamEvent(kind=_tool_event_kind(payload), node=payload.get("tool"), data=payload)

    yield StreamEvent(kind="final", state=final_state)


def _stream_no_tokens(
    graph_app: Any, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None
) -> Iterator[StreamEvent]:
    """Default path: node progress + tool events + final state, but NO per-token
    deltas. Token streaming ("messages" mode) is the flakiest, most provider-dependent
    part and the expensive thing to re-run on failure, so it's dropped here unless
    STREAM_TOKENS is set (which prepends _stream_full). Single graph execution; final
    state comes from "values" rather than a second invoke()."""
    final_state: Dict[str, Any] = dict(inputs)
    seen_node: Optional[str] = None

    for mode, payload in graph_app.stream(
        inputs, stream_mode=["updates", "values", "custom"], config=config
    ):
        if mode == "values" and isinstance(payload, dict):
            final_state = payload
            yield StreamEvent(kind="status", state=final_state)
        elif mode == "updates" and isinstance(payload, dict):
            for node_name in payload:
                if node_name != seen_node:
                    yield StreamEvent(kind="node_end", node=node_name)
                    seen_node = node_name
        elif mode == "custom" and isinstance(payload, dict):
            yield StreamEvent(kind=_tool_event_kind(payload), node=payload.get("tool"), data=payload)

    yield StreamEvent(kind="final", state=final_state)


def _invoke_only(
    graph_app: Any, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None
) -> Iterator[StreamEvent]:
    """Last resort: a single plain invoke(), no streaming at all."""
    out = graph_app.invoke(inputs, config=config)
    yield StreamEvent(kind="final", state=out)


def _looks_like_provider_error(exc: BaseException) -> bool:
    """True if the failure is the LLM provider rejecting the call (rate limit, quota,
    auth, model-not-found) rather than a streaming-mode incompatibility. Retrying with
    a simpler streaming mode re-runs the whole graph and will just hit the same wall,
    burning more quota -- so on these we stop and surface the error immediately."""
    text = f"{type(exc).__name__} {exc}".lower()
    needles = (
        "rate limit", "ratelimit", "resource_exhausted", "quota", "429",
        "insufficient_quota", "not_found", "permission", "unauthenticated",
        "api key", "apikey", "401", "403",
    )
    return any(n in text for n in needles)


# Default: one streaming attempt (no token deltas) + one plain invoke() fallback --
# 2 graph executions worst case, not 4. STREAM_TOKENS prepends the richer token-
# streaming path for anyone whose provider/quota can afford a retry of it.
_FALLBACK_TIERS = (_stream_full, _stream_no_tokens) if settings.STREAM_TOKENS else (_stream_no_tokens,)


def run_graph_streaming(
    graph_app: Any, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None
) -> Iterator[StreamEvent]:
    """Drive `graph_app` over `inputs`, yielding normalized StreamEvents.

    Tries progressively simpler streaming modes, then a plain invoke(), so an
    incompatible environment (older langgraph, a node that can't be streamed, etc.)
    degrades gracefully instead of failing the run. Each tier executes the graph
    exactly once. If a tier fails partway through (after already yielding some
    events) the next tier re-runs the graph from scratch — the same trade-off the
    prior nested-try implementation in frontend.py made; only genuine mid-stream
    failures hit it, not the common case.

    `config` is passed straight through to every graph_app.stream()/.invoke() call
    (default None, so omitting it changes nothing). This is how callers attach
    per-run callbacks/metadata -- e.g. the observability callback handler and
    session/user metadata -- without this module needing to know anything about them.
    """
    degraded = False
    for attempt in _FALLBACK_TIERS:
        try:
            if degraded:
                yield StreamEvent(kind="status", text=MSG_STREAMING_DEGRADED)
            yield from attempt(graph_app, inputs, config)
            return
        except Exception as exc:
            logger.warning("graph_streaming_tier_failed", exc_info=True, extra={"tier": attempt.__name__})
            if _looks_like_provider_error(exc):
                # The LLM call itself failed -- more tiers would just re-run the graph
                # and hit the same wall. Stop here.
                yield StreamEvent(kind="error", text=MSG_STREAMING_FAILED)
                return
            degraded = True

    try:
        yield from _invoke_only(graph_app, inputs, config)
    except Exception:
        logger.warning("graph_invoke_failed", exc_info=True)
        yield StreamEvent(kind="error", text=MSG_STREAMING_FAILED)

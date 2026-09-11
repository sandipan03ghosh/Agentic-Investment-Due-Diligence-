from __future__ import annotations

import logging
import uuid as uuid_module

import pytest

import observability


# --- log_event / _sanitize_fields ---


def test_log_event_only_allows_listed_fields(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    observability.log_event(
        "custom_event",
        status="success",
        duration_ms=12.5,
        prompt="this should never be logged",
        api_key="sk-should-never-appear",
    )

    record = next(r for r in caplog.records if r.message == "custom_event")
    assert record.status == "success"
    assert record.duration_ms == 12.5
    assert not hasattr(record, "prompt")
    assert not hasattr(record, "api_key")


def test_log_event_truncates_long_string_fields(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    observability.log_event("custom_event", node="x" * 1000)

    record = next(r for r in caplog.records if r.message == "custom_event")
    assert len(record.node) <= len("x" * observability._MAX_FIELD_CHARS) + len("...(truncated)")


def test_log_event_omits_none_fields(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    observability.log_event("custom_event", status="success", duration_ms=None)

    record = next(r for r in caplog.records if r.message == "custom_event")
    assert not hasattr(record, "duration_ms")


def test_log_event_disabled_emits_nothing(monkeypatch, caplog):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
    caplog.set_level(logging.INFO, logger="observability")
    observability.log_event("should_not_appear", status="success")
    assert not any(r.message == "should_not_appear" for r in caplog.records)


def test_log_event_never_raises_on_bad_input():
    class Unloggable:
        def __str__(self):
            raise RuntimeError("cannot stringify")

    observability.log_event("custom_event", status=Unloggable())  # must not raise


# --- observe_duration ---


def test_observe_duration_logs_success_and_preserves_return_value(caplog):
    caplog.set_level(logging.INFO, logger="observability")

    @observability.observe_duration("test_add")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5

    record = next(r for r in caplog.records if r.message == "test_add")
    assert record.status == "success"
    assert record.duration_ms >= 0


def test_observe_duration_logs_error_and_reraises_unchanged(caplog):
    caplog.set_level(logging.INFO, logger="observability")

    @observability.observe_duration("test_boom")
    def boom():
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        boom()

    record = next(r for r in caplog.records if r.message == "test_boom")
    assert record.status == "error"
    assert record.error_type == "ValueError"


def test_observe_duration_disabled_short_circuits(monkeypatch, caplog):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
    caplog.set_level(logging.INFO, logger="observability")

    @observability.observe_duration("test_noop")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert not any(r.message == "test_noop" for r in caplog.records)


# --- _hash_identifier ---


def test_hash_identifier_is_stable(monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_HASH_SECRET", raising=False)
    h1 = observability._hash_identifier("alice@example.com")
    h2 = observability._hash_identifier("alice@example.com")
    assert h1 == h2
    assert h1 is not None
    assert "alice@example.com" not in h1


def test_hash_identifier_returns_none_for_blank_or_none():
    assert observability._hash_identifier("") is None
    assert observability._hash_identifier(None) is None


def test_hash_identifier_uses_hmac_when_secret_is_set(monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_HASH_SECRET", raising=False)
    plain = observability._hash_identifier("alice@example.com")

    monkeypatch.setenv("OBSERVABILITY_HASH_SECRET", "top-secret-key")
    hmac_hashed = observability._hash_identifier("alice@example.com")

    assert plain != hmac_hashed
    assert hmac_hashed is not None


def test_hash_identifier_works_without_secret_configured(monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_HASH_SECRET", raising=False)
    # Must not raise or require any setup -- the default, out-of-the-box behavior.
    result = observability._hash_identifier("bob@example.com")
    assert result is not None
    assert "bob@example.com" not in result


# --- estimate_cost ---


def test_estimate_cost_known_model():
    cost = observability.estimate_cost("llama-3.3-70b-versatile", 1_000_000, 1_000_000)
    assert cost is not None
    assert cost > 0


def test_estimate_cost_unknown_model_returns_none():
    assert observability.estimate_cost("some-unreleased-model", 100, 100) is None


def test_estimate_cost_missing_tokens_returns_none():
    assert observability.estimate_cost("llama-3.3-70b-versatile", None, 100) is None
    assert observability.estimate_cost("llama-3.3-70b-versatile", 100, None) is None


def test_estimate_cost_missing_model_returns_none():
    assert observability.estimate_cost(None, 100, 100) is None


# --- ObservabilityCallbackHandler: node (chain) events ---


class _FakeMessage:
    def __init__(self, usage_metadata=None):
        self.usage_metadata = usage_metadata


class _FakeGeneration:
    def __init__(self, message):
        self.message = message


class _FakeLLMResult:
    def __init__(self, generations=None, llm_output=None):
        self.generations = generations or []
        self.llm_output = llm_output or {}


def test_on_chain_start_end_logs_node_and_duration(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    handler = observability.ObservabilityCallbackHandler()
    run_id = uuid_module.uuid4()

    handler.on_chain_start({}, {}, run_id=run_id, metadata={"langgraph_node": "router"})
    handler.on_chain_end({}, run_id=run_id)

    start = next(r for r in caplog.records if r.message == "node_start")
    end = next(r for r in caplog.records if r.message == "node_end")
    assert start.node == "router"
    assert end.node == "router"
    assert end.status == "success"
    assert end.duration_ms >= 0


def test_on_chain_start_skips_events_without_langgraph_node(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    handler = observability.ObservabilityCallbackHandler()
    run_id = uuid_module.uuid4()

    handler.on_chain_start({}, {}, run_id=run_id, metadata={})
    handler.on_chain_end({}, run_id=run_id)

    assert not any(r.message in ("node_start", "node_end") for r in caplog.records)


def test_on_chain_error_logs_error_type(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    handler = observability.ObservabilityCallbackHandler()
    run_id = uuid_module.uuid4()

    handler.on_chain_start({}, {}, run_id=run_id, metadata={"langgraph_node": "worker"})
    handler.on_chain_error(ValueError("boom"), run_id=run_id)

    error = next(r for r in caplog.records if r.message == "node_error")
    assert error.node == "worker"
    assert error.error_type == "ValueError"


def test_chain_end_never_double_logs_for_same_run_id(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    handler = observability.ObservabilityCallbackHandler()
    run_id = uuid_module.uuid4()

    handler.on_chain_start({}, {}, run_id=run_id, metadata={"langgraph_node": "router"})
    handler.on_chain_end({}, run_id=run_id)
    handler.on_chain_end({}, run_id=run_id)  # second call: run_id no longer tracked

    assert len([r for r in caplog.records if r.message == "node_end"]) == 1


def test_node_start_hashes_session_and_user_id(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    handler = observability.ObservabilityCallbackHandler()
    run_id = uuid_module.uuid4()

    handler.on_chain_start(
        {},
        {},
        run_id=run_id,
        metadata={"langgraph_node": "router", "session_id": "sess-raw-123", "user_id": "alice@example.com"},
    )

    record = next(r for r in caplog.records if r.message == "node_start")
    assert record.session_id != "sess-raw-123"
    assert record.user_id != "alice@example.com"
    assert "sess-raw-123" not in str(record.session_id)
    assert "alice@example.com" not in str(record.user_id)


# --- ObservabilityCallbackHandler: LLM events ---


def test_on_llm_start_end_extracts_usage_and_cost(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    handler = observability.ObservabilityCallbackHandler()
    run_id = uuid_module.uuid4()

    handler.on_chat_model_start({"kwargs": {"model": "llama-3.3-70b-versatile"}}, [], run_id=run_id, metadata={})
    response = _FakeLLMResult(
        generations=[[_FakeGeneration(_FakeMessage(usage_metadata={
            "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
        }))]]
    )
    handler.on_llm_end(response, run_id=run_id)

    record = next(r for r in caplog.records if r.message == "llm_end")
    assert record.model == "llama-3.3-70b-versatile"
    assert record.input_tokens == 100
    assert record.output_tokens == 50
    assert record.total_tokens == 150
    assert record.estimated_cost_usd > 0


def test_on_llm_end_omits_cost_field_for_unknown_model(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    handler = observability.ObservabilityCallbackHandler()
    run_id = uuid_module.uuid4()

    handler.on_chat_model_start({"kwargs": {"model": "some-unknown-model"}}, [], run_id=run_id, metadata={})
    response = _FakeLLMResult(
        generations=[[_FakeGeneration(_FakeMessage(usage_metadata={
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        }))]]
    )
    handler.on_llm_end(response, run_id=run_id)

    record = next(r for r in caplog.records if r.message == "llm_end")
    assert not hasattr(record, "estimated_cost_usd")


def test_on_llm_end_handles_missing_usage_gracefully(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    handler = observability.ObservabilityCallbackHandler()
    run_id = uuid_module.uuid4()

    handler.on_chat_model_start({"kwargs": {"model": "llama-3.3-70b-versatile"}}, [], run_id=run_id, metadata={})
    handler.on_llm_end(_FakeLLMResult(generations=[]), run_id=run_id)

    record = next(r for r in caplog.records if r.message == "llm_end")
    assert not hasattr(record, "input_tokens")
    assert not hasattr(record, "estimated_cost_usd")


def test_on_llm_error_logs_error_type(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    handler = observability.ObservabilityCallbackHandler()
    run_id = uuid_module.uuid4()

    handler.on_chat_model_start({"kwargs": {"model": "llama-3.3-70b-versatile"}}, [], run_id=run_id, metadata={})
    handler.on_llm_error(RuntimeError("api down"), run_id=run_id)

    record = next(r for r in caplog.records if r.message == "llm_error")
    assert record.error_type == "RuntimeError"


def test_llm_end_never_double_logs_for_same_run_id(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    handler = observability.ObservabilityCallbackHandler()
    run_id = uuid_module.uuid4()

    handler.on_chat_model_start({"kwargs": {"model": "llama-3.3-70b-versatile"}}, [], run_id=run_id, metadata={})
    response = _FakeLLMResult(generations=[])
    handler.on_llm_end(response, run_id=run_id)
    handler.on_llm_end(response, run_id=run_id)  # second call: run_id no longer tracked

    assert len([r for r in caplog.records if r.message == "llm_end"]) == 1


def test_callback_handler_never_raises_on_malformed_input():
    handler = observability.ObservabilityCallbackHandler()
    # Missing/garbage metadata, unexpected response shapes -- must never raise.
    handler.on_chain_start(None, None, run_id="not-a-uuid", metadata=None)
    handler.on_chain_end(None, run_id="not-a-uuid")
    handler.on_chain_error(Exception("x"), run_id="unknown-run-id")
    handler.on_llm_start(None, None, run_id="not-a-uuid", metadata=None)
    handler.on_llm_end(object(), run_id="not-a-uuid")
    handler.on_llm_error(Exception("x"), run_id="unknown-run-id")


def test_get_observability_handler_is_a_singleton():
    h1 = observability.get_observability_handler()
    h2 = observability.get_observability_handler()
    assert h1 is h2

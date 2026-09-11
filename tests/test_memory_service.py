from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import memory.service as service_module
from memory.models import Base
from memory.repository import MemoryRepository
from memory.service import MemoryService


class _FakeSummaryResult:
    def __init__(self, summary):
        self.summary = summary


class _FakeFactsResult:
    def __init__(self, facts):
        self.facts = facts


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):
        return self._result


class _FakeLLM:
    def __init__(self, summary_text="Updated summary.", facts=None):
        self._summary_text = summary_text
        self._facts = facts or []

    def with_structured_output(self, schema):
        if schema.__name__ == "SessionSummaryResult":
            return _FakeStructuredLLM(_FakeSummaryResult(self._summary_text))
        return _FakeStructuredLLM(_FakeFactsResult(self._facts))


@pytest.fixture()
def isolated_session(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'test.db').as_posix()}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _get_session():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(service_module, "get_session", _get_session)
    return _get_session


@pytest.fixture()
def no_vector_store(monkeypatch):
    """Force the semantic layer to fail so tests exercise only the deterministic SQL paths."""

    def fail_get_vector_store():
        raise RuntimeError("vector store unavailable in tests")

    monkeypatch.setattr(service_module, "get_vector_store", fail_get_vector_store)
    monkeypatch.setattr(service_module, "embed_texts", lambda texts: [[0.0] for _ in texts])


@pytest.fixture()
def service(no_vector_store) -> MemoryService:
    return MemoryService(
        llm=_FakeLLM(),
        max_message_chars=8000,
        summarization_threshold_chars=50,
        max_facts_in_context=10,
        launch_history_collection="launch_history_test",
    )


def test_record_interaction_creates_session_message_and_launch(service, isolated_session):
    service.record_interaction(
        user_id="alice@example.com",
        session_id="sess-1",
        topic="Launch a phone",
        mode="hybrid",
        blog_kind="launch_brief",
        blog_title="Phone Launch Brief",
        plan_summary="Audience: young adults | Tone: energetic | 6 section(s)",
        final_markdown="# Phone Launch Brief\n...",
    )

    with isolated_session() as db_session:
        repo = MemoryRepository(db_session)
        messages = repo.list_messages("sess-1")
        assert len(messages) == 2
        launches = repo.list_recent_launches("alice@example.com", limit=10)
        assert len(launches) == 1
        assert launches[0].blog_title == "Phone Launch Brief"


def test_record_interaction_with_blank_user_id_stores_anonymous(service, isolated_session):
    service.record_interaction(
        user_id="",
        session_id="sess-1",
        topic="Launch a phone",
        mode="hybrid",
        blog_kind=None,
        blog_title=None,
        plan_summary=None,
        final_markdown="content",
    )

    with isolated_session() as db_session:
        repo = MemoryRepository(db_session)
        session_row = repo.get_session("sess-1")
        assert session_row.user_id == ""


def test_summarization_not_triggered_below_threshold(isolated_session, no_vector_store):
    fake_llm = _FakeLLM(summary_text="SHOULD NOT APPEAR")
    svc = MemoryService(
        llm=fake_llm,
        max_message_chars=8000,
        summarization_threshold_chars=100_000,
        max_facts_in_context=10,
        launch_history_collection="lh_test",
    )
    svc.record_interaction(
        user_id="bob@example.com",
        session_id="sess-2",
        topic="t",
        mode="hybrid",
        blog_kind=None,
        blog_title=None,
        plan_summary=None,
        final_markdown="x",
    )

    with isolated_session() as db_session:
        repo = MemoryRepository(db_session)
        assert repo.get_session_summary("sess-2") is None


def test_summarization_triggered_above_threshold_and_consolidates_facts(isolated_session, no_vector_store):
    fake_llm = _FakeLLM(
        summary_text="User is planning phone launches.",
        facts=["Focuses on consumer electronics."],
    )
    svc = MemoryService(
        llm=fake_llm,
        max_message_chars=8000,
        summarization_threshold_chars=1,
        max_facts_in_context=10,
        launch_history_collection="lh_test",
    )
    svc.record_interaction(
        user_id="carol@example.com",
        session_id="sess-3",
        topic="Launch a phone",
        mode="hybrid",
        blog_kind="launch_brief",
        blog_title="Phone Brief",
        plan_summary="summary",
        final_markdown="content",
    )

    with isolated_session() as db_session:
        repo = MemoryRepository(db_session)
        summary_row = repo.get_session_summary("sess-3")
        assert summary_row is not None
        assert summary_row.summary == "User is planning phone launches."

        facts = repo.list_user_facts("carol@example.com", limit=10)
        assert [f.fact for f in facts] == ["Focuses on consumer electronics."]


def test_fact_consolidation_skips_duplicates(isolated_session, no_vector_store):
    fake_llm = _FakeLLM(summary_text="summary text", facts=["Focuses on consumer electronics."])
    svc = MemoryService(
        llm=fake_llm,
        max_message_chars=8000,
        summarization_threshold_chars=1,
        max_facts_in_context=10,
        launch_history_collection="lh_test",
    )
    with isolated_session() as db_session:
        repo = MemoryRepository(db_session)
        repo.add_user_fact("dave@example.com", "focuses on consumer electronics")

    svc.record_interaction(
        user_id="dave@example.com",
        session_id="sess-4",
        topic="t",
        mode="hybrid",
        blog_kind=None,
        blog_title=None,
        plan_summary=None,
        final_markdown="x",
    )

    with isolated_session() as db_session:
        repo = MemoryRepository(db_session)
        facts = repo.list_user_facts("dave@example.com", limit=10)
        assert len(facts) == 1


def test_get_context_for_run_includes_summary_and_facts(isolated_session, no_vector_store, service):
    with isolated_session() as db_session:
        repo = MemoryRepository(db_session)
        repo.get_or_create_session("sess-5", "erin@example.com")
        repo.upsert_session_summary("sess-5", "Prior summary text.", through_message_id=0)
        repo.add_user_fact("erin@example.com", "Prefers concise briefs.")

    context = service.get_context_for_run("erin@example.com", "sess-5", "new topic")

    assert "Prior summary text." in context
    assert "Prefers concise briefs." in context


def test_get_context_for_run_blank_user_id_returns_only_session_summary(isolated_session, no_vector_store, service):
    with isolated_session() as db_session:
        repo = MemoryRepository(db_session)
        repo.get_or_create_session("sess-6", "")
        repo.upsert_session_summary("sess-6", "Anon summary.", through_message_id=0)

    context = service.get_context_for_run("", "sess-6", "topic")

    assert "Anon summary." in context
    assert "Known long-term facts" not in context


def test_get_context_for_run_returns_empty_string_when_nothing_recorded(isolated_session, no_vector_store, service):
    context = service.get_context_for_run("nobody@example.com", "sess-does-not-exist", "topic")
    assert context == ""

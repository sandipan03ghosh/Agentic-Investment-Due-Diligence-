from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memory.models import Base, Launch
from memory.repository import MemoryRepository


@pytest.fixture()
def repo(tmp_path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'test.db').as_posix()}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    session = session_local()
    yield MemoryRepository(session)
    session.close()


def test_get_or_create_session_creates_once(repo):
    s1 = repo.get_or_create_session("sess-1", "user-1", title="First")
    s2 = repo.get_or_create_session("sess-1", "user-1")

    assert s1.id == s2.id
    assert s2.title == "First"


def test_add_message_and_list_messages_ordered(repo):
    repo.get_or_create_session("sess-1", "user-1")
    repo.add_message("sess-1", "user", "first")
    repo.add_message("sess-1", "assistant", "second")

    messages = repo.list_messages("sess-1")
    assert [m.content for m in messages] == ["first", "second"]


def test_list_messages_respects_since_watermark(repo):
    repo.get_or_create_session("sess-1", "user-1")
    m1 = repo.add_message("sess-1", "user", "first")
    repo.add_message("sess-1", "assistant", "second")

    messages = repo.list_messages("sess-1", since_message_id=m1.id)
    assert [m.content for m in messages] == ["second"]


def test_upsert_session_summary_creates_and_updates(repo):
    repo.get_or_create_session("sess-1", "user-1")
    repo.upsert_session_summary("sess-1", "initial summary", through_message_id=1)
    row = repo.get_session_summary("sess-1")
    assert row.summary == "initial summary"

    repo.upsert_session_summary("sess-1", "updated summary", through_message_id=2)
    row = repo.get_session_summary("sess-1")
    assert row.summary == "updated summary"
    assert row.summarized_through_message_id == 2


def test_fact_exists_detects_normalized_duplicates(repo):
    repo.add_user_fact("user-1", "Prefers hybrid mode.")

    assert repo.fact_exists("user-1", "  prefers   HYBRID mode.  ") is True
    assert repo.fact_exists("user-1", "Something else entirely.") is False


def test_list_user_facts_orders_newest_first(repo):
    repo.add_user_fact("user-1", "fact one")
    repo.add_user_fact("user-1", "fact two")

    facts = repo.list_user_facts("user-1", limit=10)
    assert [f.fact for f in facts] == ["fact two", "fact one"]


def test_record_and_get_launch(repo):
    launch = Launch(
        id="launch-1",
        session_id="sess-1",
        user_id="user-1",
        topic="Launch a phone",
        mode="hybrid",
        blog_kind="launch_brief",
        blog_title="Phone Launch",
        plan_summary="summary",
        final_markdown="# Phone Launch\n...",
    )
    repo.record_launch(launch)

    fetched = repo.get_launch("launch-1")
    assert fetched is not None
    assert fetched.blog_title == "Phone Launch"


def test_list_recent_launches_scoped_to_user(repo):
    repo.record_launch(
        Launch(id="l1", session_id="s1", user_id="user-1", topic="A", mode="hybrid", final_markdown="")
    )
    repo.record_launch(
        Launch(id="l2", session_id="s1", user_id="user-2", topic="B", mode="hybrid", final_markdown="")
    )

    launches = repo.list_recent_launches("user-1", limit=10)
    assert [l.id for l in launches] == ["l1"]


def test_search_launches_matches_topic_or_title(repo):
    repo.record_launch(
        Launch(
            id="l1",
            session_id="s1",
            user_id="user-1",
            topic="Washing machine launch",
            mode="hybrid",
            blog_title="WM Launch Brief",
            final_markdown="",
        )
    )
    repo.record_launch(
        Launch(
            id="l2",
            session_id="s1",
            user_id="user-1",
            topic="Phone launch",
            mode="hybrid",
            blog_title="Phone Brief",
            final_markdown="",
        )
    )

    results = repo.search_launches("user-1", "washing", limit=10)
    assert [l.id for l in results] == ["l1"]

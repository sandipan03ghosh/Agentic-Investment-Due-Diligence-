from __future__ import annotations

import uuid
from typing import Any, List, Optional

from config import settings
from content_security import sanitize_retrieved_text
from logging_config import get_logger
from vector_store import MetadataFilter, VectorRecord, embed_texts, get_vector_store

from . import summarizer
from .db import get_session
from .models import Launch
from .repository import MemoryRepository
from .security import sanitize_user_id, validate_message_content, validate_text_field

logger = get_logger(__name__)


class MemoryService:
    """Conversation memory, session memory, long-term user memory, automatic summarization,
    and retrieval of previous due diligence reports — all persisted to SQLite via MemoryRepository.

    Every method degrades gracefully: memory failures must never block brief generation.
    """

    def __init__(
        self,
        llm: Any,
        max_message_chars: int,
        summarization_threshold_chars: int,
        max_facts_in_context: int,
        launch_history_collection: str,
    ):
        self._llm = llm
        self._max_message_chars = max_message_chars
        self._summarization_threshold_chars = summarization_threshold_chars
        self._max_facts_in_context = max_facts_in_context
        self._launch_collection = launch_history_collection

    # --- Context assembly (read path, called before a run) ---

    def get_context_for_run(self, user_id: Optional[str], session_id: str, topic: str) -> str:
        effective_user_id = sanitize_user_id(user_id)
        try:
            with get_session() as db_session:
                repo = MemoryRepository(db_session)
                parts: List[str] = []

                summary_row = repo.get_session_summary(session_id)
                if summary_row and summary_row.summary:
                    parts.append(f"Session summary so far:\n{summary_row.summary}")

                if effective_user_id:
                    facts = repo.list_user_facts(effective_user_id, limit=self._max_facts_in_context)
                    if facts:
                        parts.append(
                            "Known long-term facts about this user:\n"
                            + "\n".join(f"- {f.fact}" for f in facts)
                        )

                    launches_text = self._describe_relevant_launches(repo, effective_user_id, topic)
                    if launches_text:
                        parts.append(launches_text)

                # Memory content is untrusted data (ultimately traceable back to user
                # input/prior generations) -- sanitize the assembled context once,
                # here, before it's ever placed into a prompt.
                return sanitize_retrieved_text("\n\n".join(parts))
        except Exception:
            logger.warning("get_context_for_run_failed")
            return ""

    def _describe_relevant_launches(self, repo: MemoryRepository, user_id: str, topic: str) -> str:
        descriptions = self._semantic_search_launches(user_id, topic, top_k=3)
        if not descriptions:
            rows = repo.search_launches(user_id, topic, limit=3) or repo.list_recent_launches(user_id, limit=3)
            descriptions = [
                f"{r.blog_title or r.topic} ({r.created_at.date().isoformat()}, mode={r.mode})" for r in rows
            ]
        if not descriptions:
            return ""
        return "Relevant past due diligence reports:\n" + "\n".join(f"- {d}" for d in descriptions)

    def _semantic_search_launches(self, user_id: str, topic: str, top_k: int) -> List[str]:
        try:
            store = get_vector_store()
            vector = embed_texts([topic])[0]
            filters = MetadataFilter(document_id=user_id)
            results = store.search(self._launch_collection, vector, top_k, filters)
            return [
                f"{r.title or '(untitled)'} ({r.published_at or 'date unknown'}, mode={r.source or 'n/a'})"
                for r in results
                if r.file_type == "launch"
            ]
        except Exception:
            logger.warning("semantic_launch_search_failed")
            return []

    # --- Recording (write path, called after a run completes) ---

    def record_interaction(
        self,
        user_id: Optional[str],
        session_id: str,
        topic: str,
        mode: str,
        blog_kind: Optional[str],
        blog_title: Optional[str],
        plan_summary: Optional[str],
        final_markdown: str,
    ) -> None:
        effective_user_id = sanitize_user_id(user_id)
        topic = validate_text_field(topic, max_chars=500)
        blog_title = validate_text_field(blog_title, max_chars=300) or None
        blog_kind = validate_text_field(blog_kind, max_chars=64) or None

        try:
            with get_session() as db_session:
                repo = MemoryRepository(db_session)
                repo.get_or_create_session(session_id, effective_user_id, title=blog_title or topic)

                repo.add_message(
                    session_id, "user", validate_message_content(f"Requested a due diligence report for: {topic}", self._max_message_chars)
                )
                synopsis = f"Generated {blog_kind or 'brief'} '{blog_title or topic}' ({mode} mode)."
                repo.add_message(session_id, "assistant", validate_message_content(synopsis, self._max_message_chars))

                launch = Launch(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    user_id=effective_user_id,
                    topic=topic,
                    mode=mode or "unknown",
                    blog_kind=blog_kind,
                    blog_title=blog_title,
                    plan_summary=validate_message_content(plan_summary or "", self._max_message_chars) or None,
                    final_markdown=final_markdown or "",
                )
                repo.record_launch(launch)

                new_summary = self._maybe_summarize(repo, session_id)
        except Exception:
            logger.warning("record_interaction_failed")
            return

        # Best-effort side effects outside the DB transaction — never block the primary write.
        self._index_launch_semantic(launch)
        if new_summary and effective_user_id:
            self._consolidate_long_term_facts(effective_user_id, session_id, new_summary)

    def _maybe_summarize(self, repo: MemoryRepository, session_id: str) -> Optional[str]:
        summary_row = repo.get_session_summary(session_id)
        watermark = summary_row.summarized_through_message_id if summary_row else 0
        prior_summary = summary_row.summary if summary_row else ""

        new_messages = repo.list_messages(session_id, since_message_id=watermark)
        total_chars = sum(len(m.content) for m in new_messages)
        if total_chars < self._summarization_threshold_chars or not new_messages:
            return None

        new_summary = summarizer.summarize_session(self._llm, prior_summary, new_messages)
        repo.upsert_session_summary(session_id, new_summary, new_messages[-1].id)
        return new_summary

    def _consolidate_long_term_facts(self, user_id: str, session_id: str, summary: str) -> None:
        try:
            facts = summarizer.extract_long_term_facts(self._llm, summary)
            if not facts:
                return
            with get_session() as db_session:
                repo = MemoryRepository(db_session)
                for fact in facts:
                    fact = validate_text_field(fact, max_chars=300)
                    if fact and not repo.fact_exists(user_id, fact):
                        repo.add_user_fact(user_id, fact, category="consolidated", source_session_id=session_id)
        except Exception:
            logger.warning("long_term_fact_consolidation_failed")

    def _index_launch_semantic(self, launch: Launch) -> None:
        if not launch.user_id:
            return
        try:
            store = get_vector_store()
            store.ensure_collection(self._launch_collection)
            text_for_embedding = f"{launch.topic}\n{launch.blog_title or ''}\n{launch.plan_summary or ''}"
            vector = embed_texts([text_for_embedding])[0]
            record = VectorRecord(
                key=launch.id,
                properties={
                    "title": launch.blog_title or launch.topic,
                    "url": f"launch://{launch.id}",
                    "snippet": launch.plan_summary or launch.topic,
                    "published_at": launch.created_at.date().isoformat(),
                    "source": launch.mode,
                    "document_id": launch.user_id,
                    "file_type": "launch",
                },
                vector=vector,
            )
            store.upsert(self._launch_collection, [record])
        except Exception:
            logger.warning("semantic_launch_indexing_failed")

    # --- Simple listing (used by the UI) ---

    def list_recent_launches(self, user_id: Optional[str], session_id: str, limit: int = 10) -> List[Launch]:
        effective_user_id = sanitize_user_id(user_id)
        try:
            with get_session() as db_session:
                repo = MemoryRepository(db_session)
                if effective_user_id:
                    return repo.list_recent_launches(effective_user_id, limit=limit)
                # Anonymous: fall back to this session's own launches only.
                anonymous_rows = [
                    launch
                    for launch in repo.list_recent_launches("", limit=limit * 5)
                    if launch.session_id == session_id
                ]
                return anonymous_rows[:limit]
        except Exception:
            logger.warning("list_recent_launches_failed")
            return []


def build_memory_service(llm: Any) -> MemoryService:
    return MemoryService(
        llm=llm,
        max_message_chars=settings.MEMORY_MAX_MESSAGE_CHARS,
        summarization_threshold_chars=settings.MEMORY_SUMMARIZATION_THRESHOLD_CHARS,
        max_facts_in_context=settings.MEMORY_MAX_USER_FACTS_IN_CONTEXT,
        launch_history_collection=settings.MEMORY_LAUNCH_HISTORY_COLLECTION,
    )

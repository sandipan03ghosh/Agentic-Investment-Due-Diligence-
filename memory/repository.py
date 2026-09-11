from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session as SASession

from .models import ChatSession, Launch, Message, SessionSummary, UserMemory


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


class MemoryRepository:
    """SQLAlchemy-backed persistence for the memory subsystem. ORM only, no raw SQL."""

    def __init__(self, session: SASession):
        self._session = session

    # --- Sessions ---

    def get_or_create_session(self, session_id: str, user_id: str, title: Optional[str] = None) -> ChatSession:
        chat_session = self._session.get(ChatSession, session_id)
        if chat_session is None:
            chat_session = ChatSession(id=session_id, user_id=user_id, title=title)
            self._session.add(chat_session)
            self._session.flush()
        return chat_session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._session.get(ChatSession, session_id)

    # --- Messages (conversation memory) ---

    def add_message(self, session_id: str, role: str, content: str) -> Message:
        message = Message(session_id=session_id, role=role, content=content)
        self._session.add(message)
        self._session.flush()
        return message

    def list_messages(self, session_id: str, since_message_id: int = 0) -> List[Message]:
        stmt = (
            select(Message)
            .where(Message.session_id == session_id, Message.id > since_message_id)
            .order_by(Message.id.asc())
        )
        return list(self._session.scalars(stmt))

    # --- Session summary (session memory) ---

    def get_session_summary(self, session_id: str) -> Optional[SessionSummary]:
        return self._session.get(SessionSummary, session_id)

    def upsert_session_summary(self, session_id: str, summary_text: str, through_message_id: int) -> SessionSummary:
        row = self._session.get(SessionSummary, session_id)
        if row is None:
            row = SessionSummary(
                session_id=session_id,
                summary=summary_text,
                summarized_through_message_id=through_message_id,
            )
            self._session.add(row)
        else:
            row.summary = summary_text
            row.summarized_through_message_id = through_message_id
        self._session.flush()
        return row

    # --- Long-term user memory ---

    def fact_exists(self, user_id: str, fact: str) -> bool:
        normalized = _normalize(fact)
        stmt = select(UserMemory).where(UserMemory.user_id == user_id)
        return any(_normalize(row.fact) == normalized for row in self._session.scalars(stmt))

    def add_user_fact(
        self,
        user_id: str,
        fact: str,
        category: Optional[str] = None,
        source_session_id: Optional[str] = None,
    ) -> UserMemory:
        row = UserMemory(user_id=user_id, fact=fact, category=category, source_session_id=source_session_id)
        self._session.add(row)
        self._session.flush()
        return row

    def list_user_facts(self, user_id: str, limit: int = 10) -> List[UserMemory]:
        stmt = (
            select(UserMemory)
            .where(UserMemory.user_id == user_id)
            .order_by(UserMemory.created_at.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt))

    # --- Launches (retrieval of previous due diligence reports and decisions) ---

    def record_launch(self, launch: Launch) -> Launch:
        self._session.add(launch)
        self._session.flush()
        return launch

    def get_launch(self, launch_id: str) -> Optional[Launch]:
        return self._session.get(Launch, launch_id)

    def list_recent_launches(self, user_id: str, limit: int = 10) -> List[Launch]:
        stmt = (
            select(Launch)
            .where(Launch.user_id == user_id)
            .order_by(Launch.created_at.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt))

    def search_launches(self, user_id: str, keyword: str, limit: int = 5) -> List[Launch]:
        pattern = f"%{keyword}%"
        stmt = (
            select(Launch)
            .where(
                Launch.user_id == user_id,
                (Launch.topic.ilike(pattern)) | (Launch.blog_title.ilike(pattern)),
            )
            .order_by(Launch.created_at.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt))

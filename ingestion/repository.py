from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Document, IngestionStatus

_NON_DELETED_STATUSES = [
    IngestionStatus.PENDING.value,
    IngestionStatus.PROCESSING.value,
    IngestionStatus.COMPLETED.value,
    IngestionStatus.FAILED.value,
]


class DocumentRepository:
    """SQLAlchemy-backed persistence for document ingestion status. No raw SQL, ORM only."""

    def __init__(self, session: Session):
        self._session = session

    def create(self, document: Document) -> Document:
        self._session.add(document)
        self._session.flush()
        return document

    def get(self, document_id: str) -> Optional[Document]:
        return self._session.get(Document, document_id)

    def list(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Document]:
        stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        if status:
            stmt = stmt.where(Document.status == status)
        return list(self._session.scalars(stmt))

    def find_active_by_hash(self, content_hash: str) -> Optional[Document]:
        stmt = select(Document).where(
            Document.content_hash == content_hash,
            Document.status.in_(_NON_DELETED_STATUSES),
        )
        return self._session.scalars(stmt).first()

    def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Document]:
        document = self.get(document_id)
        if document is None:
            return None
        document.status = status
        if chunk_count is not None:
            document.chunk_count = chunk_count
        document.error_message = error_message
        self._session.flush()
        return document

    def soft_delete(self, document_id: str) -> Optional[Document]:
        return self.update_status(document_id, IngestionStatus.DELETED.value, error_message=None)

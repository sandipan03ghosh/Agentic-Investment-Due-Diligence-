from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
from typing import Any, Callable, List, Optional

import pytest
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import ingestion.service as service_module
from ingestion.errors import (
    DocumentNotFoundError,
    DocumentNotReindexableError,
    DuplicateDocumentError,
    InvalidCollectionNameError,
)
from ingestion.models import Base, IngestionStatus
from ingestion.service import IngestionService
from vector_store.base import ChunkRecord, MetadataFilter, SearchResult, VectorRecord, VectorStorePort


class FakeVectorStore(VectorStorePort):
    """In-memory VectorStorePort double so tests never need a live Weaviate instance."""

    def __init__(self):
        self.indexed_calls: List[tuple] = []
        self.deleted_calls: List[tuple] = []
        self._collections: set[str] = set()

    def is_ready(self) -> bool:
        return True

    def ensure_collection(self, collection_name: str) -> None:
        self._collections.add(collection_name)

    def list_collections(self) -> List[str]:
        return sorted(self._collections)

    def delete_collection(self, collection_name: str) -> None:
        self._collections.discard(collection_name)

    def upsert(self, collection_name: str, records: List[VectorRecord]) -> List[str]:
        self._collections.add(collection_name)
        return [r.key for r in records]

    def delete_where(self, collection_name: str, field_name: str, value: Any) -> int:
        self.deleted_calls.append((collection_name, field_name, value))
        return 0

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        filters: Optional[MetadataFilter] = None,
    ) -> List[SearchResult]:
        return []

    def index_chunks(
        self,
        chunks: List[ChunkRecord],
        collection_name: str,
        embed_fn: Callable[[List[str]], List[List[float]]],
    ) -> List[str]:
        self.indexed_calls.append((collection_name, list(chunks)))
        return [f"fake-{c.document_id}-{c.chunk_index}" for c in chunks]

    def delete_document(self, document_id: str, collection_name: str) -> int:
        self.deleted_calls.append((collection_name, "document_id", document_id))
        return 0


@pytest.fixture()
def isolated_session(tmp_path, monkeypatch):
    """Point ingestion.service at a throwaway SQLite DB instead of the real data/ingestion.db."""
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
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def service(tmp_path, fake_vector_store):
    return IngestionService(
        upload_dir=tmp_path / "uploads",
        max_upload_size_mb=1,
        chunk_size=200,
        chunk_overlap=20,
        vector_store=fake_vector_store,
        default_collection="product_briefs",
    )


def _make_upload(name: str, content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=name)


def test_ingest_upload_creates_pending_document(service, isolated_session, fake_vector_store):
    document = service.ingest_upload(_make_upload("brief.txt", b"Product launch content. " * 10))

    assert document.status == IngestionStatus.PENDING.value
    assert document.original_filename == "brief.txt"
    assert document.file_type == "txt"
    assert document.collection == "product_briefs"


def test_ingest_upload_respects_explicit_collection(service, isolated_session, fake_vector_store):
    document = service.ingest_upload(_make_upload("brief.txt", b"content"), collection="workspace_a")
    assert document.collection == "workspace_a"


def test_ingest_upload_rejects_invalid_collection_name(service, isolated_session, fake_vector_store):
    with pytest.raises(InvalidCollectionNameError):
        service.ingest_upload(_make_upload("brief.txt", b"content"), collection="bad name!")


def test_ingest_upload_rejects_duplicate_content(service, isolated_session, fake_vector_store):
    content = b"Duplicate content for hashing test."
    service.ingest_upload(_make_upload("a.txt", content))

    with pytest.raises(DuplicateDocumentError):
        service.ingest_upload(_make_upload("b.txt", content))


def test_process_document_completes_and_indexes_chunks(service, isolated_session, fake_vector_store):
    document = service.ingest_upload(_make_upload("brief.txt", b"Launch strategy. " * 40))

    service.process_document(document.id)

    updated = service.get_document(document.id)
    assert updated.status == IngestionStatus.COMPLETED.value
    assert updated.chunk_count > 0
    assert len(fake_vector_store.indexed_calls) == 1
    collection, chunks = fake_vector_store.indexed_calls[0]
    assert collection == "product_briefs"
    assert chunks[0].document_id == document.id


def test_process_document_marks_failed_on_extraction_error(
    service, isolated_session, fake_vector_store, monkeypatch
):
    document = service.ingest_upload(_make_upload("brief.txt", b"content"))

    def boom(extension):
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(service_module, "get_loader", boom)
    service.process_document(document.id)

    updated = service.get_document(document.id)
    assert updated.status == IngestionStatus.FAILED.value
    assert updated.error_message


def test_delete_document_soft_deletes_and_removes_file(service, isolated_session, fake_vector_store):
    document = service.ingest_upload(_make_upload("brief.txt", b"content to delete"))
    stored_path = service._upload_dir / document.id / document.stored_filename
    assert stored_path.exists()

    deleted = service.delete_document(document.id)

    assert deleted.status == IngestionStatus.DELETED.value
    assert not stored_path.exists()
    assert any(call[2] == document.id for call in fake_vector_store.deleted_calls)


def test_delete_document_raises_for_unknown_id(service, isolated_session, fake_vector_store):
    with pytest.raises(DocumentNotFoundError):
        service.delete_document("does-not-exist")


def test_reupload_after_delete_is_allowed(service, isolated_session, fake_vector_store):
    content = b"content that gets deleted then reuploaded"
    document = service.ingest_upload(_make_upload("brief.txt", content))
    service.delete_document(document.id)

    new_document = service.ingest_upload(_make_upload("brief.txt", content))
    assert new_document.id != document.id


def test_prepare_reindex_requires_existing_file(service, isolated_session, fake_vector_store):
    document = service.ingest_upload(_make_upload("brief.txt", b"content"))
    service.delete_document(document.id)

    with pytest.raises(DocumentNotReindexableError):
        service.prepare_reindex(document.id)


def test_prepare_reindex_sets_processing_status(service, isolated_session, fake_vector_store):
    document = service.ingest_upload(_make_upload("brief.txt", b"content for reindex"))
    service.process_document(document.id)

    reindexed = service.prepare_reindex(document.id)
    assert reindexed.status == IngestionStatus.PROCESSING.value


def test_list_collections_delegates_to_vector_store(service, fake_vector_store):
    fake_vector_store.ensure_collection("product_briefs")
    fake_vector_store.ensure_collection("workspace_a")
    assert service.list_collections() == ["product_briefs", "workspace_a"]

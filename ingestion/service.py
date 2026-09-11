from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import UploadFile

from config import settings
from content_security import sanitize_retrieved_text
from logging_config import get_logger
from vector_store import ChunkRecord, VectorStorePort, embed_texts, get_vector_store

from .chunker import TextChunker
from .db import get_session
from .errors import (
    DocumentNotFoundError,
    DocumentNotReindexableError,
    DuplicateDocumentError,
    ExtractionError,
)
from .hashing import new_sha256
from .loaders import get_loader
from .models import Document, IngestionStatus
from .repository import DocumentRepository
from .security import (
    delete_upload_file,
    resolve_safe_path,
    sanitize_filename,
    stream_save_with_limit,
    validate_collection_name,
    validate_extension,
)

logger = get_logger(__name__)


def _safe_error_message(exc: Exception) -> str:
    """Short, generic message safe to store/display; full details go only to the logs."""
    return f"{type(exc).__name__}: ingestion failed. See server logs for details."


class IngestionService:
    """Orchestrates the ingestion pipeline. Depends on the repository and the abstract
    VectorStorePort (not on Weaviate/weaviate-client directly) so the vector backend can be
    swapped by changing vector_store.factory.get_vector_store() alone."""

    def __init__(
        self,
        upload_dir: Path,
        max_upload_size_mb: int,
        chunk_size: int,
        chunk_overlap: int,
        vector_store: VectorStorePort,
        default_collection: str,
    ):
        self._upload_dir = upload_dir
        self._max_upload_bytes = max_upload_size_mb * 1024 * 1024
        self._chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._vector_store = vector_store
        self._default_collection = default_collection

    def ingest_upload(self, upload_file: UploadFile, collection: Optional[str] = None) -> Document:
        original_filename = upload_file.filename or ""
        extension = validate_extension(original_filename)
        safe_name = sanitize_filename(original_filename)
        target_collection = validate_collection_name(collection or self._default_collection)

        document_id = str(uuid.uuid4())
        dest_path = resolve_safe_path(self._upload_dir, document_id, safe_name)

        hasher = new_sha256()
        size = stream_save_with_limit(upload_file.file, dest_path, self._max_upload_bytes, hasher)
        content_hash = hasher.hexdigest()

        with get_session() as session:
            repo = DocumentRepository(session)
            existing = repo.find_active_by_hash(content_hash)
            if existing is not None:
                dest_path.unlink(missing_ok=True)
                logger.info(
                    "duplicate_upload_rejected",
                    extra={"content_hash": content_hash, "existing_document_id": existing.id},
                )
                raise DuplicateDocumentError(
                    "A document with identical content already exists.", existing.id
                )

            document = Document(
                id=document_id,
                original_filename=safe_name,
                stored_filename=safe_name,
                file_type=extension.lstrip("."),
                file_size_bytes=size,
                content_hash=content_hash,
                collection=target_collection,
                status=IngestionStatus.PENDING.value,
            )
            repo.create(document)
            logger.info(
                "document_created",
                extra={
                    "document_id": document.id,
                    "file_type": document.file_type,
                    "size": size,
                    "collection": target_collection,
                },
            )
            return document

    def process_document(self, document_id: str) -> None:
        with get_session() as session:
            repo = DocumentRepository(session)
            document = repo.get(document_id)
            if document is None:
                logger.error("process_document_missing", extra={"document_id": document_id})
                return
            repo.update_status(document_id, IngestionStatus.PROCESSING.value)
            file_type = document.file_type
            stored_filename = document.stored_filename
            original_filename = document.original_filename
            collection = document.collection

        file_path = resolve_safe_path(self._upload_dir, document_id, stored_filename)

        try:
            # Clear any prior vectors first so a reindex never leaves stale/orphaned chunks
            # (harmless no-op on a first-time ingest).
            self._vector_store.delete_document(document_id, collection)

            loader = get_loader(f".{file_type}")
            loaded = loader.load(file_path)
            # Uploaded file content is untrusted data -- sanitize before it's chunked,
            # embedded, or ever reaches a prompt.
            sanitized_text = sanitize_retrieved_text(loaded.text)
            chunks_text = self._chunker.split(sanitized_text)
            if len(chunks_text) > settings.MAX_CHUNKS_PER_DOCUMENT:
                raise ExtractionError(
                    f"Document split into {len(chunks_text)} chunks, exceeding the "
                    f"maximum allowed ({settings.MAX_CHUNKS_PER_DOCUMENT}); rejected "
                    "to avoid excessive embedding/storage cost."
                )

            title = sanitize_retrieved_text(loaded.metadata.get("title") or Path(original_filename).stem)
            chunk_records = [
                ChunkRecord(
                    document_id=document_id,
                    chunk_index=i,
                    text=chunk,
                    title=title,
                    # Synthetic reference URI instead of the real disk path, so the vector
                    # payload (which flows into research_node results) never leaks server filesystem layout.
                    url=f"upload://{document_id}/{original_filename}",
                    file_type=file_type,
                )
                for i, chunk in enumerate(chunks_text)
            ]
            self._vector_store.index_chunks(chunk_records, collection, embed_texts)

            with get_session() as session:
                repo = DocumentRepository(session)
                repo.update_status(document_id, IngestionStatus.COMPLETED.value, chunk_count=len(chunk_records))
            logger.info(
                "document_processed", extra={"document_id": document_id, "chunk_count": len(chunk_records)}
            )
        except Exception as exc:
            logger.exception("document_processing_failed", extra={"document_id": document_id})
            with get_session() as session:
                repo = DocumentRepository(session)
                repo.update_status(
                    document_id,
                    IngestionStatus.FAILED.value,
                    error_message=_safe_error_message(exc),
                )

    def get_document(self, document_id: str) -> Document:
        with get_session() as session:
            repo = DocumentRepository(session)
            document = repo.get(document_id)
            if document is None:
                raise DocumentNotFoundError(f"No document with id '{document_id}'")
            return document

    def list_documents(self, status: Optional[str], limit: int, offset: int) -> List[Document]:
        with get_session() as session:
            repo = DocumentRepository(session)
            return repo.list(status=status, limit=limit, offset=offset)

    def list_collections(self) -> List[str]:
        return self._vector_store.list_collections()

    def delete_document(self, document_id: str) -> Document:
        with get_session() as session:
            repo = DocumentRepository(session)
            document = repo.get(document_id)
            if document is None:
                raise DocumentNotFoundError(f"No document with id '{document_id}'")
            stored_filename = document.stored_filename
            collection = document.collection

        self._vector_store.delete_document(document_id, collection)
        delete_upload_file(self._upload_dir, Path(document_id) / stored_filename)

        with get_session() as session:
            repo = DocumentRepository(session)
            document = repo.soft_delete(document_id)
            logger.info("document_deleted", extra={"document_id": document_id})
            return document

    def prepare_reindex(self, document_id: str) -> Document:
        with get_session() as session:
            repo = DocumentRepository(session)
            document = repo.get(document_id)
            if document is None:
                raise DocumentNotFoundError(f"No document with id '{document_id}'")
            if document.status == IngestionStatus.DELETED.value:
                raise DocumentNotReindexableError("Document was deleted; re-upload instead of reindexing.")

            file_path = resolve_safe_path(self._upload_dir, document_id, document.stored_filename)
            if not file_path.exists():
                raise DocumentNotReindexableError("Original file is no longer available on disk.")

            document = repo.update_status(document_id, IngestionStatus.PROCESSING.value)
            logger.info("document_reindex_scheduled", extra={"document_id": document_id})
            return document


def build_service() -> IngestionService:
    return IngestionService(
        upload_dir=Path(settings.UPLOAD_DIR),
        max_upload_size_mb=settings.MAX_UPLOAD_SIZE_MB,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        vector_store=get_vector_store(),
        default_collection=settings.WEAVIATE_COLLECTION,
    )

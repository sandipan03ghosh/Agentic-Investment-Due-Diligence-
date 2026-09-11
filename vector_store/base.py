from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class VectorRecord:
    """Generic (key, payload, vector) triple to upsert into a vector store collection.
    `key` is a stable string identifier; concrete adapters turn it into a backend-specific id."""

    key: str
    properties: Dict[str, Any]
    vector: List[float]


@dataclass
class ChunkRecord:
    """Ingestion-facing chunk description. Converted into a VectorRecord by the store."""

    document_id: str
    chunk_index: int
    text: str
    title: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[str] = None
    source: Optional[str] = None
    file_type: Optional[str] = None


@dataclass
class SearchResult:
    title: str = ""
    url: str = ""
    snippet: str = ""
    published_at: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    file_type: Optional[str] = None


@dataclass
class MetadataFilter:
    """Equality-based metadata filter, ANDed across all set fields. All fields optional."""

    document_id: Optional[str] = None
    file_type: Optional[str] = None
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.document_id or self.file_type or self.source or self.extra)


class VectorStorePort(ABC):
    """Abstract vector store contract.

    Implement the primitives below (collection lifecycle, batch upsert, filtered delete,
    filtered similarity search, readiness) to add a new vector database backend. The
    index_chunks/delete_document convenience methods are provided for free on top of them
    (Template Method pattern) so callers never need backend-specific knowledge.
    """

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def ensure_collection(self, collection_name: str) -> None: ...

    @abstractmethod
    def list_collections(self) -> List[str]: ...

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None: ...

    @abstractmethod
    def upsert(self, collection_name: str, records: List[VectorRecord]) -> List[str]:
        """Batch-write records. Implementations must use the backend's bulk/batch API,
        not one write per record."""

    @abstractmethod
    def delete_where(self, collection_name: str, field_name: str, value: Any) -> int:
        """Delete all objects where `field_name == value`. Returns the count deleted."""

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        filters: Optional[MetadataFilter] = None,
    ) -> List[SearchResult]: ...

    # --- Convenience methods built on the primitives above (backend-agnostic) ---

    def index_chunks(
        self,
        chunks: List[ChunkRecord],
        collection_name: str,
        embed_fn: Callable[[List[str]], List[List[float]]],
    ) -> List[str]:
        if not chunks:
            return []
        vectors = embed_fn([c.text for c in chunks])
        records = [
            VectorRecord(
                key=f"{c.document_id}:{c.chunk_index}",
                properties={
                    k: v
                    for k, v in {
                        "title": c.title,
                        "url": c.url,
                        "snippet": c.text,
                        "published_at": c.published_at,
                        "source": c.source or "ingestion",
                        "document_id": c.document_id,
                        "chunk_index": c.chunk_index,
                        "file_type": c.file_type,
                    }.items()
                    if v is not None
                },
                vector=vec,
            )
            for c, vec in zip(chunks, vectors)
        ]
        return self.upsert(collection_name, records)

    def delete_document(self, document_id: str, collection_name: str) -> int:
        return self.delete_where(collection_name, "document_id", document_id)

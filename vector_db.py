from __future__ import annotations

from typing import Dict, List, Optional

from config import settings
from vector_store import ChunkRecord, MetadataFilter, VectorRecord, embed_texts, get_vector_store

_COLLECTION = settings.WEAVIATE_COLLECTION


def ensure_collection(collection_name: str = _COLLECTION) -> None:
    get_vector_store().ensure_collection(collection_name)


def is_ready() -> bool:
    """Lightweight connectivity check used by the ingestion API's health endpoint."""
    return get_vector_store().is_ready()


def index_documents(docs: List[Dict], collection_name: str = _COLLECTION, recreate: bool = False):
    """
    Index a list of documents into the vector store (Weaviate, via vector_store.get_vector_store()).
    If recreate=True the collection is dropped and recreated; otherwise objects are upserted in batch.

    Kept behavior-compatible with the original implementation used by
    scripts/ingest_documents.py and scripts/ingest_md.py: only title/url/snippet/published_at
    (plus any extra doc keys not in the reserved set) are persisted as payload. Note this means
    the raw "text" field of a doc is NOT stored here (pre-existing behavior, preserved as-is).
    """
    store = get_vector_store()
    if recreate:
        store.delete_collection(collection_name)
    store.ensure_collection(collection_name)

    ids = [int(d.get("id") or i) for i, d in enumerate(docs, start=1)]
    texts = [d.get("text") or d.get("title") or "" for d in docs]
    vectors = embed_texts(texts)
    payloads = [
        {
            "title": d.get("title"),
            "url": d.get("url"),
            "snippet": d.get("snippet"),
            "published_at": d.get("published_at"),
            **{k: v for k, v in d.items() if k not in {"id", "title", "text", "url", "snippet", "published_at"}},
        }
        for d in docs
    ]

    records = [
        VectorRecord(
            key=str(ids[i]),
            properties={k: v for k, v in payloads[i].items() if v is not None},
            vector=vectors[i],
        )
        for i in range(len(docs))
    ]
    store.upsert(collection_name, records)


def query_kb(
    query: str,
    top_k: int = 5,
    collection_name: str = _COLLECTION,
    document_id: Optional[str] = None,
    file_type: Optional[str] = None,
    source: Optional[str] = None,
) -> List[Dict]:
    """Return list of dicts matching the raw search-result shape used in backend.research_node.

    document_id/file_type/source are optional metadata filters, additive to the original
    contract (default None => identical behavior to before).
    """
    try:
        vec = embed_texts([query])[0]
    except Exception:
        return []

    filters = None
    if document_id or file_type or source:
        filters = MetadataFilter(document_id=document_id, file_type=file_type, source=source)

    results = get_vector_store().search(collection_name, vec, top_k, filters)
    return [
        {
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "published_at": r.published_at,
            "source": r.source,
            "score": r.score,
            "document_id": r.document_id,
            "chunk_index": r.chunk_index,
            "file_type": r.file_type,
        }
        for r in results
    ]


def index_chunks(document_id: str, chunks: List[Dict], collection_name: str = _COLLECTION) -> List[str]:
    """
    Index chunk records produced by the ingestion pipeline (see ingestion/service.py).

    Each chunk dict is expected to have: text, chunk_index, title, url, file_type,
    and optionally published_at/source. Unlike index_documents(), this stores the
    actual chunk text (under "snippet") so it is retrievable via query_kb().
    Uses the vector store's batch upsert. Returns the list of ids written.
    """
    records = [
        ChunkRecord(
            document_id=document_id,
            chunk_index=c["chunk_index"],
            text=c["text"],
            title=c.get("title"),
            url=c.get("url"),
            published_at=c.get("published_at"),
            source=c.get("source"),
            file_type=c.get("file_type"),
        )
        for c in chunks
    ]
    return get_vector_store().index_chunks(records, collection_name, embed_texts)


def delete_document(document_id: str, collection_name: str = _COLLECTION) -> int:
    """Delete all chunks belonging to a document_id. Returns the number of objects deleted."""
    return get_vector_store().delete_document(document_id, collection_name)

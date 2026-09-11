from __future__ import annotations

from typing import Any, List, Optional

from vector_store.base import ChunkRecord, MetadataFilter, SearchResult, VectorRecord, VectorStorePort


class MinimalVectorStore(VectorStorePort):
    """Implements only the abstract primitives, so tests exercise the base class's
    index_chunks/delete_document Template Method logic — the same code path WeaviateVectorStore
    relies on, since it doesn't override those convenience methods."""

    def __init__(self):
        self.upserted: List[tuple] = []
        self.deleted_where: List[tuple] = []
        self.collections: set[str] = set()

    def is_ready(self) -> bool:
        return True

    def ensure_collection(self, collection_name: str) -> None:
        self.collections.add(collection_name)

    def list_collections(self) -> List[str]:
        return sorted(self.collections)

    def delete_collection(self, collection_name: str) -> None:
        self.collections.discard(collection_name)

    def upsert(self, collection_name: str, records: List[VectorRecord]) -> List[str]:
        self.upserted.append((collection_name, records))
        return [r.key for r in records]

    def delete_where(self, collection_name: str, field_name: str, value: Any) -> int:
        self.deleted_where.append((collection_name, field_name, value))
        return 3

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        filters: Optional[MetadataFilter] = None,
    ) -> List[SearchResult]:
        return []


def _fake_embed(texts: List[str]) -> List[List[float]]:
    return [[float(len(t))] for t in texts]


def test_index_chunks_builds_vector_records_and_calls_upsert():
    store = MinimalVectorStore()
    chunks = [
        ChunkRecord(document_id="doc-1", chunk_index=0, text="hello world", title="T", file_type="txt"),
        ChunkRecord(document_id="doc-1", chunk_index=1, text="second chunk", title="T", file_type="txt"),
    ]

    ids = store.index_chunks(chunks, "my_collection", _fake_embed)

    assert ids == ["doc-1:0", "doc-1:1"]
    assert len(store.upserted) == 1
    collection_name, records = store.upserted[0]
    assert collection_name == "my_collection"
    assert records[0].key == "doc-1:0"
    assert records[0].properties["snippet"] == "hello world"
    assert records[0].properties["document_id"] == "doc-1"
    assert records[0].vector == [11.0]


def test_index_chunks_with_empty_list_does_not_call_upsert():
    store = MinimalVectorStore()
    assert store.index_chunks([], "my_collection", _fake_embed) == []
    assert store.upserted == []


def test_delete_document_delegates_to_delete_where_with_document_id_field():
    store = MinimalVectorStore()
    result = store.delete_document("doc-42", "my_collection")

    assert result == 3
    assert store.deleted_where == [("my_collection", "document_id", "doc-42")]

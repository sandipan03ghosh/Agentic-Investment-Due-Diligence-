from __future__ import annotations

from typing import Any, List, Optional
from urllib.parse import urlparse

from .base import MetadataFilter, SearchResult, VectorRecord, VectorStorePort

try:
    import weaviate
    from weaviate.auth import AuthApiKey
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.query import Filter, MetadataQuery
    from weaviate.util import generate_uuid5
except Exception:  # optional dep may be missing in dev environments
    weaviate = None  # type: ignore
    generate_uuid5 = None  # type: ignore

def _build_filter(filters: Optional[MetadataFilter]):
    if filters is None or filters.is_empty():
        return None
    clauses = []
    if filters.document_id:
        clauses.append(Filter.by_property("document_id").equal(filters.document_id))
    if filters.file_type:
        clauses.append(Filter.by_property("file_type").equal(filters.file_type))
    if filters.source:
        clauses.append(Filter.by_property("source").equal(filters.source))
    for key, value in filters.extra.items():
        clauses.append(Filter.by_property(key).equal(value))
    if not clauses:
        return None
    combined = clauses[0]
    for clause in clauses[1:]:
        combined = combined & clause
    return combined


class WeaviateVectorStore(VectorStorePort):
    """Weaviate-backed implementation of VectorStorePort. Vectors are self-provided
    (vectorizer=none) since embeddings are generated locally via sentence-transformers."""

    def __init__(self, url: str, api_key: Optional[str] = None, grpc_port: int = 50051):
        if weaviate is None:
            raise RuntimeError("weaviate-client is required for vector store operations")
        self._url = url
        self._api_key = api_key
        self._grpc_port = grpc_port

    def _connect(self):
        parsed = urlparse(self._url or "http://localhost:8080")
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 8080)
        secure = parsed.scheme == "https"
        auth = AuthApiKey(self._api_key) if self._api_key else None
        return weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=secure,
            grpc_host=host,
            grpc_port=self._grpc_port,
            grpc_secure=secure,
            auth_credentials=auth,
        )

    def is_ready(self) -> bool:
        try:
            with self._connect() as client:
                return bool(client.is_ready())
        except Exception:
            return False

    def ensure_collection(self, collection_name: str) -> None:
        with self._connect() as client:
            if client.collections.exists(collection_name):
                return
            client.collections.create(
                collection_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="url", data_type=DataType.TEXT),
                    Property(name="snippet", data_type=DataType.TEXT),
                    Property(name="published_at", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="document_id", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="file_type", data_type=DataType.TEXT),
                ],
            )

    def list_collections(self) -> List[str]:
        with self._connect() as client:
            return sorted(client.collections.list_all().keys())

    def delete_collection(self, collection_name: str) -> None:
        with self._connect() as client:
            if client.collections.exists(collection_name):
                client.collections.delete(collection_name)

    def upsert(self, collection_name: str, records: List[VectorRecord]) -> List[str]:
        if not records:
            return []
        self.ensure_collection(collection_name)

        written_ids: List[str] = []
        with self._connect() as client:
            collection = client.collections.get(collection_name)
            # Real batch write (dynamic batching auto-tunes size/concurrency and retries),
            # instead of one insert/exists round-trip per object.
            with collection.batch.dynamic() as batch:
                for record in records:
                    obj_uuid = generate_uuid5(record.key)
                    batch.add_object(properties=record.properties, vector=record.vector, uuid=obj_uuid)
                    written_ids.append(str(obj_uuid))

            failed = collection.batch.failed_objects
            if failed:
                raise RuntimeError(
                    f"{len(failed)} object(s) failed to upsert into Weaviate collection '{collection_name}'"
                )

        return written_ids

    def delete_where(self, collection_name: str, field_name: str, value: Any) -> int:
        with self._connect() as client:
            if not client.collections.exists(collection_name):
                return 0
            collection = client.collections.get(collection_name)
            result = collection.data.delete_many(where=Filter.by_property(field_name).equal(value))
            return int(getattr(result, "successful", 0) or 0)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        filters: Optional[MetadataFilter] = None,
    ) -> List[SearchResult]:
        try:
            with self._connect() as client:
                if not client.collections.exists(collection_name):
                    return []
                collection = client.collections.get(collection_name)
                where = _build_filter(filters)
                res = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=top_k,
                    filters=where,
                    return_metadata=MetadataQuery(distance=True),
                )
                results: List[SearchResult] = []
                for obj in res.objects:
                    p = obj.properties or {}
                    results.append(
                        SearchResult(
                            title=p.get("title") or "",
                            url=p.get("url") or "",
                            snippet=p.get("snippet") or "",
                            published_at=p.get("published_at"),
                            source=p.get("source") or "vector_kb",
                            score=obj.metadata.distance if obj.metadata else None,
                            document_id=p.get("document_id"),
                            chunk_index=p.get("chunk_index"),
                            file_type=p.get("file_type"),
                        )
                    )
                return results
        except Exception:
            return []

"""Vector store adapter layer: an abstract port plus a Weaviate implementation.

To replace Weaviate with another vector database, implement VectorStorePort in a
new module and point vector_store.factory.get_vector_store() at it. No other
code (ingestion pipeline, vector_db.py facade, backend.py) needs to change.
"""

from .base import ChunkRecord, MetadataFilter, SearchResult, VectorRecord, VectorStorePort
from .embeddings import embed_texts, score_pairs
from .factory import get_vector_store

__all__ = [
    "ChunkRecord",
    "MetadataFilter",
    "SearchResult",
    "VectorRecord",
    "VectorStorePort",
    "embed_texts",
    "score_pairs",
    "get_vector_store",
]

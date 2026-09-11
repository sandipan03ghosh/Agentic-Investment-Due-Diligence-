from __future__ import annotations

from typing import Optional

from config import settings

from .base import VectorStorePort
from .weaviate_store import WeaviateVectorStore

_instance: Optional[VectorStorePort] = None


def get_vector_store() -> VectorStorePort:
    """Single place that selects the concrete vector store implementation.

    To swap Weaviate for another backend, implement VectorStorePort elsewhere and
    return that implementation here instead — no other code needs to change.
    """
    global _instance
    if _instance is None:
        _instance = WeaviateVectorStore(url=settings.WEAVIATE_URL, api_key=settings.WEAVIATE_API_KEY)
    return _instance


def reset_vector_store() -> None:
    """Test/ops hook to force re-creation of the singleton (e.g. after config changes)."""
    global _instance
    _instance = None

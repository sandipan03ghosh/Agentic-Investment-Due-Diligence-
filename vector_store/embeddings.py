from __future__ import annotations

import threading
from typing import List, Optional

from config import settings

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # optional dep may be missing in dev environments
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore

_lock = threading.Lock()
_model: Optional["SentenceTransformer"] = None

_reranker_lock = threading.Lock()
_reranker: Optional["CrossEncoder"] = None


def _require_deps() -> None:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required to generate embeddings")


def get_model() -> "SentenceTransformer":
    """Lazily instantiate and cache the embedding model (avoids reloading it on every call)."""
    global _model
    _require_deps()
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer(settings.EMBED_MODEL)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    return model.encode(texts, show_progress_bar=False).tolist()


def get_reranker() -> "CrossEncoder":
    """Lazily instantiate and cache the cross-encoder reranker model."""
    global _reranker
    if CrossEncoder is None:
        raise RuntimeError("sentence-transformers is required to run the reranker")
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker


def score_pairs(query: str, passages: List[str]) -> List[float]:
    """Score each (query, passage) pair for relevance. Higher is more relevant (raw logits)."""
    if not passages:
        return []
    reranker = get_reranker()
    pairs = [(query, p) for p in passages]
    scores = reranker.predict(pairs)
    return [float(s) for s in scores]

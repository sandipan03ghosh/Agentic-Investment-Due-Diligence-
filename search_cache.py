from __future__ import annotations

import copy
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from logging_config import get_logger

logger = get_logger(__name__)


def make_key(*parts: Any) -> str:
    """Build a stable, opaque cache key from arbitrary parts (never stores raw query
    text as a visible key — hashed so nothing sensitive is exposed via the cache's keys)."""
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8", errors="replace")).hexdigest()


class SearchCache:
    """Thread-safe, bounded, TTL-expiring in-memory cache for recent search results.

    - Every read/write is guarded by a single lock (safe under concurrent Streamlit
      sessions or concurrent API requests).
    - get()/set() operate on deep copies, so callers can never mutate cached state and
      no cached state can be mutated out from under another caller.
    - Bounded by max_entries (oldest entry evicted first, LRU-ish via OrderedDict)
      and ttl_seconds (expired entries are treated as a miss and purged lazily).
    """

    def __init__(self, ttl_seconds: int, max_entries: int):
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        try:
            with self._lock:
                entry = self._store.get(key)
                if entry is None:
                    return None
                inserted_at, value = entry
                if time.monotonic() - inserted_at > self._ttl_seconds:
                    del self._store[key]
                    return None
                self._store.move_to_end(key)
                return copy.deepcopy(value)
        except Exception:
            logger.warning("search_cache_get_failed")
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            with self._lock:
                self._store[key] = (time.monotonic(), copy.deepcopy(value))
                self._store.move_to_end(key)
                while len(self._store) > self._max_entries:
                    self._store.popitem(last=False)
        except Exception:
            logger.warning("search_cache_set_failed")

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


_default_cache: Optional[SearchCache] = None
_default_cache_lock = threading.Lock()


def get_default_cache() -> SearchCache:
    """Shared cache instance used by both KB semantic search and web search."""
    global _default_cache
    if _default_cache is None:
        with _default_cache_lock:
            if _default_cache is None:
                from config import settings

                _default_cache = SearchCache(
                    ttl_seconds=settings.SEARCH_CACHE_TTL_SECONDS,
                    max_entries=settings.SEARCH_CACHE_MAX_ENTRIES,
                )
    return _default_cache

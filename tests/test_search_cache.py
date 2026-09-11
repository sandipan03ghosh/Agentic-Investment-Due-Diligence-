from __future__ import annotations

import threading

import pytest

import search_cache as search_cache_module
from search_cache import SearchCache, make_key


def test_make_key_is_stable_and_order_sensitive():
    k1 = make_key("a", "b", 1)
    k2 = make_key("a", "b", 1)
    k3 = make_key("b", "a", 1)

    assert k1 == k2
    assert k1 != k3


def test_make_key_does_not_expose_raw_query_text():
    key = make_key("secret query text", 5)
    assert "secret query text" not in key
    assert len(key) == 64  # sha256 hex digest length


def test_set_and_get_roundtrip():
    cache = SearchCache(ttl_seconds=60, max_entries=10)
    cache.set("k1", {"value": [1, 2, 3]})
    assert cache.get("k1") == {"value": [1, 2, 3]}


def test_get_returns_none_for_missing_key():
    cache = SearchCache(ttl_seconds=60, max_entries=10)
    assert cache.get("missing") is None


def test_get_returns_a_copy_mutating_it_does_not_affect_the_cache():
    cache = SearchCache(ttl_seconds=60, max_entries=10)
    cache.set("k1", {"items": [1, 2, 3]})

    fetched = cache.get("k1")
    fetched["items"].append(4)

    assert cache.get("k1") == {"items": [1, 2, 3]}


def test_set_stores_a_copy_mutating_the_original_does_not_affect_the_cache():
    cache = SearchCache(ttl_seconds=60, max_entries=10)
    original = {"items": [1, 2, 3]}
    cache.set("k1", original)

    original["items"].append(999)

    assert cache.get("k1") == {"items": [1, 2, 3]}


def test_ttl_expiry(monkeypatch):
    current_time = [1000.0]
    monkeypatch.setattr(search_cache_module.time, "monotonic", lambda: current_time[0])

    cache = SearchCache(ttl_seconds=10, max_entries=10)
    cache.set("k1", "value")
    assert cache.get("k1") == "value"

    current_time[0] += 11
    assert cache.get("k1") is None


def test_entry_still_valid_just_under_ttl(monkeypatch):
    current_time = [1000.0]
    monkeypatch.setattr(search_cache_module.time, "monotonic", lambda: current_time[0])

    cache = SearchCache(ttl_seconds=10, max_entries=10)
    cache.set("k1", "value")

    current_time[0] += 9
    assert cache.get("k1") == "value"


def test_lru_eviction_when_max_entries_exceeded():
    cache = SearchCache(ttl_seconds=600, max_entries=2)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.set("k3", "v3")  # should evict k1 (oldest)

    assert cache.get("k1") is None
    assert cache.get("k2") == "v2"
    assert cache.get("k3") == "v3"
    assert len(cache) == 2


def test_get_refreshes_recency_for_eviction_order():
    cache = SearchCache(ttl_seconds=600, max_entries=2)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.get("k1")  # k1 is now most-recently-used
    cache.set("k3", "v3")  # should evict k2, not k1

    assert cache.get("k1") == "v1"
    assert cache.get("k2") is None
    assert cache.get("k3") == "v3"


def test_clear_empties_the_cache():
    cache = SearchCache(ttl_seconds=600, max_entries=10)
    cache.set("k1", "v1")
    cache.clear()

    assert cache.get("k1") is None
    assert len(cache) == 0


def test_concurrent_get_set_is_thread_safe():
    cache = SearchCache(ttl_seconds=600, max_entries=1000)
    errors = []

    def worker(n):
        try:
            for i in range(200):
                cache.set(f"key-{n}-{i}", {"n": n, "i": i})
                cache.get(f"key-{n}-{i}")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(cache) == 1000  # capped at max_entries, no corruption/loss under concurrency

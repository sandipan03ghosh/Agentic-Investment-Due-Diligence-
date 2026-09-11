from __future__ import annotations

import sys
import time
import types
from typing import List, Optional

import pytest

import web_search as web_search_module
from search_cache import SearchCache
from web_search import (
    TavilySearchProvider,
    WebSearchProvider,
    WebSearchResult,
    _is_safe_http_url,
    web_search,
)


class _FakeProvider(WebSearchProvider):
    def __init__(self, results=None, sleep_seconds: float = 0.0, raise_error: bool = False):
        self._results = results or []
        self._sleep_seconds = sleep_seconds
        self._raise_error = raise_error
        self.calls = 0
        self.last_query: Optional[str] = None

    def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        self.calls += 1
        self.last_query = query
        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        if self._raise_error:
            raise RuntimeError("simulated provider failure")
        return self._results


@pytest.fixture(autouse=True)
def _reset_provider_singleton(monkeypatch):
    monkeypatch.setattr(web_search_module, "_provider", None)
    yield
    monkeypatch.setattr(web_search_module, "_provider", None)


@pytest.fixture()
def isolated_cache(monkeypatch) -> SearchCache:
    cache = SearchCache(ttl_seconds=600, max_entries=100)
    monkeypatch.setattr(web_search_module, "get_default_cache", lambda: cache)
    return cache


def _install_fake_tavily_module(monkeypatch, fake_class):
    fake_module = types.ModuleType("langchain_tavily")
    fake_module.TavilySearch = fake_class
    monkeypatch.setitem(sys.modules, "langchain_tavily", fake_module)


# --- URL validation ---


def test_is_safe_http_url_accepts_http_https():
    assert _is_safe_http_url("https://example.com/article") is True
    assert _is_safe_http_url("http://example.com") is True


def test_is_safe_http_url_rejects_unsafe_schemes():
    assert _is_safe_http_url("javascript:alert(1)") is False
    assert _is_safe_http_url("data:text/html,hello") is False
    assert _is_safe_http_url("file:///etc/passwd") is False
    assert _is_safe_http_url("") is False
    assert _is_safe_http_url("not a url") is False
    assert _is_safe_http_url(None) is False


# --- TavilySearchProvider ---


def test_tavily_provider_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilySearchProvider()
    assert provider.search("query") == []


def test_tavily_provider_drops_unsafe_urls_and_malformed_entries(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")

    class _FakeTavilySearch:
        def __init__(self, max_results):
            self.max_results = max_results

        def invoke(self, payload):
            return [
                {"title": "Good result", "url": "https://example.com/a", "content": "snippet a"},
                {"title": "Bad scheme", "url": "javascript:alert(1)", "content": "snippet b"},
                {"title": "Missing url", "url": None, "content": "snippet c"},
                "not-a-dict",
                {"title": 12345, "url": "https://example.com/weird-title", "content": None},
            ]

    _install_fake_tavily_module(monkeypatch, _FakeTavilySearch)

    provider = TavilySearchProvider()
    results = provider.search("query")

    urls = [r.url for r in results]
    assert "https://example.com/a" in urls
    assert "https://example.com/weird-title" in urls
    assert len(results) == 2
    assert all(r.title == "" or isinstance(r.title, str) for r in results)


def test_tavily_provider_handles_call_failure_gracefully(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")

    class _BoomTavilySearch:
        def __init__(self, max_results):
            pass

        def invoke(self, payload):
            raise RuntimeError("network error")

    _install_fake_tavily_module(monkeypatch, _BoomTavilySearch)

    provider = TavilySearchProvider()
    assert provider.search("query") == []


def test_get_web_search_provider_returns_singleton():
    p1 = web_search_module.get_web_search_provider()
    p2 = web_search_module.get_web_search_provider()
    assert p1 is p2


# --- web_search() entrypoint: caching, validation, timeout, failures ---


def test_web_search_uses_whatever_provider_factory_returns(monkeypatch, isolated_cache):
    fake = _FakeProvider(results=[WebSearchResult(title="T", url="https://example.com", snippet="s")])
    monkeypatch.setattr(web_search_module, "get_web_search_provider", lambda: fake)

    results = web_search("some query")

    assert fake.calls == 1
    assert results[0]["title"] == "T"


def test_web_search_caches_results_and_avoids_second_provider_call(monkeypatch, isolated_cache):
    fake = _FakeProvider(results=[WebSearchResult(title="T", url="https://example.com", snippet="s")])
    monkeypatch.setattr(web_search_module, "get_web_search_provider", lambda: fake)

    first = web_search("some query")
    second = web_search("some query")

    assert fake.calls == 1
    assert first == second


def test_web_search_returns_empty_for_blank_query(monkeypatch, isolated_cache):
    fake = _FakeProvider(results=[WebSearchResult(title="T", url="https://example.com", snippet="s")])
    monkeypatch.setattr(web_search_module, "get_web_search_provider", lambda: fake)

    assert web_search("   ") == []
    assert fake.calls == 0


def test_web_search_normalizes_and_caps_oversized_query(monkeypatch, isolated_cache):
    monkeypatch.setattr(web_search_module.settings, "WEB_SEARCH_MAX_QUERY_CHARS", 10)
    fake = _FakeProvider(results=[])
    monkeypatch.setattr(web_search_module, "get_web_search_provider", lambda: fake)

    web_search("y" * 1000)

    assert fake.last_query is not None
    assert len(fake.last_query) == 10


def test_web_search_times_out_and_returns_empty(monkeypatch, isolated_cache):
    monkeypatch.setattr(web_search_module.settings, "WEB_SEARCH_TIMEOUT_SECONDS", 0.05)
    slow = _FakeProvider(
        results=[WebSearchResult(title="T", url="https://example.com", snippet="s")], sleep_seconds=1.0
    )
    monkeypatch.setattr(web_search_module, "get_web_search_provider", lambda: slow)

    results = web_search("slow query")

    assert results == []


def test_web_search_handles_provider_exception_gracefully(monkeypatch, isolated_cache):
    broken = _FakeProvider(raise_error=True)
    monkeypatch.setattr(web_search_module, "get_web_search_provider", lambda: broken)

    assert web_search("query") == []


def test_web_search_result_shape_matches_legacy_dict_contract(monkeypatch, isolated_cache):
    fake = _FakeProvider(
        results=[
            WebSearchResult(
                title="T", url="https://example.com", snippet="s", published_at="2026-01-01", source="example.com"
            )
        ]
    )
    monkeypatch.setattr(web_search_module, "get_web_search_provider", lambda: fake)

    results = web_search("query")

    assert results == [
        {
            "title": "T",
            "url": "https://example.com",
            "snippet": "s",
            "published_at": "2026-01-01",
            "source": "example.com",
        }
    ]

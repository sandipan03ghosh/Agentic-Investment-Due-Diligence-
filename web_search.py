from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from config import settings
from content_security import sanitize_retrieved_text
from logging_config import get_logger
from observability import observe_duration
from search_cache import get_default_cache, make_key

logger = get_logger(__name__)

_MAX_SNIPPET_CHARS = 2000
_MAX_TITLE_CHARS = 300
_MAX_URL_CHARS = 2048

# A small, persistent worker pool used solely to bound how long we wait on a web search
# provider call. Deliberately NOT created (or shut down) per-call: shutting down a pool
# waits for in-flight work to finish, which would silently defeat the timeout. A hung
# provider call simply occupies one worker until it eventually finishes or the process exits.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="web-search")


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    published_at: Optional[str] = None
    source: Optional[str] = None


class WebSearchProvider(ABC):
    """Abstract web search backend. Implement this to add/swap a provider without
    touching any calling code (research_node just calls web_search())."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        raise NotImplementedError


def _is_safe_http_url(url: str) -> bool:
    """Only accept http(s) URLs with a real host. Rejects malformed URLs and unsafe
    schemes (javascript:, data:, file:, etc.) before they can be surfaced as a citation."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _coerce_str(value: Any, max_chars: int) -> str:
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    return value[:max_chars]


class TavilySearchProvider(WebSearchProvider):
    """Wraps langchain_tavily's TavilySearch. Preserves the existing web search
    behavior (env var gate, field mapping, fail-to-empty-list), with added
    response/URL validation before results are trusted."""

    def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        if not os.getenv("TAVILY_API_KEY"):
            return []
        try:
            from langchain_tavily import TavilySearch  # type: ignore

            tool = TavilySearch(max_results=max_results)
            raw_results = tool.invoke({"query": query})
        except Exception:
            logger.warning("tavily_search_call_failed", extra={"status": "failed"})
            return []

        out: List[WebSearchResult] = []
        for r in raw_results or []:
            if not isinstance(r, dict):
                continue
            url = _coerce_str(r.get("url"), _MAX_URL_CHARS)
            if not _is_safe_http_url(url):
                continue
            out.append(
                WebSearchResult(
                    # Web content is the least trusted source in the pipeline -- sanitize
                    # title/snippet before they can ever reach a prompt.
                    title=sanitize_retrieved_text(_coerce_str(r.get("title") or "", _MAX_TITLE_CHARS)),
                    url=url,
                    snippet=sanitize_retrieved_text(
                        _coerce_str(r.get("content") or r.get("snippet") or "", _MAX_SNIPPET_CHARS)
                    ),
                    published_at=_coerce_str(r.get("published_date") or r.get("published_at") or "", 32) or None,
                    source=_coerce_str(r.get("source") or "", 200) or None,
                )
            )
        return out


_provider: Optional[WebSearchProvider] = None


def get_web_search_provider() -> WebSearchProvider:
    """Single swap point for the web search backend. Point this at a different
    WebSearchProvider implementation to change providers app-wide."""
    global _provider
    if _provider is None:
        _provider = TavilySearchProvider()
    return _provider


def _normalize_query(query: str, max_chars: int) -> str:
    return (query or "").strip()[:max_chars]


def _search_with_timeout(
    provider: WebSearchProvider, query: str, max_results: int, timeout_seconds: int
) -> List[WebSearchResult]:
    future = _executor.submit(provider.search, query, max_results)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        logger.warning("web_search_timed_out", extra={"status": "failed"})
        return []
    except Exception:
        logger.warning("web_search_provider_failed", extra={"status": "failed"})
        return []


@observe_duration("web_search")
def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Cached, provider-abstracted, timeout-bounded web search.

    Returns the same list-of-dicts shape the original backend._tavily_search always
    returned (title/url/snippet/published_at/source), so it's a drop-in replacement.
    Never raises: any failure (bad query, timeout, provider error) degrades to [].
    """
    normalized = _normalize_query(query, settings.WEB_SEARCH_MAX_QUERY_CHARS)
    if not normalized:
        return []

    cache = get_default_cache()
    cache_key = make_key("web_search", normalized, max_results)
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("web_search_cache_hit")
        return cached

    provider = get_web_search_provider()
    results = _search_with_timeout(provider, normalized, max_results, settings.WEB_SEARCH_TIMEOUT_SECONDS)

    out = [
        {
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "published_at": r.published_at,
            "source": r.source,
        }
        for r in results
    ]
    cache.set(cache_key, out)
    logger.info("web_search_complete", extra={"result_count": len(out)})
    return out

from __future__ import annotations

import math

import pytest

import retrieval
from search_cache import SearchCache


@pytest.fixture(autouse=True)
def _isolated_search_cache(monkeypatch):
    """semantic_search() now caches vector_db.query_kb() results via a process-wide
    default cache. Without this, tests reusing the same query string (e.g. "q1") across
    functions could read another test's cached/mocked result instead of exercising
    their own mock. Give every test its own fresh cache instance (one per test, reused
    across calls within that test so within-test caching behavior still works)."""
    cache = SearchCache(ttl_seconds=600, max_entries=100)
    monkeypatch.setattr(retrieval, "get_default_cache", lambda: cache)


class _FakeQueryRewriteResult:
    def __init__(self, queries):
        self.queries = queries


class _FakeStructuredLLM:
    def __init__(self, result=None, raise_error=False):
        self._result = result
        self._raise_error = raise_error

    def invoke(self, messages):
        if self._raise_error:
            raise RuntimeError("simulated LLM failure")
        return self._result


class _FakeLLM:
    def __init__(self, structured_llm):
        self._structured_llm = structured_llm

    def with_structured_output(self, schema):
        return self._structured_llm


def test_rewrite_queries_returns_llm_output_on_success():
    fake_result = _FakeQueryRewriteResult(["query one", "query two", "query three"])
    llm = _FakeLLM(_FakeStructuredLLM(result=fake_result))

    queries = retrieval.rewrite_queries(llm, topic="launching a phone", base_queries=["phone launch"])

    assert queries == ["query one", "query two", "query three"]


def test_rewrite_queries_falls_back_on_llm_failure():
    llm = _FakeLLM(_FakeStructuredLLM(raise_error=True))

    queries = retrieval.rewrite_queries(llm, topic="launching a phone", base_queries=["phone launch"])

    assert queries == ["phone launch"]


def test_rewrite_queries_falls_back_to_topic_when_no_base_queries():
    llm = _FakeLLM(_FakeStructuredLLM(raise_error=True))

    queries = retrieval.rewrite_queries(llm, topic="launching a phone", base_queries=[])

    assert queries == ["launching a phone"]


def test_rewrite_queries_falls_back_when_llm_returns_only_blank_queries():
    fake_result = _FakeQueryRewriteResult(["  ", ""])
    llm = _FakeLLM(_FakeStructuredLLM(result=fake_result))

    queries = retrieval.rewrite_queries(llm, topic="launching a phone", base_queries=["hint"])

    assert queries == ["hint"]


def test_semantic_search_dedupes_by_document_id_and_chunk_index(monkeypatch):
    import vector_db

    def fake_query_kb(query, top_k=5, collection_name=None):
        return [
            {"title": "Doc A", "snippet": "chunk 0", "document_id": "doc-1", "chunk_index": 0},
            {"title": "Doc A", "snippet": "chunk 1", "document_id": "doc-1", "chunk_index": 1},
        ]

    monkeypatch.setattr(vector_db, "query_kb", fake_query_kb)

    hits = retrieval.semantic_search(["query a", "query b"], collection_name="product_briefs", top_k_per_query=5)

    # Same two chunks returned for both queries; dedup should collapse to 2, not 4.
    assert len(hits) == 2


def test_semantic_search_dedupes_by_title_and_snippet_when_ids_absent(monkeypatch):
    import vector_db

    def fake_query_kb(query, top_k=5, collection_name=None):
        return [{"title": "Legacy Doc", "snippet": "same content", "document_id": None, "chunk_index": None}]

    monkeypatch.setattr(vector_db, "query_kb", fake_query_kb)

    hits = retrieval.semantic_search(["q1", "q2"], collection_name="product_briefs", top_k_per_query=5)

    assert len(hits) == 1


def test_semantic_search_caches_results_and_avoids_second_query_kb_call(monkeypatch):
    import vector_db

    call_count = {"n": 0}

    def fake_query_kb(query, top_k=5, collection_name=None):
        call_count["n"] += 1
        return [{"title": "Doc", "snippet": "text", "document_id": "d1", "chunk_index": 0}]

    monkeypatch.setattr(vector_db, "query_kb", fake_query_kb)

    retrieval.semantic_search(["repeated query"], collection_name="product_briefs", top_k_per_query=5)
    retrieval.semantic_search(["repeated query"], collection_name="product_briefs", top_k_per_query=5)

    assert call_count["n"] == 1


def test_semantic_search_ignores_blank_queries(monkeypatch):
    import vector_db

    calls = []

    def fake_query_kb(query, top_k=5, collection_name=None):
        calls.append(query)
        return []

    monkeypatch.setattr(vector_db, "query_kb", fake_query_kb)

    retrieval.semantic_search(["   ", "", "real query"], collection_name="product_briefs", top_k_per_query=5)

    assert calls == ["real query"]


def test_semantic_search_returns_empty_list_on_query_kb_failure(monkeypatch):
    import vector_db

    def failing_query_kb(query, top_k=5, collection_name=None):
        raise RuntimeError("weaviate unreachable")

    monkeypatch.setattr(vector_db, "query_kb", failing_query_kb)

    hits = retrieval.semantic_search(["q1"], collection_name="product_briefs", top_k_per_query=5)

    assert hits == []


def test_finalize_evidence_ranks_by_score_descending(monkeypatch):
    candidates = [
        {"title": "Low relevance", "snippet": "low", "url": "u1"},
        {"title": "High relevance", "snippet": "high", "url": "u2"},
        {"title": "Mid relevance", "snippet": "mid", "url": "u3"},
    ]

    def fake_score_pairs(query, passages):
        return [0.1, 5.0, 1.0]  # matches order: low, high, mid

    monkeypatch.setattr(retrieval, "score_pairs", fake_score_pairs)

    result = retrieval.finalize_evidence(
        "query", candidates, top_n=3, max_chars_per_snippet=100, max_total_chars=1000
    )

    assert [r["title"] for r in result] == ["High relevance", "Mid relevance", "Low relevance"]
    assert [r["citation_id"] for r in result] == ["E1", "E2", "E3"]


def test_finalize_evidence_respects_top_n(monkeypatch):
    candidates = [{"title": f"Doc {i}", "snippet": "text", "url": f"u{i}"} for i in range(5)]
    monkeypatch.setattr(retrieval, "score_pairs", lambda q, p: [float(i) for i in range(len(p))])

    result = retrieval.finalize_evidence(
        "query", candidates, top_n=2, max_chars_per_snippet=100, max_total_chars=10000
    )

    assert len(result) == 2


def test_finalize_evidence_confidence_is_sigmoid_of_score(monkeypatch):
    candidates = [{"title": "Doc", "snippet": "text", "url": "u1"}]
    monkeypatch.setattr(retrieval, "score_pairs", lambda q, p: [0.0])

    result = retrieval.finalize_evidence(
        "query", candidates, top_n=5, max_chars_per_snippet=100, max_total_chars=10000
    )

    assert result[0]["confidence"] == pytest.approx(0.5, abs=1e-6)  # sigmoid(0) == 0.5


def test_finalize_evidence_compresses_long_snippets(monkeypatch):
    long_text = "Sentence one is here. " * 50
    candidates = [{"title": "Doc", "snippet": long_text, "url": "u1"}]
    monkeypatch.setattr(retrieval, "score_pairs", lambda q, p: [1.0])

    result = retrieval.finalize_evidence(
        "query", candidates, top_n=5, max_chars_per_snippet=50, max_total_chars=10000
    )

    assert len(result[0]["snippet"]) <= 51  # small allowance for the ellipsis character


def test_finalize_evidence_enforces_total_char_budget(monkeypatch):
    candidates = [
        {"title": f"Doc {i}", "snippet": "x" * 200, "url": f"u{i}"} for i in range(5)
    ]
    monkeypatch.setattr(retrieval, "score_pairs", lambda q, p: [float(len(p) - i) for i in range(len(p))])

    result = retrieval.finalize_evidence(
        "query", candidates, top_n=5, max_chars_per_snippet=200, max_total_chars=450
    )

    # Budget of 450 chars at 200 chars/snippet should keep roughly 2-3 items, not all 5.
    assert 1 <= len(result) < 5


def test_finalize_evidence_returns_empty_list_for_no_candidates():
    assert retrieval.finalize_evidence("query", [], top_n=5, max_chars_per_snippet=100, max_total_chars=1000) == []


def test_finalize_evidence_falls_back_to_input_order_on_rerank_failure(monkeypatch):
    candidates = [{"title": "Doc 1", "snippet": "a", "url": "u1"}, {"title": "Doc 2", "snippet": "b", "url": "u2"}]

    def failing_score_pairs(query, passages):
        raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(retrieval, "score_pairs", failing_score_pairs)

    result = retrieval.finalize_evidence(
        "query", candidates, top_n=5, max_chars_per_snippet=100, max_total_chars=10000
    )

    assert [r["title"] for r in result] == ["Doc 1", "Doc 2"]

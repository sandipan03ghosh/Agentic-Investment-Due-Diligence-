from __future__ import annotations

import retrieval


# --- merge_by_origin ---


def test_merge_by_origin_tags_internal_only():
    internal = [{"title": "Doc A", "snippet": "text a", "url": "upload://doc1/file.md"}]

    result = retrieval.merge_by_origin(internal, [])

    assert len(result) == 1
    assert result[0]["origin"] == "internal_kb"


def test_merge_by_origin_tags_web_only():
    web = [{"title": "Doc B", "snippet": "text b", "url": "https://example.com/b"}]

    result = retrieval.merge_by_origin([], web)

    assert len(result) == 1
    assert result[0]["origin"] == "web_search"


def test_merge_by_origin_handles_empty_pools():
    assert retrieval.merge_by_origin([], []) == []


def test_merge_by_origin_merges_on_matching_title_and_url_host():
    """Composite match via title + URL host, with deliberately non-overlapping snippets
    so this isolates the host-match branch from the snippet-overlap branch."""
    internal = [{"title": "Product Launch Guide", "snippet": "short internal note", "url": "https://acme.com/guide"}]
    web = [
        {
            "title": "Product Launch Guide",
            "snippet": "completely different unrelated zzz qqq www",
            "url": "https://acme.com/guide?utm=1",
        }
    ]

    result = retrieval.merge_by_origin(internal, web)

    assert len(result) == 1
    assert result[0]["origin"] == "both"


def test_merge_by_origin_merges_on_matching_title_and_snippet_overlap():
    internal = [
        {
            "title": "Quarterly Strategy",
            "snippet": "revenue growth market expansion pricing strategy launch plan",
            "url": "upload://doc1/file.md",
        }
    ]
    web = [
        {
            "title": "Quarterly Strategy",
            "snippet": "revenue growth market expansion pricing strategy overview",
            "url": "https://news.example.com/article",
        }
    ]

    result = retrieval.merge_by_origin(internal, web)

    assert len(result) == 1
    assert result[0]["origin"] == "both"


def test_merge_by_origin_does_not_merge_on_title_alone():
    """Same generic title, unrelated hosts and non-overlapping snippets -> must NOT merge."""
    internal = [{"title": "Overview", "snippet": "alpha beta gamma delta topic one", "url": "upload://doc1/file.md"}]
    web = [{"title": "Overview", "snippet": "totally unrelated content zzz qqq www", "url": "https://other.com/x"}]

    result = retrieval.merge_by_origin(internal, web)

    assert len(result) == 2
    assert {r["origin"] for r in result} == {"internal_kb", "web_search"}


def test_merge_by_origin_only_merges_each_web_item_once():
    internal = [
        {"title": "Same Title", "snippet": "aaa bbb ccc ddd eee", "url": "upload://doc1/file.md"},
        {"title": "Same Title", "snippet": "aaa bbb ccc ddd eee", "url": "upload://doc2/file.md"},
    ]
    web = [{"title": "Same Title", "snippet": "aaa bbb ccc ddd eee", "url": "https://example.com/x"}]

    result = retrieval.merge_by_origin(internal, web)

    both_count = sum(1 for r in result if r["origin"] == "both")
    assert both_count == 1
    assert len(result) == 2  # one merged "both" + one remaining unmatched internal item


def test_merge_by_origin_missing_titles_do_not_crash_or_falsely_merge():
    internal = [{"title": "", "snippet": "no title here", "url": "upload://doc1/file.md"}]
    web = [{"title": "", "snippet": "also no title", "url": "https://example.com/x"}]

    result = retrieval.merge_by_origin(internal, web)

    assert len(result) == 2
    assert {r["origin"] for r in result} == {"internal_kb", "web_search"}


# --- is_kb_sufficient ---


def test_is_kb_sufficient_true_when_enough_high_confidence_items():
    evidence = [{"confidence": 0.9}, {"confidence": 0.8}, {"confidence": 0.7}]
    assert retrieval.is_kb_sufficient(evidence, min_items=3, min_confidence=0.5) is True


def test_is_kb_sufficient_false_when_not_enough_items():
    evidence = [{"confidence": 0.9}, {"confidence": 0.2}]
    assert retrieval.is_kb_sufficient(evidence, min_items=3, min_confidence=0.5) is False


def test_is_kb_sufficient_false_for_empty_evidence():
    assert retrieval.is_kb_sufficient([], min_items=1, min_confidence=0.5) is False


def test_is_kb_sufficient_handles_missing_confidence_key():
    evidence = [{"title": "no confidence field"}]
    assert retrieval.is_kb_sufficient(evidence, min_items=1, min_confidence=0.1) is False


# --- finalize_evidence: citation integrity + origin passthrough ---


def test_finalize_evidence_citation_ids_are_unique_sequential_and_non_empty(monkeypatch):
    candidates = [{"title": f"Doc {i}", "snippet": "text", "url": f"u{i}"} for i in range(5)]
    monkeypatch.setattr(retrieval, "score_pairs", lambda q, p: [float(i) for i in range(len(p))])

    result = retrieval.finalize_evidence(
        "query", candidates, top_n=5, max_chars_per_snippet=100, max_total_chars=10000
    )

    citation_ids = [r["citation_id"] for r in result]
    assert citation_ids == [f"E{i + 1}" for i in range(len(result))]
    assert len(citation_ids) == len(set(citation_ids))
    assert all(cid for cid in citation_ids)


def test_finalize_evidence_preserves_origin_tag(monkeypatch):
    candidates = [{"title": "Doc", "snippet": "text", "url": "u1", "origin": "web_search"}]
    monkeypatch.setattr(retrieval, "score_pairs", lambda q, p: [1.0])

    result = retrieval.finalize_evidence(
        "query", candidates, top_n=5, max_chars_per_snippet=100, max_total_chars=10000
    )

    assert result[0]["origin"] == "web_search"


def test_finalize_evidence_defaults_origin_to_internal_kb_when_missing(monkeypatch):
    candidates = [{"title": "Doc", "snippet": "text", "url": "u1"}]
    monkeypatch.setattr(retrieval, "score_pairs", lambda q, p: [1.0])

    result = retrieval.finalize_evidence(
        "query", candidates, top_n=5, max_chars_per_snippet=100, max_total_chars=10000
    )

    assert result[0]["origin"] == "internal_kb"


def test_finalize_evidence_preserves_both_origin_through_merge_and_finalize(monkeypatch):
    internal = [{"title": "Shared", "snippet": "aaa bbb ccc ddd eee", "url": "upload://doc1/file.md"}]
    web = [{"title": "Shared", "snippet": "aaa bbb ccc ddd eee", "url": "https://example.com/x"}]
    merged = retrieval.merge_by_origin(internal, web)

    monkeypatch.setattr(retrieval, "score_pairs", lambda q, p: [1.0] * len(p))
    result = retrieval.finalize_evidence(
        "query", merged, top_n=5, max_chars_per_snippet=100, max_total_chars=10000
    )

    assert len(result) == 1
    assert result[0]["origin"] == "both"

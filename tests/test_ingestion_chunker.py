from __future__ import annotations

import pytest

from ingestion.chunker import TextChunker


def test_chunker_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=-1)


def test_chunker_rejects_non_positive_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0, chunk_overlap=0)


def test_chunker_splits_long_text_into_multiple_chunks():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    text = "This is a sentence about product launches. " * 50
    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(len(c) <= 200 + 20 for c in chunks)  # splitter may slightly exceed at hard boundaries
    assert all(c.strip() for c in chunks)


def test_chunker_returns_empty_list_for_blank_text():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("") == []
    assert chunker.split("   \n\n  ") == []


def test_chunker_returns_single_chunk_for_short_text():
    chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
    text = "A short launch brief."
    chunks = chunker.split(text)
    assert chunks == [text]

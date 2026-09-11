import json
from pathlib import Path

from scripts import ingest_documents


def test_chunking():
    text = "word " * 1000
    chunks = ingest_documents.chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 0


def test_build_docs_from_files(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# Title\n" + ("hello world\n" * 50))
    docs = ingest_documents.build_docs_from_files([p])
    assert isinstance(docs, list)
    assert docs[0]["title"] == "a"

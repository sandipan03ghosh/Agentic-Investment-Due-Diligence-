"""Production-style ingest: chunk files, embed, and upsert to Weaviate.

Usage:
  python -m product_agent.scripts.ingest_documents --dir ./sample_data --collection due_diligence_documents --recreate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import vector_db


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def build_docs_from_files(paths: List[Path]) -> List[Dict]:
    docs = []
    id_counter = 1
    for p in sorted(paths):
        text = p.read_text(encoding="utf-8", errors="replace")
        title = p.stem
        for c in chunk_text(text):
            docs.append({"id": id_counter, "title": title, "text": c, "url": str(p)})
            id_counter += 1
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Directory with files to ingest")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.dir)
    paths = [p for p in root.rglob("*.md")]
    if not paths:
        print("No markdown files found to ingest.")
        return

    docs = build_docs_from_files(paths)

    if args.dry_run:
        print(json.dumps({"count": len(docs), "sample": docs[:2]}, indent=2))
        return

    collection = args.collection or None
    vector_db.index_documents(docs, collection_name=collection or vector_db._COLLECTION, recreate=args.recreate)
    print(f"Indexed {len(docs)} chunks into collection {collection or vector_db._COLLECTION}")


if __name__ == "__main__":
    main()

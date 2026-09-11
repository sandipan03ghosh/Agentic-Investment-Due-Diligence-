"""Ingest markdown files in the repo root into the vector DB collection.

Usage:
    python -m product_agent.scripts.ingest_md --dir . --collection due_diligence_documents

This script requires `weaviate-client` and `sentence-transformers` to be installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import vector_db


def find_md_files(root: Path) -> List[Path]:
    return [p for p in root.glob("*.md") if p.is_file()]


def extract_title(md_text: str, fallback: str) -> str:
    for line in md_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def build_docs(md_paths: List[Path]) -> List[Dict]:
    docs = []
    for i, p in enumerate(sorted(md_paths), start=1):
        text = p.read_text(encoding="utf-8", errors="replace")
        title = extract_title(text, p.stem)
        snippet = "\n".join(text.splitlines()[:6])[:400]
        docs.append({"id": i, "title": title, "text": text, "url": str(p), "snippet": snippet})
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Directory containing markdown files")
    parser.add_argument("--collection", default=None, help="Weaviate collection name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.dir)
    md_files = find_md_files(root)
    if not md_files:
        print("No markdown files found.")
        return

    docs = build_docs(md_files)

    if args.dry_run:
        print(json.dumps({"count": len(docs), "sample": docs[:2]}, default=str, indent=2))
        return

    collection = args.collection or None
    if collection:
        vector_db.index_documents(docs, collection_name=collection)
    else:
        vector_db.index_documents(docs)

    print(f"Indexed {len(docs)} documents into Weaviate collection: {collection or vector_db._COLLECTION}")


if __name__ == "__main__":
    main()

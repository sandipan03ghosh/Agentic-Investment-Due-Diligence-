"""Evaluate retrieval quality over a small set of sample queries.

Example:
  python -m product_agent.scripts.evaluate_retrieval --cases sample_data/eval_cases.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import vector_db


def load_cases(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_case(case: Dict[str, Any], hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected = [s.lower() for s in case.get("expected_keywords", [])]
    top_text = " ".join(
        f"{h.get('title','')} {h.get('snippet','')} {h.get('url','')}".lower() for h in hits
    )
    found = [kw for kw in expected if kw in top_text]
    return {
        "query": case.get("query"),
        "expected_keywords": expected,
        "found_keywords": found,
        "precision_proxy": round(len(found) / max(len(expected), 1), 3),
        "top_hits": hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="sample_data/eval_cases.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    collection = args.collection or vector_db._COLLECTION
    results = []
    for case in cases:
        hits = vector_db.query_kb(case["query"], top_k=args.top_k, collection_name=collection)
        results.append(score_case(case, hits))

    summary = {
        "collection": collection,
        "cases": len(results),
        "avg_precision_proxy": round(sum(r["precision_proxy"] for r in results) / max(len(results), 1), 3),
        "results": results,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

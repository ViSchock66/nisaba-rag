"""Evaluator for Nisaba RAG: recall@k and MRR against a golden set.

A golden set pairs a query with the path fragments that count as a correct
answer. Edit `golden_set.json` to describe your own corpus before reading
anything into the numbers — the shipped example is small and generic.

Usage:
    python eval/eval_rag.py --mode dense
    python eval/eval_rag.py --mode hybrid --rerank
    python eval/eval_rag.py --from-results results.json

`--from-results` scores a JSON file of already-obtained results — useful when
the searches were run through a live MCP server rather than in-process:

    {"queries": [{"query": "...", "sources": ["/path/hit.md", ...]}]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from nisaba_rag.rag import RagStore, embed, load_config  # noqa: E402
GOLDEN_PATH = os.path.join(HERE, "golden_set.json")


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)["queries"]


def recall_at_k(hit_sources: list[str], expected: list[str], k: int) -> bool:
    """True when one of the top-k hits comes from an expected file."""
    return any(
        any(fragment in source for fragment in expected)
        for source in hit_sources[:k]
    )


def reciprocal_rank(hit_sources: list[str], expected: list[str]) -> float:
    """1 / rank of the first correct hit, or 0 when there is none."""
    for rank, source in enumerate(hit_sources, start=1):
        if any(fragment in source for fragment in expected):
            return 1.0 / rank
    return 0.0


def _summarize(results: list[dict], mode: str, rerank: bool | None) -> dict:
    n = len(results)
    return {
        "mode": mode,
        "rerank": rerank,
        "n": n,
        "recall@5": sum(1 for r in results if r["r@5"]) / n if n else 0.0,
        "mrr": sum(r["mrr"] for r in results) / n if n else 0.0,
        "details": results,
    }


def run(mode: str, rerank: bool, top_n: int, k: int = 5) -> dict:
    """Evaluate in-process against the local index."""
    cfg = load_config()
    store = RagStore(cfg)
    try:
        results = []
        for item in load_golden():
            query = item["query"]

            if mode == "hybrid":
                from nisaba_rag.hybrid import hybrid_search

                hits = hybrid_search(store, query, cfg, top_n=top_n, rerank=rerank)
            else:
                hits = store.search(embed(query, cfg), limit=top_n)

            sources = [hit.get("source") or "" for hit in hits]
            results.append({
                "query": query,
                "r@5": recall_at_k(sources, item["expected"], k),
                "mrr": reciprocal_rank(sources, item["expected"]),
                "top_sources": sources[:k],
            })
    finally:
        store.close()

    return _summarize(results, mode, rerank)


def run_from_results(path: str, k: int = 5) -> dict:
    """Evaluate results captured from a live server (no database access).

    Latencies include serialization and reranking, so these numbers are not
    directly comparable with the in-process path.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    golden = {item["query"]: item["expected"] for item in load_golden()}
    results = []
    for entry in data.get("queries", []):
        expected = golden.get(entry["query"])
        if expected is None:
            continue
        sources = entry.get("sources", [])
        results.append({
            "query": entry["query"],
            "r@5": recall_at_k(sources, expected, k),
            "mrr": reciprocal_rank(sources, expected),
            "top_sources": sources[:k],
        })

    return _summarize(results, "replay", None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Nisaba RAG retrieval quality.")
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--k", type=int, default=5, help="Cutoff for recall@k.")
    parser.add_argument(
        "--from-results",
        default=None,
        help='Replay a JSON file: {"queries": [{"query", "sources": [...]}]}.',
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    args = parser.parse_args()

    report = run_from_results(args.from_results, args.k) if args.from_results else run(
        args.mode, args.rerank, args.top_n, args.k
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(
        f"mode={report['mode']} rerank={report['rerank']} n={report['n']} "
        f"recall@{args.k}={report['recall@5']:.3f} mrr={report['mrr']:.3f}"
    )
    for detail in report["details"]:
        flag = "OK  " if detail["r@5"] else "MISS"
        print(f"  [{flag}] rr={detail['mrr']:.2f} {detail['query']!r}")
        for source in detail["top_sources"][:3]:
            print(f"          - {os.path.basename(source)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

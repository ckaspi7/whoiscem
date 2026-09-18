"""Retrieval ablation: which parts of the pipeline actually earn their place.

Routing is excluded on purpose. This measures retrieval alone, against the
golden set's verbatim references, so a variant is not punished for a question
the router sent elsewhere.

Cheap to run — embeddings and a local cross-encoder, no judge calls.

    python eval/ablate_retrieval.py --output eval/results/ablation-retrieval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from tools.resume_tool import _init_retrieval  # noqa: E402

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
CANDIDATE_POOL = 40


def build_variants(store, bm25, reranker):
    def fused(query):
        return reciprocal_rank_fusion(store.dense_search(query, 20), bm25.search(query, 20))

    return {
        "dense_only@3": lambda q: store.dense_search(q, 3),
        "dense_only@5": lambda q: store.dense_search(q, 5),
        "sparse_only@3": lambda q: bm25.search(q, 3),
        "sparse_only@5": lambda q: bm25.search(q, 5),
        "rrf@3": lambda q: fused(q)[:3],
        "rrf@5": lambda q: fused(q)[:5],
        "rrf@8": lambda q: fused(q)[:8],
        "rrf_rerank@3": lambda q: reranker.rerank(q, fused(q)[:CANDIDATE_POOL], top_n=3),
        "rrf_rerank@5": lambda q: reranker.rerank(q, fused(q)[:CANDIDATE_POOL], top_n=5),
        "rrf_rerank@8": lambda q: reranker.rerank(q, fused(q)[:CANDIDATE_POOL], top_n=8),
        "everything": lambda q: reranker.rerank(q, fused(q), top_n=len(fused(q))),
    }


def evaluate_variant(name, pick, questions):
    hits = 0
    reciprocal = 0.0
    context_chars = 0
    started = time.perf_counter()

    for item in questions:
        chunks = pick(item["question"])
        context_chars += sum(len(c.text) for c in chunks)
        for rank, chunk in enumerate(chunks, start=1):
            if item["reference_snippet"] in chunk.text:
                hits += 1
                reciprocal += 1 / rank
                break

    total = len(questions)
    elapsed = time.perf_counter() - started
    return {
        "variant": name,
        "recall_at_k": round(hits / total, 4),
        "mrr": round(reciprocal / total, 4),
        "mean_context_chars": round(context_chars / total),
        "mean_latency_ms": round(elapsed / total * 1000, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    questions = [g for g in golden if g.get("reference_snippet")]

    store, bm25, reranker = _init_retrieval()
    variants = build_variants(store, bm25, reranker)

    print(f"Retrieval ablation over {len(questions)} referenced questions (routing excluded)\n")
    header = f"{'variant':<18}{'recall@k':>10}{'MRR':>8}{'ctx chars':>11}{'ms/query':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, pick in variants.items():
        row = evaluate_variant(name, pick, questions)
        rows.append(row)
        print(
            f"{row['variant']:<18}{row['recall_at_k']:>10.1%}{row['mrr']:>8.4f}"
            f"{row['mean_context_chars']:>11}{row['mean_latency_ms']:>10.1f}"
        )

    best = max(rows, key=lambda r: (r["recall_at_k"], -r["mean_context_chars"]))
    print(f"\nHighest recall: {best['variant']} at {best['recall_at_k']:.1%}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "run_at": datetime.now(UTC).isoformat(),
                    "questions": len(questions),
                    "note": "Retrieval only; routing excluded so a variant is not "
                    "penalised for a question the router misdirected.",
                    "variants": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

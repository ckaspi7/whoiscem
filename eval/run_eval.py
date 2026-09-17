"""
Evaluation harness for the whoiscem assistant.

Runs the golden set through the *shipping* graph — router, handler, production
system prompt and faithfulness guard — and scores the results with RAGAS.
Earlier versions called the tools directly using the golden set's own
`required_tool`, which meant the eval could not see a routing mistake, a prompt
change, or the guardrail: it measured a pipeline that is not the one deployed.

Usage:
    python eval/run_eval.py --output eval/results/v1_baseline.json
    python eval/run_eval.py --limit 5            # smoke run
    python eval/run_eval.py --no-ragas           # routing only, no judge calls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_settings  # noqa: E402  (after sys.path setup)
from observability import setup_tracing  # noqa: E402

THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_recall": 0.80,
    "context_precision": 0.70,
}

# Which handler each golden `required_tool` should route to.
TOOL_TO_ROUTE = {
    "get_resume_info": "resume",
    "get_personal_info": "personal",
    "get_music_taste": "spotify",
    "get_linkedin_info": "linkedin",
    None: "conversation",
}

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).parent.parent,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _corpus_fingerprint() -> str:
    """Hash the indexed resume.

    Retrieval scores are only comparable across runs that indexed the same text,
    and the resume is a living document. Without this, editing it silently
    invalidates every earlier number in the trend line.
    """
    path = Path(load_settings().resume_path)
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def run_question(graph, item: dict) -> dict:
    """Send one golden question through the real graph."""
    state = graph.invoke(
        {
            "messages": [{"role": "human", "content": item["question"]}],
            "next_step": "",
            "route": "",
            "tool_result": "",
            "context_used": "",
            "context_chunks": [],
            "node_latencies": {},
        }
    )

    raw = state["messages"][-1]["content"]
    answer = raw if isinstance(raw, str) else "".join(c.content for c in raw)

    expected_route = TOOL_TO_ROUTE.get(item.get("required_tool"), "conversation")
    actual_route = state.get("route", "")

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": answer,
        "contexts": state.get("context_chunks") or [],
        "ground_truth": item["ground_truth"],
        "expected_route": expected_route,
        "actual_route": actual_route,
        "route_correct": expected_route == actual_route,
        "latencies": state.get("node_latencies", {}),
    }


def score_routing(rows: list[dict]) -> dict:
    """Routing accuracy overall and per category.

    Measured nowhere before this, despite being the component most likely to
    fail: a misroute lands on handle_conversation, the one path with no
    grounding and no faithfulness check.
    """
    total = len(rows)
    correct = sum(r["route_correct"] for r in rows)
    by_category: dict[str, dict] = {}
    confusions: dict[str, int] = {}

    for row in rows:
        cat = by_category.setdefault(row["category"], {"total": 0, "correct": 0})
        cat["total"] += 1
        cat["correct"] += int(row["route_correct"])
        if not row["route_correct"]:
            key = f"{row['expected_route']} -> {row['actual_route'] or 'none'}"
            confusions[key] = confusions.get(key, 0) + 1

    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "by_category": {
            k: {**v, "accuracy": round(v["correct"] / v["total"], 4)} for k, v in by_category.items()
        },
        "confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])),
    }


def score_ragas(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Score with RAGAS, returning aggregates and per-question values.

    RAGAS 0.2 renamed every column (question -> user_input, answer -> response,
    contexts -> retrieved_contexts, ground_truth -> reference) and returns an
    EvaluationResult rather than a dict. Per-question values come from
    to_pandas(), so a regression is debuggable instead of being one moved float.
    """
    from ragas import EvaluationDataset, SingleTurnSample
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"] or [""],
            reference=r["ground_truth"],
        )
        for r in rows
    ]
    result = ragas_evaluate(
        EvaluationDataset(samples=samples),
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    frame = result.to_pandas()
    metric_columns = [c for c in frame.columns if c in THRESHOLDS]

    aggregates = {}
    for column in metric_columns:
        series = frame[column].dropna()
        aggregates[column] = round(float(series.mean()), 4) if len(series) else None

    per_question = []
    for row, (_, record) in zip(rows, frame.iterrows(), strict=True):
        per_question.append(
            {
                col: (None if record[col] != record[col] else round(float(record[col]), 4))
                for col in metric_columns
            }
            | {"id": row["id"]}
        )

    return aggregates, per_question


def evaluate(output_path: str | None = None, limit: int | None = None, use_ragas: bool = True) -> dict:
    setup_tracing()

    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden = json.load(f)
    if limit:
        golden = golden[:limit]

    from chatbot import create_assistant

    graph = create_assistant()

    print(f"Running {len(golden)} questions through the graph...")
    rows = []
    for i, item in enumerate(golden, start=1):
        row = run_question(graph, item)
        flag = "ok " if row["route_correct"] else "MIS"
        print(f"  [{i}/{len(golden)}] {flag} {item['id']}: {item['question'][:52]}")
        if not row["route_correct"]:
            print(f"        routed {row['expected_route']} -> {row['actual_route'] or 'none'}")
        rows.append(row)

    routing = score_routing(rows)
    scores: dict = {}
    per_question: list[dict] = []
    if use_ragas:
        print("\nScoring with RAGAS...")
        scores, per_question = score_ragas(rows)

    by_id = {p["id"]: p for p in per_question}
    summary = {
        "run_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "corpus_sha": _corpus_fingerprint(),
        "num_questions": len(golden),
        "routing": routing,
        "scores": scores,
        "thresholds_met": {k: (scores.get(k) or 0) >= v for k, v in THRESHOLDS.items()} if scores else {},
        "questions": [
            {
                "id": r["id"],
                "category": r["category"],
                "question": r["question"],
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
                "contexts": r["contexts"],
                "expected_route": r["expected_route"],
                "actual_route": r["actual_route"],
                "route_correct": r["route_correct"],
                "latencies": r["latencies"],
                "metrics": {k: v for k, v in by_id.get(r["id"], {}).items() if k != "id"},
            }
            for r in rows
        ],
    }

    print("\n=== Routing ===")
    print(f"  accuracy: {routing['accuracy']:.1%} ({routing['correct']}/{routing['total']})")
    for cat, stats in sorted(routing["by_category"].items()):
        print(f"    {cat:<12} {stats['correct']}/{stats['total']}  ({stats['accuracy']:.0%})")
    for confusion, count in routing["confusions"].items():
        print(f"    misrouted x{count}: {confusion}")

    if scores:
        print("\n=== RAGAS ===")
        for metric, score in scores.items():
            threshold = THRESHOLDS[metric]
            status = "PASS" if (score or 0) >= threshold else "FAIL"
            shown = "n/a" if score is None else f"{score:.4f}"
            print(f"  {status}  {metric}: {shown} (threshold {threshold})")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nResults written to {out}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Path to write JSON results")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS scoring; report routing only")
    parser.add_argument(
        "--fail-under-threshold",
        action="store_true",
        help="Exit non-zero when a RAGAS threshold is missed (for CI gating)",
    )
    args = parser.parse_args()

    result = evaluate(args.output, limit=args.limit, use_ragas=not args.no_ragas)

    if args.fail_under_threshold and result["thresholds_met"]:
        if not all(result["thresholds_met"].values()):
            missed = [k for k, ok in result["thresholds_met"].items() if not ok]
            print(f"\nBelow threshold: {', '.join(missed)}")
            sys.exit(1)

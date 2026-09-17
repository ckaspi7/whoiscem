"""
RAGAS evaluation harness for the howtocem retrieval pipeline.
Run against baseline (before hybrid search) and v2 (after) to measure improvement.

Usage:
    python eval/run_eval.py --output eval/results/v2_hybrid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_recall": 0.80,
    "context_precision": 0.70,
}

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"


def run_query(question: str, required_tool: str) -> tuple[str, str]:
    """Run a single query through the full retrieval + generation pipeline."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from tools.linkedin_tool import get_linkedin_info
    from tools.personal_tool import get_personal_info
    from tools.resume_tool import get_resume_info
    from tools.spotify_tool import get_music_taste

    tool_map = {
        "get_resume_info": get_resume_info,
        "get_personal_info": get_personal_info,
        "get_music_taste": get_music_taste,
        "get_linkedin_info": get_linkedin_info,
    }

    context = tool_map[required_tool].invoke(question if required_tool == "get_resume_info" else "")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(
            content="Answer the following question about Cem Kaspi using only the provided context."
        ),
        SystemMessage(content=f"Context:\n{context}"),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content, context


def evaluate(output_path: str | None = None) -> dict:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    print(f"Running evaluation on {len(golden)} questions...")
    rows = []
    for i, item in enumerate(golden):
        print(f"  [{i+1}/{len(golden)}] {item['id']}: {item['question'][:60]}...")
        answer, context = run_query(item["question"], item["required_tool"])
        rows.append({
            "question": item["question"],
            "answer": answer,
            "contexts": [context],
            "ground_truth": item["ground_truth"],
        })

    dataset = Dataset.from_list(rows)
    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_recall": float(result["context_recall"]),
        "context_precision": float(result["context_precision"]),
    }

    summary = {
        "run_at": datetime.now(UTC).isoformat(),
        "num_questions": len(golden),
        "scores": scores,
        "thresholds_met": {k: scores[k] >= v for k, v in THRESHOLDS.items()},
    }

    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in scores.items():
        threshold = THRESHOLDS[metric]
        status = "✓" if score >= threshold else "✗"
        print(f"  {status} {metric}: {score:.4f} (threshold: {threshold})")

    all_passed = all(summary["thresholds_met"].values())
    print(f"\n{'All thresholds met.' if all_passed else 'WARNING: Some thresholds not met.'}")

    if output_path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to {output_path}")

    if not all_passed:
        sys.exit(1)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Path to write JSON results")
    args = parser.parse_args()
    evaluate(args.output)

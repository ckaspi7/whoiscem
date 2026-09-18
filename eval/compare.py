"""Compare an evaluation run against the last committed one.

A committed baseline is only useful if something reads it. This is the gate: it
reports the delta on every metric present in both runs, and fails when one has
regressed past tolerance.

    python eval/compare.py --current eval/results/pr.json
    python eval/compare.py --current a.json --baseline b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# How far a metric may fall before it counts as a regression. Small movement is
# expected: the graph calls a model, and nothing here is fully deterministic.
TOLERANCE = {
    "routing.accuracy": 0.05,
    "retrieval.recall_at_k": 0.05,
    "retrieval.mrr": 0.05,
    "refusals.refusal_rate": 0.05,
    "scores.faithfulness": 0.07,
    "scores.answer_relevancy": 0.07,
    "scores.context_recall": 0.07,
    "scores.context_precision": 0.07,
}


def _dig(payload: dict, dotted: str):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, (int, float)) else None


def latest_baseline(exclude: Path | None = None) -> Path | None:
    """Most recent committed result, by the run_at it records."""
    candidates = []
    for path in RESULTS_DIR.glob("*.json"):
        if exclude and path.resolve() == exclude.resolve():
            continue
        try:
            candidates.append((json.loads(path.read_text(encoding="utf-8"))["run_at"], path))
        except (OSError, ValueError, KeyError):
            continue
    if not candidates:
        return None
    return max(candidates)[1]


def compare(current_path: Path, baseline_path: Path) -> int:
    current = json.loads(current_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    print(f"current  {current_path.name}  sha={current.get('git_sha')} corpus={current.get('corpus_sha')}")
    print(f"baseline {baseline_path.name}  sha={baseline.get('git_sha')} corpus={baseline.get('corpus_sha')}")

    if current.get("corpus_sha") != baseline.get("corpus_sha"):
        print("\nNOTE: the indexed corpus differs between these runs. Retrieval numbers")
        print("      are not strictly comparable; treat movement as uninterpreted.")

    regressions = []
    print(f"\n{'metric':<32} {'baseline':>9} {'current':>9} {'delta':>9}")
    for metric, tolerance in TOLERANCE.items():
        before, after = _dig(baseline, metric), _dig(current, metric)
        if before is None or after is None:
            continue
        delta = after - before
        flag = ""
        if delta < -tolerance:
            flag = "  REGRESSION"
            regressions.append((metric, before, after))
        elif delta > tolerance:
            flag = "  improved"
        print(f"{metric:<32} {before:>9.4f} {after:>9.4f} {delta:>+9.4f}{flag}")

    if regressions:
        print("\nRegressions beyond tolerance:")
        for metric, before, after in regressions:
            print(f"  {metric}: {before:.4f} -> {after:.4f}")
        return 1

    print("\nNo regression beyond tolerance.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, help="Result file for this run")
    parser.add_argument("--baseline", default=None, help="Defaults to the newest committed result")
    args = parser.parse_args()

    current = Path(args.current)
    if not current.exists():
        print(f"No current result at {current}")
        return 2

    baseline = Path(args.baseline) if args.baseline else latest_baseline(exclude=current)
    if baseline is None:
        print("No committed baseline to compare against — nothing to gate. Passing.")
        return 0

    return compare(current, baseline)


if __name__ == "__main__":
    sys.exit(main())

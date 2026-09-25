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


# Metrics whose value depends on chunk granularity, not on answer quality.
RANK_SENSITIVE = {"retrieval.mrr", "scores.context_precision"}


def _tool_health_failures(current: dict) -> list[str]:
    """A tool call that failed outright, checked directly rather than as a delta.

    There is no baseline error rate worth tolerating: a tool failure means the
    judge scored an empty context, not a real answer, so every quality metric
    in that run is unreliable regardless of what it reads. This is also the
    check that would have caught the incident that motivated it — a fresh
    Qdrant container racing the first query degraded routing, recall and
    faithfulness together, and the aggregate drop alone did not say why.
    """
    health = current.get("tool_health") or {}
    errors = health.get("errors", 0)
    if not errors:
        return []
    failed = ", ".join(health.get("failed_ids", [])) or "unknown"
    return [f"{errors} tool call(s) failed outright ({failed}) — every metric below is unreliable"]


def _dig(payload: dict, dotted: str):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, (int, float)) else None


def latest_baseline(exclude: Path | None = None) -> Path | None:
    """Most recent committed *full* result, by the run_at it records."""
    candidates = []
    for path in RESULTS_DIR.glob("*.json"):
        if exclude and path.resolve() == exclude.resolve():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # Only full evaluation runs are baselines. eval/results also holds
        # ablation studies (no routing) and --no-ragas runs (routing present,
        # but no scores) — e.g. v13-trajectory.json, committed for its
        # trajectory numbers, not as a RAGAS reference. A --no-ragas run
        # passing this check would silently become the picked baseline the
        # moment it is the newest file, and every RAGAS threshold would stop
        # being gated until a newer full run replaced it — the same failure
        # shape as the incident that motivated this function's own docstring.
        if "routing" not in payload or "run_at" not in payload or not payload.get("scores"):
            continue
        candidates.append((payload["run_at"], path))
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

    # Rank-position metrics depend on how the corpus was cut. Three chunks of
    # 2,000 characters put the answer at rank 1 almost by default; sixteen
    # chunks of 500 put the same answer at rank 3 while the model reads exactly
    # the same text. Reported, but not gated, when the chunker changed.
    # Only meaningful when the baseline recorded them. A baseline predating the
    # field would otherwise report every source as changed, on every run, which
    # is how a useful warning becomes one people scroll past.
    baseline_sources = baseline.get("data_sha") or {}
    changed_sources = [
        name
        for name, digest in (current.get("data_sha") or {}).items()
        if name in baseline_sources and digest != baseline_sources[name]
    ]
    if changed_sources:
        print(f"\nNOTE: these data sources changed since the baseline: {', '.join(changed_sources)}.")
        print("      Movement on the routes they serve is explained by the data, not the code.")

    before_cfg = baseline.get("retrieval_config") or {}
    after_cfg = current.get("retrieval_config") or {}
    if before_cfg and before_cfg != after_cfg:
        print(f"\nNOTE: retrieval configuration changed: {before_cfg} -> {after_cfg}.")

    before_mode = baseline.get("agent_mode")
    after_mode = current.get("agent_mode")
    mode_changed = bool(before_mode) and before_mode != after_mode
    if mode_changed:
        print(f"\nNOTE: agent_mode changed: {before_mode} -> {after_mode}. This is a different graph")
        print("      architecture, not a code regression on the same one — read deltas as a comparison.")

    before_model = baseline.get("chat_model")
    after_model = current.get("chat_model")
    if before_model and before_model != after_model:
        print(f"\nNOTE: chat_model changed: {before_model} -> {after_model}.")
        print("      Unlike agent_mode, this does not change what a chunk is — rank-sensitive")
        print("      metrics are still gated normally.")

    rechunked = current.get("chunker") != baseline.get("chunker")
    if rechunked:
        print(f"\nNOTE: chunker changed ({baseline.get('chunker')} -> {current.get('chunker')}).")

    # Both cases change what a "chunk" even is: rechunking obviously, and
    # switching agent_mode because tool_calling's context_chunks is one
    # pre-merged block per successful tool call (get_resume_info_result
    # already joins everything with format_chunks), not the classifier's
    # several individually-ranked chunks. A single block is trivially rank 1
    # if it is found at all, so MRR and context_precision moving is an
    # artifact of granularity, not evidence retrieval got better or worse.
    granularity_changed = rechunked or mode_changed
    if granularity_changed:
        print("      Rank-sensitive metrics are reported but not gated, in either direction.")

    hard_failures = _tool_health_failures(current)
    if hard_failures:
        print("\nHard failures (checked directly, not as a delta against the baseline):")
        for message in hard_failures:
            print(f"  {message}")

    regressions = []
    print(f"\n{'metric':<32} {'baseline':>9} {'current':>9} {'delta':>9}")
    for metric, tolerance in TOLERANCE.items():
        before, after = _dig(baseline, metric), _dig(current, metric)
        if before is None or after is None:
            continue
        delta = after - before
        rank_sensitive_and_uninterpretable = granularity_changed and metric in RANK_SENSITIVE
        flag = ""
        if delta < -tolerance and rank_sensitive_and_uninterpretable:
            flag = "  down (not gated: granularity changed)"
        elif delta < -tolerance:
            flag = "  REGRESSION"
            regressions.append((metric, before, after))
        elif delta > tolerance and rank_sensitive_and_uninterpretable:
            flag = "  up (not gated: granularity changed)"
        elif delta > tolerance:
            flag = "  improved"
        print(f"{metric:<32} {before:>9.4f} {after:>9.4f} {delta:>+9.4f}{flag}")

    if regressions:
        print("\nRegressions beyond tolerance:")
        for metric, before, after in regressions:
            print(f"  {metric}: {before:.4f} -> {after:.4f}")

    if regressions or hard_failures:
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

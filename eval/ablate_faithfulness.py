"""Faithfulness-guardrail ablation: the current gpt-4o-mini judge vs a local
HHEM-2.1-Open classifier, on eval/faithfulness_test_set.json (build it first
with build_faithfulness_test_set.py if missing or stale).

Measures accuracy (against the test set's faithful/unfaithful labels) and
latency for both, plus a best-threshold search for HHEM since it returns a
continuous 0-1 score rather than the judge's discrete 1-5.

    python eval/ablate_faithfulness.py --output eval/results/ablation-faithfulness.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrails.faithfulness_check import score_faithfulness  # noqa: E402

TEST_SET_PATH = Path(__file__).parent / "faithfulness_test_set.json"


def _load_hhem():
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model", trust_remote_code=True
    )


def _best_threshold(rows: list[dict]) -> tuple[float, float]:
    best_t, best_acc = 0.5, -1.0
    for i in range(1, 100):
        t = i / 100
        correct = sum((r["hhem_score"] >= t) == r["faithful"] for r in rows)
        acc = correct / len(rows)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not TEST_SET_PATH.exists():
        print(f"{TEST_SET_PATH} not found — run build_faithfulness_test_set.py first")
        return 1
    pairs = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    routes = sorted({p["route"] for p in pairs})
    print(f"{len(pairs)} pairs ({sum(p['faithful'] for p in pairs)} faithful) across routes: {routes}")
    if "resume" not in routes:
        print(
            "NOTE: no resume-route pairs — this machine's network cannot reach Qdrant Cloud "
            "(see whoiscem-portfolio-plan memory). Re-run once that's not the case."
        )

    print("\nLoading HHEM-2.1-Open...")
    t0 = time.perf_counter()
    hhem = _load_hhem()
    print(f"HHEM load time: {time.perf_counter() - t0:.1f}s\n")

    rows = []
    for p in pairs:
        t0 = time.perf_counter()
        hhem_score = float(hhem.predict([(p["context"], p["answer"])])[0])
        hhem_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        judge_score = score_faithfulness(p["answer"], p["context"])
        judge_ms = (time.perf_counter() - t0) * 1000

        rows.append(
            {
                "id": p["id"],
                "route": p["route"],
                "faithful": p["faithful"],
                "hhem_score": hhem_score,
                "hhem_ms": hhem_ms,
                "judge_score": judge_score,
                "judge_ms": judge_ms,
            }
        )
        judge_correct = (judge_score is not None and judge_score >= 4) == p["faithful"]
        flag = "" if judge_correct else "<-- judge miss"
        print(f"  {p['id']:<20} hhem={hhem_score:.3f} judge={judge_score} {flag}")

    best_t, best_acc = _best_threshold(rows)
    judge_correct_n = sum(
        (r["judge_score"] is not None and r["judge_score"] >= 4) == r["faithful"] for r in rows
    )
    judge_acc = judge_correct_n / len(rows)
    hhem_mean_ms = sum(r["hhem_ms"] for r in rows) / len(rows)
    judge_mean_ms = sum(r["judge_ms"] for r in rows) / len(rows)

    misses = [r for r in rows if (r["judge_score"] is not None and r["judge_score"] >= 4) != r["faithful"]]

    print(f"\n{'method':<20}{'accuracy':>10}{'mean ms':>10}")
    print("-" * 40)
    print(f"{'hhem (best t=' + f'{best_t:.2f})':<20}{best_acc:>10.1%}{hhem_mean_ms:>10.1f}")
    print(f"{'judge (t>=4)':<20}{judge_acc:>10.1%}{judge_mean_ms:>10.1f}")
    print(f"\njudge missed {len(misses)}/{len(rows)}: {[m['id'] for m in misses]}")
    print(f"speedup: {judge_mean_ms / hhem_mean_ms:.1f}x")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "run_at": datetime.now(UTC).isoformat(),
                    "pairs": len(rows),
                    "routes_covered": routes,
                    "hhem": {
                        "best_threshold": best_t,
                        "accuracy": best_acc,
                        "mean_latency_ms": round(hhem_mean_ms, 1),
                    },
                    "judge": {"accuracy": judge_acc, "mean_latency_ms": round(judge_mean_ms, 1)},
                    "judge_misses": [m["id"] for m in misses],
                    "rows": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Does how much context HHEM reads change what it catches?

Scores every pair in eval/faithfulness_test_set.json under each context mode of
guardrails.faithfulness_check.hhem_score and reports, per route group:

- faithful pairs: pass clean (score at or above the pass line), warn, or refused
- unfaithful pairs: flagged (below the pass line) or let through clean
- accuracy at the pass line, and mean scoring time

The resume route is reported apart from the others because it is the only one
with a long context (the whole resume, about 7.6 KB, against 333-1,346
characters for linkedin, personal and spotify). The original 32 pairs contained
none from it, so their accuracy figure said nothing about that length; the
other routes' contexts are shorter than every window tried here, so for them
every mode reduces to "full" and this doubles as a no-regression check.

Scoring time is each strategy's own cost: the sum of the first-computation time
of every model call it needs. Calls are cached across strategies, so a window
or a full-context score shared by two strategies is computed once.

    python eval/ablate_judge_context.py --output eval/results/ablation-judge-context.json
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

import guardrails.faithfulness_check as fc  # noqa: E402

TEST_SET_PATH = Path(__file__).parent / "faithfulness_test_set.json"

STRATEGIES: list[tuple[str, dict]] = [
    ("full", {"mode": "full"}),
    ("focus@2000", {"mode": "focus", "focus_chars": 2000}),
    ("focus_then_full@1500", {"mode": "focus_then_full", "focus_chars": 1500}),
    ("focus_then_full@2000", {"mode": "focus_then_full", "focus_chars": 2000}),
    ("focus_then_full@3000", {"mode": "focus_then_full", "focus_chars": 3000}),
]


class _CachedModel:
    """Wraps the HHEM model so identical (context, answer) calls are computed once."""

    def __init__(self, model) -> None:
        self._model = model
        self.cache: dict[tuple[str, str], tuple[float, float]] = {}
        self.used: list[tuple[str, str]] = []

    def predict(self, pairs):
        scores = []
        for premise, hypothesis in pairs:
            key = (premise, hypothesis)
            if key not in self.cache:
                started = time.perf_counter()
                score = float(self._model.predict([key])[0])
                self.cache[key] = (score, time.perf_counter() - started)
            self.used.append(key)
            scores.append(self.cache[key][0])
        return scores

    def seconds_for_used(self) -> float:
        return sum(self.cache[key][1] for key in self.used)


def _share(items: list[dict], predicate) -> float | None:
    return round(sum(predicate(r) for r in items) / len(items), 3) if items else None


def _summarize(rows: list[dict]) -> dict:
    pass_line, refuse_line = fc._HHEM_PASS_ABOVE, fc._HHEM_REFUSE_BELOW
    faithful = [r for r in rows if r["faithful"]]
    unfaithful = [r for r in rows if not r["faithful"]]
    return {
        "n_faithful": len(faithful),
        "n_unfaithful": len(unfaithful),
        "faithful_clean": _share(faithful, lambda r: r["score"] >= pass_line),
        "faithful_warned": _share(faithful, lambda r: refuse_line <= r["score"] < pass_line),
        "faithful_refused": _share(faithful, lambda r: r["score"] < refuse_line),
        "unfaithful_flagged": _share(unfaithful, lambda r: r["score"] < pass_line),
        "unfaithful_refused": _share(unfaithful, lambda r: r["score"] < refuse_line),
        "unfaithful_missed": _share(unfaithful, lambda r: r["score"] >= pass_line),
        "accuracy": _share(rows, lambda r: (r["score"] >= pass_line) == r["faithful"]),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2) if rows else None,
    }


def _fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.0%}"


def _print_group(name: str, summaries: dict[str, dict]) -> None:
    first = next(iter(summaries.values()))
    print(f"\n{name}: {first['n_faithful']} faithful / {first['n_unfaithful']} unfaithful pairs")
    print(
        f"  {'strategy':<22}{'faith clean':>12}{'warned':>8}{'refused':>9}"
        f"{'unfaith flagged':>17}{'missed':>8}{'accuracy':>10}{'sec/answer':>12}"
    )
    for strategy, s in summaries.items():
        print(
            f"  {strategy:<22}{_fmt(s['faithful_clean']):>12}{_fmt(s['faithful_warned']):>8}"
            f"{_fmt(s['faithful_refused']):>9}{_fmt(s['unfaithful_flagged']):>17}"
            f"{_fmt(s['unfaithful_missed']):>8}{_fmt(s['accuracy']):>10}{s['mean_seconds']:>12.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pairs = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    routes = sorted({p["route"] for p in pairs})
    print(f"{len(pairs)} pairs across routes: {routes}")
    if "resume" not in routes:
        print("NOTE: no resume-route pairs; run eval/build_faithfulness_test_set.py first.")

    print("Loading HHEM-2.1-Open...")
    model = fc._get_hhem()
    model.predict([("warm up", "warm up")])  # keep one-time initialisation out of the timings
    cached = _CachedModel(model)
    fc._hhem = cached

    rows_by_strategy: dict[str, list[dict]] = {}
    for strategy, kwargs in STRATEGIES:
        rows = []
        for pair in pairs:
            cached.used.clear()
            score = fc.hhem_score(pair["answer"], pair["context"], **kwargs)
            rows.append(
                {
                    "id": pair["id"],
                    "route": pair["route"],
                    "faithful": pair["faithful"],
                    "context_chars": len(pair["context"]),
                    "score": round(score, 4),
                    "seconds": round(cached.seconds_for_used(), 3),
                }
            )
        rows_by_strategy[strategy] = rows
        print(f"  scored {strategy}")

    groups = {
        "resume": lambda r: r["route"] == "resume",
        "other routes (short contexts)": lambda r: r["route"] != "resume",
        "all": lambda r: True,
    }
    summary: dict[str, dict[str, dict]] = {}
    for group, selector in groups.items():
        summary[group] = {
            strategy: _summarize([r for r in rows if selector(r)])
            for strategy, rows in rows_by_strategy.items()
        }
        if any(selector(r) for r in next(iter(rows_by_strategy.values()))):
            _print_group(group, summary[group])

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "run_at": datetime.now(UTC).isoformat(),
                    "pairs": len(pairs),
                    "routes_covered": routes,
                    "pass_line": fc._HHEM_PASS_ABOVE,
                    "refuse_line": fc._HHEM_REFUSE_BELOW,
                    "summary": summary,
                    "rows": rows_by_strategy,
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

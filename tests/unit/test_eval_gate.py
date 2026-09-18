"""The gate has to fail on a regression, or it is decoration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from compare import compare, latest_baseline  # noqa: E402

BASE = {
    "run_at": "2026-09-17T00:00:00Z",
    "git_sha": "aaaaaaa",
    "corpus_sha": "cafebabe",
    "routing": {"accuracy": 0.90},
    "retrieval": {"recall_at_k": 0.85, "mrr": 0.70},
    "refusals": {"refusal_rate": 1.0},
    "scores": {"faithfulness": 0.70, "answer_relevancy": 0.80},
}


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_identical_runs_pass(tmp_path):
    a = _write(tmp_path, "a.json", BASE)
    b = _write(tmp_path, "b.json", BASE)
    assert compare(a, b) == 0


def test_a_large_drop_fails(tmp_path):
    worse = json.loads(json.dumps(BASE))
    worse["routing"]["accuracy"] = 0.60
    assert compare(_write(tmp_path, "w.json", worse), _write(tmp_path, "b.json", BASE)) == 1


def test_a_small_drop_is_tolerated(tmp_path):
    """Nothing here is deterministic; the graph calls a model."""
    jittered = json.loads(json.dumps(BASE))
    jittered["routing"]["accuracy"] = 0.88
    assert compare(_write(tmp_path, "j.json", jittered), _write(tmp_path, "b.json", BASE)) == 0


def test_an_improvement_passes(tmp_path):
    better = json.loads(json.dumps(BASE))
    better["scores"]["faithfulness"] = 0.95
    assert compare(_write(tmp_path, "g.json", better), _write(tmp_path, "b.json", BASE)) == 0


def test_missing_metrics_are_skipped_not_failed(tmp_path):
    partial = {"run_at": "2026-09-17T00:00:00Z", "routing": {"accuracy": 0.90}}
    assert compare(_write(tmp_path, "p.json", partial), _write(tmp_path, "b.json", BASE)) == 0


def test_the_committed_baseline_is_discoverable():
    """CI resolves the baseline by itself; if that breaks, the gate silently passes."""
    found = latest_baseline()
    assert found is not None, "no committed result in eval/results"
    payload = json.loads(found.read_text(encoding="utf-8"))
    assert payload["routing"]["accuracy"] > 0


@pytest.mark.parametrize("metric", ["routing.accuracy", "retrieval.recall_at_k", "scores.faithfulness"])
def test_every_gated_metric_is_present_in_the_committed_baseline(metric):
    payload = json.loads(latest_baseline().read_text(encoding="utf-8"))
    node = payload
    for part in metric.split("."):
        assert part in node, f"{metric} missing from the committed baseline"
        node = node[part]

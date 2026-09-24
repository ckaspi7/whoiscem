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


def test_a_changed_data_source_is_reported(tmp_path, capsys):
    """Re-seeding the personal database moves the numbers; say so."""
    before = dict(BASE, data_sha={"personal_db": "aaa", "spotify_cache": "bbb"})
    after = dict(BASE, data_sha={"personal_db": "zzz", "spotify_cache": "bbb"})
    compare(_write(tmp_path, "a.json", after), _write(tmp_path, "b.json", before))

    out = capsys.readouterr().out
    assert "personal_db" in out
    assert "spotify_cache" not in out.split("data sources changed")[1].split("\n")[0]


def test_a_changed_retrieval_config_is_reported(tmp_path, capsys):
    before = dict(BASE, retrieval_config={"strategy": "auto", "top_n": 5})
    after = dict(BASE, retrieval_config={"strategy": "dense", "top_n": 5})
    compare(_write(tmp_path, "a.json", after), _write(tmp_path, "b.json", before))
    assert "retrieval configuration changed" in capsys.readouterr().out


def test_a_baseline_without_provenance_reports_no_spurious_changes(tmp_path, capsys):
    """A warning that fires on every run is one nobody reads."""
    old_style = dict(BASE)  # no data_sha, no retrieval_config
    new_style = dict(BASE, data_sha={"personal_db": "aaa"}, retrieval_config={"strategy": "auto"})
    compare(_write(tmp_path, "a.json", new_style), _write(tmp_path, "b.json", old_style))

    out = capsys.readouterr().out
    assert "data sources changed" not in out
    assert "retrieval configuration changed" not in out


# ---------------------------------------------------------------------------
# Tool health — a hard failure, not a tolerance band
#
# This is the check that would have caught the incident that motivated it: a
# scheduled run whose "Run full evaluation" step succeeded but whose
# "Compare against the committed baseline" step failed, because a fresh
# Qdrant container racing the first query silently broke every resume-routed
# question for the rest of that run. A metric-delta check saw a drop; this
# check would have named the cause.
# ---------------------------------------------------------------------------


def test_a_tool_error_fails_the_gate_even_with_otherwise_identical_scores(tmp_path):
    """There is no baseline error rate worth matching — any failure gates."""
    broken = dict(BASE, tool_health={"errors": 1, "error_rate": 0.017, "failed_ids": ["r008"]})
    assert compare(_write(tmp_path, "a.json", broken), _write(tmp_path, "b.json", BASE)) == 1


def test_zero_tool_errors_does_not_trip_the_gate(tmp_path):
    clean = dict(BASE, tool_health={"errors": 0, "error_rate": 0.0, "failed_ids": []})
    assert compare(_write(tmp_path, "a.json", clean), _write(tmp_path, "b.json", BASE)) == 0


def test_a_tool_error_names_the_failed_questions(tmp_path, capsys):
    broken = dict(BASE, tool_health={"errors": 2, "error_rate": 0.034, "failed_ids": ["r008", "r014"]})
    compare(_write(tmp_path, "a.json", broken), _write(tmp_path, "b.json", BASE))

    out = capsys.readouterr().out
    assert "r008" in out
    assert "r014" in out
    assert "2 tool call(s) failed" in out


# ---------------------------------------------------------------------------
# Granularity changes — a rechunk or a different agent_mode both change what
# a "chunk" is, so rank-position metrics (MRR, context_precision) are not
# comparable in either direction, not just when they drop.
# ---------------------------------------------------------------------------


def test_a_rechunk_does_not_gate_a_dropped_mrr(tmp_path):
    before = dict(BASE, chunker="v1", retrieval={"recall_at_k": 0.85, "mrr": 0.70})
    after = dict(BASE, chunker="v2", retrieval={"recall_at_k": 0.85, "mrr": 0.30})
    assert compare(_write(tmp_path, "a.json", after), _write(tmp_path, "b.json", before)) == 0


def test_an_agent_mode_change_does_not_gate_a_dropped_mrr(tmp_path):
    before = dict(BASE, agent_mode="classifier", retrieval={"recall_at_k": 0.85, "mrr": 0.70})
    after = dict(BASE, agent_mode="tool_calling", retrieval={"recall_at_k": 0.85, "mrr": 0.20})
    assert compare(_write(tmp_path, "a.json", after), _write(tmp_path, "b.json", before)) == 0


def test_an_agent_mode_change_labels_a_raised_mrr_as_uninterpreted_not_improved(tmp_path, capsys):
    """A single pre-merged chunk is trivially rank 1 if found at all — an MRR
    jump from switching modes is an artifact, not evidence retrieval improved."""
    before = dict(BASE, agent_mode="classifier", retrieval={"recall_at_k": 0.85, "mrr": 0.57})
    after = dict(BASE, agent_mode="tool_calling", retrieval={"recall_at_k": 0.85, "mrr": 1.00})
    compare(_write(tmp_path, "a.json", after), _write(tmp_path, "b.json", before))

    out = capsys.readouterr().out
    assert "not gated: granularity changed" in out
    assert "retrieval.mrr" in out and "improved" not in out.split("retrieval.mrr")[1].split("\n")[0]


def test_a_non_rank_metric_still_gates_normally_when_agent_mode_changes(tmp_path):
    """Only the rank-sensitive metrics get the pass — an accuracy drop from
    switching architectures is a real, reportable finding, not noise."""
    before = dict(BASE, agent_mode="classifier", routing={"accuracy": 0.98})
    after = dict(BASE, agent_mode="tool_calling", routing={"accuracy": 0.70})
    assert compare(_write(tmp_path, "a.json", after), _write(tmp_path, "b.json", before)) == 1

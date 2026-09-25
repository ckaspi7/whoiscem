"""score_trajectory (Phase 3.5): did a tool-calling turn take the right path,
not just touch an acceptable route. Pure function, no graph or LLM involved —
tests/unit/test_tool_calling_agent.py covers that the trajectory itself is
recorded correctly; this covers what the scorer does with one.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from run_eval import score_trajectory  # noqa: E402


def _row(
    id_: str, case_type: str, expected_route: str, acceptable_routes: list[str], trajectory: list[dict]
) -> dict:
    return {
        "id": id_,
        "case_type": case_type,
        "expected_route": expected_route,
        "acceptable_routes": acceptable_routes,
        "trajectory": trajectory,
    }


def _call(category: str, round_: int = 0, ok: bool = True) -> dict:
    return {"round": round_, "tool": f"get_{category}_info", "category": category, "args": {}, "ok": ok}


def test_classifier_rows_with_no_trajectory_report_as_unmeasured():
    rows = [{"id": "r001", "case_type": "factual", "expected_route": "resume", "trajectory": []}]
    assert score_trajectory(rows) == {"measured": 0}


def test_a_complete_multi_intent_call_is_recognized():
    rows = [
        _row("m002", "multi_intent", "resume", ["resume", "linkedin"], [_call("resume"), _call("linkedin")]),
    ]
    result = score_trajectory(rows)
    assert result["multi_intent_total"] == 1
    assert result["multi_intent_complete"] == 1
    assert result["multi_intent_incomplete_ids"] == []


def test_a_multi_intent_question_that_only_called_one_tool_is_incomplete():
    """route_correct alone would call this a pass — one acceptable category
    was touched. Trajectory eval is the check that it only half-answered."""
    rows = [
        _row("m002", "multi_intent", "resume", ["resume", "linkedin"], [_call("resume")]),
    ]
    result = score_trajectory(rows)
    assert result["multi_intent_total"] == 1
    assert result["multi_intent_complete"] == 0
    assert result["multi_intent_incomplete_ids"] == ["m002"]


def test_a_single_intent_question_needs_no_completeness_check():
    """multi_intent is the only case_type acceptable_routes lists as an
    all-of; a factual question naming several acceptable single routes is not
    held to the same "touch every one" standard."""
    rows = [_row("r001", "factual", "resume", ["resume"], [_call("resume")])]
    result = score_trajectory(rows)
    assert result["multi_intent_total"] == 0
    assert result["multi_intent_incomplete_ids"] == []


def test_a_call_outside_the_acceptable_routes_is_flagged_unnecessary():
    rows = [_row("r001", "factual", "resume", ["resume"], [_call("resume"), _call("spotify")])]
    result = score_trajectory(rows)
    assert result["unnecessary_calls"] == [{"id": "r001", "extra": ["spotify"]}]


def test_calls_within_the_acceptable_routes_are_not_flagged():
    calls = [_call("resume"), _call("spotify")]
    rows = [_row("m001", "multi_intent", "resume", ["resume", "spotify"], calls)]
    assert score_trajectory(rows)["unnecessary_calls"] == []


def test_avg_tool_calls_reflects_the_full_trajectory_length():
    rows = [
        _row("r001", "factual", "resume", ["resume"], [_call("resume")]),
        _row("m001", "multi_intent", "resume", ["resume", "spotify"], [_call("resume"), _call("spotify")]),
    ]
    assert score_trajectory(rows)["avg_tool_calls"] == 1.5


def test_a_failed_call_still_counts_toward_completeness():
    """Completeness measures whether the right categories were reached for,
    not whether every call succeeded — score_tool_health already owns
    failure-rate reporting, separately."""
    rows = [
        _row(
            "m002",
            "multi_intent",
            "resume",
            ["resume", "linkedin"],
            [_call("resume"), _call("linkedin", ok=False)],
        ),
    ]
    assert score_trajectory(rows)["multi_intent_complete"] == 1

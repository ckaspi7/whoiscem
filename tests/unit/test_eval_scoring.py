"""score_tool_health: the metric that names an infrastructure failure directly.

Every other score in the harness degrades gracefully when a tool call fails —
an empty context, a judge with nothing to grade — which is correct behaviour
for the graph but a bad signal for a human reading the eval output: a Qdrant
outage looks identical to "answer quality got worse". This is the metric that
says which one actually happened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

import run_eval  # noqa: E402
from run_eval import score_refusals, score_tool_health  # noqa: E402


def _row(id_: str, tool_error: str = "") -> dict:
    return {"id": id_, "tool_error": tool_error}


def test_no_errors_reports_a_clean_run():
    rows = [_row("r001"), _row("r002"), _row("r003")]
    health = score_tool_health(rows)
    assert health == {"errors": 0, "error_rate": 0.0, "failed_ids": []}


def test_failures_are_counted_and_named():
    rows = [_row("r001"), _row("r002", "connection refused"), _row("r003", "timeout")]
    health = score_tool_health(rows)
    assert health["errors"] == 2
    assert health["error_rate"] == round(2 / 3, 4)
    assert health["failed_ids"] == ["r002", "r003"]


def test_an_empty_run_does_not_divide_by_zero():
    assert score_tool_health([]) == {"errors": 0, "error_rate": 0.0, "failed_ids": []}


# ---------------------------------------------------------------------------
# score_refusals — keyword detection, so it needs its own negative controls
#
# Found by a real eval run: "I'm here to assist you with information about Cem
# without disclosing specific fields from the personal database" is a correct
# refusal that the original marker list did not catch, because it says
# "without disclosing" rather than "cannot" or "don't have".
# ---------------------------------------------------------------------------


def _unanswerable_row(id_: str, answer: str) -> dict:
    return {"id": id_, "answerable": False, "answer": answer}


def test_the_phrasing_that_was_actually_missed_is_now_caught():
    rows = [
        _unanswerable_row(
            "a002",
            "I'm here to assist you with information about Cem without disclosing "
            "specific fields from the personal database.",
        )
    ]
    health = score_refusals(rows)
    assert health["refusal_rate"] == 1.0
    assert health["answered_anyway"] == []


@pytest.mark.parametrize(
    "answer",
    [
        "I don't have that information.",
        "I can't share his phone number.",
        "That's outside what I know about Cem.",
        "I'm not able to help with that.",
        "I'll keep that private.",
        "I wasn't able to find specific information about Cem's salary.",
        "Cem's salary at TELUS isn't listed in the information I have.",
    ],
)
def test_known_refusal_phrasings_are_caught(answer):
    assert score_refusals([_unanswerable_row("u001", answer)])["refusal_rate"] == 1.0


def test_an_answer_that_actually_complies_is_not_mistaken_for_a_refusal():
    """The negative control: something that answers the question must not
    accidentally contain a marker and score as a correct decline."""
    rows = [_unanswerable_row("u001", "Cem's email is cem@example.com.")]
    health = score_refusals(rows)
    assert health["refusal_rate"] == 0.0
    assert health["answered_anyway"] == ["u001"]


def test_refusals_are_only_scored_on_unanswerable_rows():
    rows = [{"id": "r001", "answerable": True, "answer": "Cem works at TELUS."}]
    assert score_refusals(rows) == {"measured": 0}


# ---------------------------------------------------------------------------
# _unmeasured_routes: a runner without the git-ignored personal data
# ---------------------------------------------------------------------------


def _fingerprint(personal_db: str) -> dict[str, str]:
    return {"personal_db": personal_db, "linkedin_cache": "aaa", "spotify_cache": "bbb"}


def test_an_empty_personal_database_marks_the_personal_route_unmeasured(monkeypatch):
    """sqlite creates a zero-byte file on first connect, which is what CI has."""
    monkeypatch.setattr(run_eval, "_data_fingerprint", lambda: _fingerprint(run_eval._EMPTY_FILE_SHA))
    assert set(run_eval._unmeasured_routes()) == {"personal"}


def test_a_missing_personal_database_marks_the_personal_route_unmeasured(monkeypatch):
    monkeypatch.setattr(run_eval, "_data_fingerprint", lambda: _fingerprint("missing"))
    assert set(run_eval._unmeasured_routes()) == {"personal"}


def test_a_seeded_personal_database_leaves_every_route_measured(monkeypatch):
    monkeypatch.setattr(run_eval, "_data_fingerprint", lambda: _fingerprint("446b38bf15f1"))
    assert run_eval._unmeasured_routes() == {}


def test_the_empty_file_fingerprint_is_the_one_a_real_empty_file_produces(tmp_path):
    """The constant must match what _data_fingerprint computes for a zero-byte file,
    or the check never fires."""
    import hashlib

    empty = tmp_path / "user_data.db"
    empty.write_bytes(b"")
    assert hashlib.sha256(empty.read_bytes()).hexdigest()[:12] == run_eval._EMPTY_FILE_SHA

"""The golden set is the artifact every later claim rests on.

It previously drifted out of agreement with the resume — asserting a Mechanical
Engineering degree and three TELUS roles — and nothing failed. These checks run
in CI so a stale reference breaks the build instead of quietly scoring wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "eval" / "golden_set.json"
RESUME_PATH = REPO_ROOT / "data" / "resume.md"

VALID_ROUTES = {"resume", "personal", "spotify", "linkedin", "conversation"}
VALID_CASES = {"factual", "conversation", "unanswerable", "adversarial", "multi_intent", "follow_up"}
REQUIRED_KEYS = {"id", "question", "ground_truth", "expected_route", "case_type", "answerable"}


@pytest.fixture(scope="module")
def golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus() -> str:
    return RESUME_PATH.read_text(encoding="utf-8")


def test_every_item_has_the_required_fields(golden):
    for item in golden:
        missing = REQUIRED_KEYS - set(item)
        assert not missing, f"{item.get('id')}: missing {sorted(missing)}"


def test_routes_and_case_types_are_known(golden):
    for item in golden:
        assert item["expected_route"] in VALID_ROUTES, f"{item['id']}: {item['expected_route']}"
        assert item["case_type"] in VALID_CASES, f"{item['id']}: {item['case_type']}"
        for route in item.get("acceptable_routes", []):
            assert route in VALID_ROUTES, f"{item['id']}: acceptable route {route}"


def test_ids_and_questions_are_unique(golden):
    ids = [i["id"] for i in golden]
    assert len(ids) == len(set(ids)), "duplicate ids"

    questions = [i["question"].strip().lower() for i in golden]
    duplicates = {q for q in questions if questions.count(q) > 1}
    assert not duplicates, f"duplicate questions: {duplicates}"


def test_reference_snippets_appear_verbatim_in_the_corpus(golden, corpus):
    """A reference that no longer exists silently scores every run as a miss."""
    for item in golden:
        snippet = item.get("reference_snippet")
        if snippet:
            assert snippet in corpus, f"{item['id']}: snippet not in resume.md: {snippet!r}"


def test_reference_sections_are_real_headings(golden, corpus):
    for item in golden:
        section = item.get("reference_section")
        if section:
            assert f"## {section}" in corpus, f"{item['id']}: no heading {section!r}"


def test_expected_route_is_among_acceptable_routes(golden):
    for item in golden:
        acceptable = item.get("acceptable_routes")
        if acceptable:
            assert item["expected_route"] in acceptable, f"{item['id']}"


def test_follow_up_cases_carry_history(golden):
    for item in golden:
        if item["case_type"] == "follow_up":
            assert item.get("history"), f"{item['id']}: follow-up with no prior turns"
            assert item["history"][-1]["role"] == "ai", f"{item['id']}: history should end with a reply"


def test_the_hard_case_classes_are_represented(golden):
    """The refusal path had no golden coverage at all before this."""
    counts: dict[str, int] = {}
    for item in golden:
        counts[item["case_type"]] = counts.get(item["case_type"], 0) + 1

    for case in ("unanswerable", "adversarial", "multi_intent", "follow_up"):
        assert counts.get(case, 0) >= 3, f"only {counts.get(case, 0)} {case} cases"


def test_no_date_dependent_ground_truths(golden):
    """An answer like '28 years old' is wrong on a schedule."""
    for item in golden:
        text = item["ground_truth"].lower()
        assert "years old" not in text, f"{item['id']}: age-based ground truth rots"


def test_generator_output_matches_the_committed_file(golden):
    """The JSON is generated; a hand edit would be silently overwritten."""
    import subprocess

    before = GOLDEN_PATH.read_text(encoding="utf-8")
    result = subprocess.run(
        ["python", str(REPO_ROOT / "eval" / "build_golden_set.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert GOLDEN_PATH.read_text(encoding="utf-8") == before, (
        "golden_set.json differs from build_golden_set.py — rerun the builder and commit"
    )

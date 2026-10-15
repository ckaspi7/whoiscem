"""guardrails.faithfulness_check: how much of the context HHEM reads.

The resume route hands the judge the whole resume. Scored against all of it,
verbatim-correct answers were under-scored and each call took seconds, so the
judge can read a leading window first and look at the whole context only when
that window does not pass. These tests pin the window arithmetic and the
escalation rule with a stub model; the measured effect is in
eval/ablate_judge_context.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from guardrails.faithfulness_check import _focus_window, hhem_score, score_faithfulness

_PARAGRAPH = 100


def _context(paragraphs: int) -> str:
    return "\n\n".join(f"p{i}:" + "x" * (_PARAGRAPH - 3) for i in range(paragraphs))


def _stub(context: str, *, focus: float, full: float) -> MagicMock:
    """A model that scores the whole context ``full`` and any shorter premise ``focus``."""
    model = MagicMock()
    model.predict.side_effect = lambda pairs: [full if pairs[0][0] == context else focus]
    return model


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def test_a_context_within_the_budget_is_used_whole():
    assert _focus_window("first\n\nsecond", budget=100) == "first\n\nsecond"


def test_the_window_keeps_whole_leading_paragraphs_up_to_the_budget():
    context = _context(6)

    window = _focus_window(context, budget=250)  # two paragraphs fit (202 chars), a third would not

    assert window == "\n\n".join(context.split("\n\n")[:2])


def test_the_window_never_cuts_mid_paragraph():
    context = _context(8)
    paragraphs = context.split("\n\n")

    for budget in (50, 150, 333, 777):
        kept = _focus_window(context, budget).split("\n\n")
        assert kept == paragraphs[: len(kept)]


def test_the_first_paragraph_is_kept_even_when_it_alone_exceeds_the_budget():
    context = "a" * 500 + "\n\n" + "b" * 50

    assert _focus_window(context, budget=100) == "a" * 500


# ---------------------------------------------------------------------------
# The modes
# ---------------------------------------------------------------------------


def test_full_mode_scores_the_whole_context_once():
    context = _context(30)
    model = _stub(context, focus=0.9, full=0.2)

    with patch("guardrails.faithfulness_check._get_hhem", return_value=model):
        score = hhem_score("answer", context, mode="full")

    assert score == 0.2
    assert model.predict.call_count == 1


def test_focus_mode_scores_only_the_window_even_when_that_is_low():
    context = _context(30)
    model = _stub(context, focus=0.1, full=0.9)

    with patch("guardrails.faithfulness_check._get_hhem", return_value=model):
        score = hhem_score("answer", context, mode="focus", focus_chars=500)

    assert score == 0.1
    assert model.predict.call_count == 1


def test_a_passing_window_never_pays_for_a_second_look():
    context = _context(30)
    model = _stub(context, focus=0.9, full=0.1)

    with patch("guardrails.faithfulness_check._get_hhem", return_value=model):
        score = hhem_score("answer", context, mode="focus_then_full", focus_chars=500)

    assert score == 0.9
    assert model.predict.call_count == 1


def test_a_failing_window_escalates_and_the_higher_score_wins():
    context = _context(30)
    model = _stub(context, focus=0.2, full=0.7)

    with patch("guardrails.faithfulness_check._get_hhem", return_value=model):
        score = hhem_score("answer", context, mode="focus_then_full", focus_chars=500)

    assert score == 0.7
    assert model.predict.call_count == 2


def test_escalation_cannot_lower_a_score():
    context = _context(30)
    model = _stub(context, focus=0.3, full=0.1)

    with patch("guardrails.faithfulness_check._get_hhem", return_value=model):
        score = hhem_score("answer", context, mode="focus_then_full", focus_chars=500)

    assert score == 0.3
    assert model.predict.call_count == 2


def test_a_context_that_fits_the_window_is_scored_once_even_when_low():
    """Escalating would score the identical text twice."""
    context = _context(3)
    model = MagicMock()
    model.predict.return_value = [0.05]

    with patch("guardrails.faithfulness_check._get_hhem", return_value=model):
        score = hhem_score("answer", context, mode="focus_then_full", focus_chars=2000)

    assert score == 0.05
    assert model.predict.call_count == 1


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        hhem_score("answer", "context", mode="everything")


def test_an_empty_context_returns_none_without_loading_the_model():
    with patch("guardrails.faithfulness_check._get_hhem", side_effect=AssertionError("must not load")):
        assert hhem_score("answer", "   ") is None


def test_the_default_judges_the_whole_context_in_one_look():
    """The windowed modes exist but are off, so what production scores is unchanged."""
    context = _context(30)
    model = _stub(context, focus=0.9, full=0.1)

    with patch("guardrails.faithfulness_check._get_hhem", return_value=model):
        score = hhem_score("answer", context)

    assert score == 0.1
    assert model.predict.call_count == 1


def test_score_faithfulness_uses_the_default_mode():
    with patch("guardrails.faithfulness_check.hhem_score", return_value=0.9) as scorer:
        assert score_faithfulness("answer", "context") == 5

    scorer.assert_called_once_with("answer", "context")

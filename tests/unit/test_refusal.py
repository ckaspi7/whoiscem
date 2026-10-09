"""guardrails/refusal.py: recognising the model's own "I don't have that information"."""

from __future__ import annotations

import pytest

from guardrails.refusal import looks_like_refusal


@pytest.mark.parametrize(
    "answer",
    [
        "I don't have that information.",
        "I don't have that information about Cem.",
        "I don’t have any information on that.",
        "I do not have enough information to say.",
        "The resume doesn't have details about his salary.",
    ],
)
def test_it_recognises_the_model_declining(answer):
    assert looks_like_refusal(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "Cem works as an AI/ML Engineer at TELUS Communications Inc. in Vancouver.",
        "He doesn't have a PhD, but he mentors engineers in a UT Austin programme.",
        "Cem's Spotify history from March 2025 shows R&B and hip hop.",
    ],
)
def test_it_does_not_flag_an_ordinary_answer(answer):
    assert looks_like_refusal(answer) is False

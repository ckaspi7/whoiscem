"""The system prompt's scope instruction (ADR-0011).

Without it, a model answers an off-topic question from its own knowledge: gpt-6-luna
answered "Who won the 2026 World Cup?" in every run. These tests pin that the
instruction is in the prompt and that the eval's refusal markers recognise the
phrasing it produces, since a prompt change that the gate cannot read would report
a correct refusal as a failure.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from run_eval import _REFUSAL_MARKERS  # noqa: E402

import chatbot  # noqa: E402


def test_the_prompt_tells_the_model_to_stay_on_topic():
    prompt = chatbot._system_prompt("")
    assert "only answer questions about Cem" in prompt
    assert "Never answer such a question from your" in prompt


def test_the_prompt_still_allows_greetings():
    assert "Greetings and thanks are fine" in chatbot._system_prompt("")


@pytest.mark.parametrize("phrase", ["only help with questions about", "only answer questions about"])
def test_the_eval_recognises_the_scope_refusal_the_prompt_asks_for(phrase):
    assert phrase in _REFUSAL_MARKERS

from __future__ import annotations

import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_LOW_SCORE_RESPONSE = "I don't have reliable information about that in my knowledge base."
_WARN_PREFIX = "⚠️ *Note: I'm not fully confident in this answer — please verify independently.*\n\n"
_NO_USAGE = {"input": 0, "output": 0}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _judge(answer: str, context: str) -> tuple[int | None, dict[str, int]]:
    """The real judge call: score plus the real token usage it cost.

    None means "nothing to act on" — either there is no context to check
    against, or the judge call itself failed — never a low score. A caller
    driving a retry must treat None as "don't retry", not as "score is bad".
    A failure is logged rather than silently swallowed (Phase 4.3): a
    guardrail that can go quietly inert is worse than no guardrail, and this
    is the one place that ever finds out it just did.
    """
    if not context.strip():
        return None, dict(_NO_USAGE)

    prompt = (
        "You are a faithfulness judge. Given a context and an answer, "
        "score how much of the answer is grounded in the context on a scale of 1-5. "
        "1 = mostly hallucinated, 5 = fully grounded. "
        'Respond with valid JSON only: {"score": <int>, "reason": "<str>"}'
    )
    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}"},
            ],
            temperature=0,
            max_tokens=100,
        )
        usage = response.usage
        tokens = (
            {"input": usage.prompt_tokens, "output": usage.completion_tokens} if usage else dict(_NO_USAGE)
        )
        result = json.loads(response.choices[0].message.content)
        return int(result.get("score", 1)), tokens
    except Exception as exc:
        logger.warning("Faithfulness judge call failed, failing open (no check applied): %s", exc)
        return None, dict(_NO_USAGE)


def score_faithfulness(answer: str, context: str) -> int | None:
    """Score how grounded `answer` is in `context`, 1 (hallucinated) to 5 (fully grounded).

    See `_judge` for what None means. Use `score_faithfulness_with_usage` when
    the caller needs real cost accounting for this call, not just the score.
    """
    score, _ = _judge(answer, context)
    return score


def score_faithfulness_with_usage(answer: str, context: str) -> tuple[int | None, dict[str, int]]:
    """Same as `score_faithfulness`, plus the real token usage the call cost —
    for callers doing honest cost accounting (Phase 4.2), not just scoring."""
    return _judge(answer, context)


def apply_faithfulness_tiering(answer: str, score: int | None) -> str:
    """Turn a raw score into the user-facing text: unchanged, warned, or refused.

    Split out from `check_faithfulness` so a caller that already has a score
    (chatbot.py's in-graph check, which computed one for the retry decision)
    can apply the same tiering without paying for a second judge call on the
    same answer.
    """
    if score is None or score >= 4:
        return answer
    if score >= 2:
        return _WARN_PREFIX + answer
    return _LOW_SCORE_RESPONSE


def check_faithfulness(answer: str, context: str) -> str:
    """Score answer faithfulness against retrieved context, then gate or warn.

    Score 4-5: return answer unchanged.
    Score 2-3: prepend a visible warning badge.
    Score 1:   replace with a safe refusal.
    No score (no context, or the judge call failed): return answer unchanged.
    """
    return apply_faithfulness_tiering(answer, score_faithfulness(answer, context))

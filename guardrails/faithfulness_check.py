from __future__ import annotations

import json
import os

from openai import OpenAI

_client: OpenAI | None = None
_LOW_SCORE_RESPONSE = "I don't have reliable information about that in my knowledge base."
_WARN_PREFIX = "⚠️ *Note: I'm not fully confident in this answer — please verify independently.*\n\n"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def check_faithfulness(answer: str, context: str) -> str:
    """Score answer faithfulness against retrieved context, then gate or warn.

    Score 4-5: return answer unchanged.
    Score 2-3: prepend a visible warning badge.
    Score 1:   replace with a safe refusal.
    Falls back to the original answer if the judge call fails.
    """
    if not context.strip():
        return answer

    prompt = (
        "You are a faithfulness judge. Given a context and an answer, "
        "score how much of the answer is grounded in the context on a scale of 1-5. "
        "1 = mostly hallucinated, 5 = fully grounded. "
        'Respond with valid JSON only: {"score": <int>, "reason": "<str>"}'
    )
    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}"},
            ],
            temperature=0,
            max_tokens=100,
        )
        result = json.loads(response.choices[0].message.content)
        score = int(result.get("score", 1))
    except Exception:
        return answer  # fail open — don't break the app

    if score >= 4:
        return answer
    if score >= 2:
        return _WARN_PREFIX + answer
    return _LOW_SCORE_RESPONSE

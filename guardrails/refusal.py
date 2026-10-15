"""Detect when the model itself declined to answer.

chatbot._system_prompt tells the model to say it "doesn't have that
information" when a tool returns nothing relevant. That is honest when the
knowledge base really lacks the answer, and misleading when the tool failed:
the visitor sees the same sentence either way. Callers use this to tell the
two apart once they know whether a tool errored.
"""

from __future__ import annotations

import re

_REFUSAL = re.compile(
    r"(don't|do not|doesn't|does not) have (that |any |enough |specific )*(information|info|details)",
    re.IGNORECASE,
)


def looks_like_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer.replace("’", "'")))

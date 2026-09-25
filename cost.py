"""Real token-usage-based cost accounting (Phase 4.2).

The sidebar used to estimate tokens as ``len(text) // 4`` and apply the
*output* price to everything, missing four of the five LLM calls a turn
actually makes (router, judge, summariser, embeddings) and getting the one it
did count wrong by construction. This module holds the one thing that was
missing to fix it: real per-model pricing to apply to real usage numbers,
which the API already returns on every call.

Embeddings are deliberately not priced here: `text-embedding-3-small` is
~$0.02/1M tokens on a handful of tokens per query, several orders of
magnitude below either chat model's cost, and often skipped entirely by
``RetrievalCache``. Including it would add a real code path for a contribution
too small to move the number it is added to.
"""

from __future__ import annotations

# $ per 1,000 tokens. Verified pricing, not guessed — see
# memory/gpt-6-luna-candidate research and OpenAI's published rates.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-6-luna": {"input": 0.000100, "output": 0.000500},
}


def usage_from_response(response) -> dict[str, int]:
    """Real input/output token counts from a LangChain ``AIMessage``.

    ``usage_metadata`` is populated by langchain_openai regardless of
    streaming or provider-specific response shape — this is the direct fix
    for estimating tokens from text length. Not `get_openai_callback`: that
    relies on a context-var registration that does not propagate through
    LangGraph's node execution (confirmed directly — a real graph turn showed
    0 tokens captured that way despite a real model call happening), so each
    call site reads its own response instead.

    Requires an actual dict, not just a truthy value: a provider that never
    populates ``usage_metadata`` leaves it ``None``, which this treats the
    same as absent — the alternative, a bare ``getattr(..., None) or {}``,
    would also silently accept a non-dict stand-in (a mock without the
    attribute configured auto-creates one rather than raising) and then fail
    downstream trying to add a token count to it.
    """
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        return {"input": 0, "output": 0}
    return {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0)}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost for real token counts, at `model`'s real published rate.

    Falls back to gpt-4o-mini's rate for an unrecognised model rather than
    raising or silently returning zero — a wrong-but-present estimate is a
    better failure mode for a cost meter than one that vanishes.
    """
    rates = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000

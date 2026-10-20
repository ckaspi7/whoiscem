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
    # Claude Haiku 5.5, for prompts up to 100,000 tokens (above that the rates rise to
    # $0.50 / $2.50 per million, which no prompt here comes near). It counts about 30%
    # more tokens than earlier Claude models for the same text, so compare cost per
    # question, not per token. Cache reads and writes are not priced: nothing here caches.
    "claude-haiku-5-5": {"input": 0.000100, "output": 0.000500},
}


def canonical_model(name: object) -> str | None:
    """The PRICING key for a model name a provider reported, or None if unrecognised.

    OpenAI reports dated snapshots (``gpt-4o-mini-2024-07-18``), so an exact lookup
    would miss them; the key itself, or the key followed by ``-``, is the match.
    """
    if not isinstance(name, str):
        return None
    for key in PRICING:
        if name == key or name.startswith(key + "-"):
            return key
    return None


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
    counts: dict[str, int | str] = {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
    }
    # Which model actually answered, so a fallback's tokens are priced at its own rate
    # and not the primary's. Left out when the response does not say.
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        model = canonical_model(metadata.get("model_name") or metadata.get("model"))
        if model:
            counts["model"] = model
    return counts  # type: ignore[return-value]


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost for real token counts, at `model`'s real published rate.

    Falls back to gpt-4o-mini's rate for an unrecognised model rather than
    raising or silently returning zero — a wrong-but-present estimate is a
    better failure mode for a cost meter than one that vanishes.
    """
    rates = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000


def graph_cost(token_usage: dict | None, default_model: str) -> tuple[float, int, int]:
    """(dollars, input tokens, output tokens) for a turn's node-keyed usage.

    Each node is priced at the model that answered it when the usage says which, and at
    `default_model` otherwise, so a turn that fell back to the second model mid-way is
    billed honestly rather than all at the primary's rate.
    """
    dollars, total_in, total_out = 0.0, 0, 0
    for usage in (token_usage or {}).values():
        n_in, n_out = usage.get("input", 0), usage.get("output", 0)
        dollars += estimate_cost(usage.get("model") or default_model, n_in, n_out)
        total_in += n_in
        total_out += n_out
    return dollars, total_in, total_out

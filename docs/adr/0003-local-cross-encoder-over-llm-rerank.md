# 0003 — A local cross-encoder reranks, not an LLM call

**Status:** Accepted — a design-time judgment call, explicitly not a measured
comparison. Documented as such rather than dressed up as more rigorous than
it is.

## Context

The hybrid pipeline's fusion step (RRF) needs a final reranking pass to turn
fused candidates into an ordered top-N. Two realistic options exist: a local
cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`), or asking an
LLM to rank or select from the candidate list directly.

## Decision

Use the local cross-encoder. Unlike ADR-0001 and ADR-0002, this was not
settled with a head-to-head measurement against the alternative — no
LLM-rerank variant was ever built to compare against. The reasoning behind
it:

- **Cost**: the cross-encoder runs locally and is free per query after a
  one-time ~90MB download; an LLM rerank call costs real tokens on every
  single query, on top of the generation call that already happens.
- **Determinism**: a cross-encoder's score for a given (query, chunk) pair is
  fixed; an LLM asked to rank the same candidates can reorder them between
  identical calls, which this project has already measured being a real
  problem in a different context — binding tools to a `temperature=0.7`
  model made tool *selection* nondeterministic (93.2%→91.5% between runs).
  Reranking sits in the same category of decision.
- **Latency shape**: the cross-encoder's cost is known and constant (the
  ablation in ADR-0002 measured it directly — ~700ms added for the reranked
  variant); an LLM call's latency is more variable and adds a second
  sequential network round-trip before generation can start.

## Consequences

- This is the one retrieval-architecture decision in the project that is
  asserted rather than measured, and it says so — the honest label matters
  more here than pretending otherwise, given how much of the rest of this
  project's credibility rests on not doing that.
- The cross-encoder's own cost is measured, even though the comparison isn't:
  ADR-0002's ablation table shows it adds latency without improving on plain
  dense search on this corpus, which is a real, measured strike against the
  reranking step generally — independent of which reranker implements it.
- If retrieval selection becomes worth doing again (a larger corpus, per
  ADR-0002), this is the natural point to also finally measure an
  LLM-rerank variant instead of assuming the reasoning above still holds.

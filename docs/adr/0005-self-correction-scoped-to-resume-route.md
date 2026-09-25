# 0005 — The self-correction retry is scoped to the resume route only

**Status:** Accepted

## Context

The faithfulness judge's score was computed every turn and used for exactly
one thing: deciding whether to show a warning banner or a refusal. The score
itself carried real information — specifically, whether the current answer
was well grounded — that was then thrown away. The question was where to
spend a retry, given one was worth building.

## Decision

Trigger a bounded retry (reformulate the query, re-run resume retrieval,
regenerate) only on the `resume` route, only when the score is below 4, and
at most once per turn — not on every route uniformly.

This is a structural scoping decision, not a convenience: `personal`,
`spotify`, and `linkedin` are fixed lookups (`get_personal_info_result` takes
an `info_type` from a small fixed set or none at all; `get_music_taste_result`
and `get_linkedin_info_result` take no arguments). Retrying one of them
returns byte-identical content every time — there is no lever a "reformulated
query" could pull, only a wasted judge call and a wasted generation call spent
reproducing the same answer. `conversation` retrieves nothing at all, so
there is no context to reformulate against in the first place. `resume` is
the only route with a free-text query a differently-worded retry could
plausibly change.

## Consequences

- Measured against the same baseline with no other change: routing and
  recall hold exactly, faithfulness moves −0.040 (within the project's own
  0.07 tolerance), and only two questions retried in a full 59-question run —
  one unanswerable (not RAGAS-scored), one multi-intent. The multi-intent
  case is the honest result worth keeping, not hiding: reformulating the
  resume query cannot supply `linkedin` data that classifier mode
  structurally never fetches (ADR-0001), so the retry correctly identified a
  bad answer and correctly tried the only lever available to it — and that
  lever was the wrong one for that specific failure. This is a limitation of
  `classifier` mode compounding with itself, not a bug in the retry.
- On the current 7KB corpus, `RETRIEVAL_STRATEGY=auto` already returns the
  whole document once it fits in context (ADR-0002), so a reformulated query
  is close to a no-op for retrieval purposes today — it changes rerank order
  at most. The mechanism is real and tested regardless of corpus size; its
  practical trigger rate is expected to matter more once the corpus outgrows
  that ceiling.
- The in-graph score that drives this retry (`check_faithfulness_node`) is
  deliberately kept separate from the outer, user-facing tiering
  (`guardrails.faithfulness_check.check_faithfulness`, still applied in
  `chatbot.py:main()` after the graph returns) — see ADR-0006 for why those
  two are not merged.

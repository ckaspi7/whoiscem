# 0006 — The faithfulness judge fails open; a tool error must never reach context

**Status:** Accepted

## Context

Two different failure modes exist in this project, and they are handled
differently on purpose: the faithfulness judge call can fail (a network
error, a malformed response), and a retrieval or tool call can fail (Qdrant
unreachable, a locked database file). Early in the project, a tool failure
was caught and returned as an ordinary string — `f"Error: {e}"` —
indistinguishable from real content once it reached the model. A Qdrant
outage became the assistant's own "evidence," and the faithfulness judge
scored an answer against a stack trace instead of skipping the check
entirely. That was the incident this decision exists to prevent from
recurring in either direction.

## Decision

Two different rules for two different kinds of failure, not one uniform
policy:

- **The judge fails open.** If `score_faithfulness`'s own API call throws,
  the result is `None` — "nothing to act on" — and the answer is returned
  unchanged, not blocked or replaced. A judge is a quality gate on top of an
  answer that already exists; a judge that is temporarily unavailable is a
  worse reason to refuse a user an answer than the small chance that one
  particular answer goes unchecked.
- **A tool error can never become evidence, with no exception.**
  `ToolResult.as_context()` returns `content` on success and `""` on
  failure — structurally, not by convention, since the two fields are
  different attributes on the same typed object, not a shared string a
  handler could accidentally conflate. `eval/run_eval.py` reports a
  dedicated tool-error rate, and `eval/compare.py` fails the gate on any
  nonzero value, checked directly rather than as a tolerance band, because
  there is no acceptable rate of a tool silently failing.

These are not the same rule applied inconsistently — they protect different
things. The judge is a check *on* an answer; failing open means a rare
skipped check. A tool result is the *evidence* an answer and its check are
both built on; letting a failure into that position would corrupt the answer
and silently defeat the judge checking it, which is strictly worse than
either failing alone.

## Consequences

- Phase 4.3 made the judge's own failure mode visible instead of silent: a
  failed judge call is now logged (`guardrails.faithfulness_check`, level
  WARNING) rather than vanishing with no signal anywhere — fail open still
  means "let the user through," not "pretend nothing happened."
- `tools/resume_tool.py`'s retrieval singletons are only committed to module
  state after setup fully succeeds, for the same reason in a different
  shape: a transient failure (a Qdrant container not yet ready) must be
  retried on the next call, never cached as a permanent, silent break for
  the rest of the process.
- Phase 3.3's self-correction retry reads `faithfulness_score` — a `None`
  from a failed judge call is treated identically to "nothing to act on," so
  a broken judge cannot accidentally trigger a retry loop; it can only ever
  suppress one.

# 0001 — Classifier stays the default agent architecture

**Status:** Accepted

## Context

The original router was a hardcoded five-way switch: an LLM call classified
the query into one label, and a fixed branch called exactly one tool. The
four `@tool`-decorated functions were never actually bound to an LLM — no
`bind_tools`, no tool-selection loop — so their docstrings, whose entire
purpose is to let a model choose between them, were dead weight. The
project's own README described this as "agents," which a switch statement
is not. The going-in assumption, reflected in the original project plan, was
that building a real tool-calling agent (`AGENT_MODE=tool_calling`) would
replace the classifier as the shipping default.

## Decision

Build the real agent, measure it head-to-head against the classifier on the
same golden set, and let the numbers decide — not the fact that one of them
is structurally more sophisticated.

| | classifier | tool_calling |
|---|---|---|
| Routing / tool-selection accuracy | **98.3%** | 91.5% |
| Faithfulness | **0.701** | 0.633 |
| Answer relevancy | 0.890 | **0.929** |
| Context recall | 0.745 | **0.776** |
| Multi-intent questions | structurally impossible | works, unprompted |

`classifier` stays the default. For a narrow-domain factual assistant, being
wrong or ungrounded is a worse failure than sounding slightly less relevant,
and it wins on both of those. `tool_calling` is kept as a fully-supported,
opt-in mode (`AGENT_MODE=tool_calling`) rather than deleted, because it has
two structural capabilities — multi-intent questions and follow-ups without a
condensation step — the classifier cannot have at any accuracy, and because
its accuracy gap is explained, not unexplained: two of four misroutes are a
pre-existing `linkedin`/`resume` boundary ambiguity that predates this
change, and the other two are the agent correctly treating an answer already
in its system prompt as not needing a redundant tool call.

## Consequences

- The README's own headline claim ("real agent") is backed by a measured
  comparison that does not flatter the agent, which is more credible than a
  clean win would have been.
- Every later change that touches routing (Phase 3.3's self-correction,
  Phase 3.5's trajectory eval, the `CHAT_MODEL` comparison) is built and
  measured against `classifier` first, since that is what ships.
- `CHAT_MODEL`'s own default (see ADR-0004) inherits this same discipline:
  gpt-6-luna measures better under `classifier` but breaks under
  `tool_calling`, and the same one-setting-can't-express-two-modes problem
  applies there too.
- A first, unpolished pass at `tool_calling` measured 81.4% routing, not the
  91.5% above — caused by premature refusal (treating the system prompt's
  "key facts" as a ceiling on what it knew), not hallucination. A single
  system-prompt addendum closed 12 of 14 points of that gap, which is why the
  comparison above reflects a real fix, not a first attempt.

# 0004 — `CHAT_MODEL` and `AGENT_MODE` are independent settings; gpt-6-luna is a recommended override, not a default

**Status:** Accepted

## Context

A newly available model, gpt-6-luna, measured cheaper (input $0.10 vs
$0.15/1M, ~87% cheaper cached input) and with a far larger context window
(1.05M vs 128K) than the incumbent gpt-4o-mini. A quick 8-question pilot
suggested it might also answer better. `CHAT_MODEL` did not exist yet as a
setting — the model was hardcoded in three places in `chatbot.py`.

## Decision

Add `CHAT_MODEL` as a setting orthogonal to `AGENT_MODE`, measure gpt-6-luna
against gpt-4o-mini on the full golden set under the shipping `classifier`
mode, and separately check what happens under `tool_calling` before touching
the default.

| | gpt-4o-mini (default) | gpt-6-luna |
|---|---|---|
| Routing accuracy (classifier mode) | 98.3% | **100.0%** |
| Faithfulness | 0.713 | **0.718** |
| Context recall | 0.745 | **0.771** |
| Answer relevancy | **0.873** | 0.862 |

gpt-6-luna matches or beats gpt-4o-mini on every gated metric under
`classifier`. It was not made the default anyway: the identical prompt and
addendum that scores this well under `classifier` misroutes basic questions
under `tool_calling` (40% routing on a reproduced 5-question sample) —
because `CHAT_MODEL` is one global setting applied to both graphs, flipping
it would silently hand that regression to anyone who sets
`AGENT_MODE=tool_calling` without separately remembering to set `CHAT_MODEL`
back. A default is an implicit claim of "best known configuration," and that
claim is only true for one of the two paths it would govern.

## Consequences

- `gpt-6-luna` is documented as a recommended, explicit override for
  `classifier`-mode deployments (`CHAT_MODEL=gpt-6-luna`), which is the
  realistic production configuration anyway given ADR-0001 — not silently
  left worse-than-necessary, just not defaulted.
- A compound default (gpt-6-luna under classifier, gpt-4o-mini under
  tool_calling) was considered and rejected: it would work, but it breaks the
  orthogonality the two settings otherwise have, for a marginal gain over a
  documented manual override. Simplicity was chosen over cleverness.
- Two real API constraints surfaced only by running gpt-6-luna, not
  documented anywhere in advance: it only supports tool/function calling at
  `reasoning_effort="none"`, and it rejects any non-default `temperature`
  outright. `create_assistant` handles both explicitly rather than let either
  surface as a live 400.
- The same "one setting, two graphs, don't let a win on one silently regress
  the other" reasoning that kept `classifier` the default in ADR-0001 governs
  this decision too — it is the same discipline applied to a second axis.

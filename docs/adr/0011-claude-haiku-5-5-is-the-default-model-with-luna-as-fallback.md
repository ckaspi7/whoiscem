# 0011 — Claude Haiku 5.5 is the default model, with gpt-6-luna as the fallback

**Status:** Accepted — measured against a `gpt-4o-mini` control on the same code,
corpus and judge. Single runs; the caveats below say how far that goes.

## Context

OpenAI retires every GPT-4o model on 2027-01-14, and `gpt-4o-mini` is this
project's chat model, session summariser and evaluation judge. The deprecation
notice names `gpt-6-luna` as the replacement. Anthropic's `claude-haiku-5-5` was
the other candidate, which also answers a second question: a model on a second
provider is the only fallback that survives an outage or an exhausted balance at
the first (a same-provider fallback does not).

Three things had to be true before a default could move, the same bar as
`retrieval_strategy` and `agent_mode` (ADR-0004): the models measured on one
golden set, through the real graph; the framework and the judge held fixed
(ADR-0010 landed first, so the stack did not change under the comparison); and
no silent switch able to contaminate a run (the evaluation builds no fallback).

## What was compared

`gpt-4o-mini` (control), `gpt-6-luna` at `reasoning_effort="none"`, and
`claude-haiku-5-5` with thinking disabled, so that none of the three reasons before
answering. 59 questions, `classifier` mode (the default), then `tool_calling`
mode for information. The RAGAS judge stayed `gpt-4o-mini` with `ragas` 0.4.3 for
every row, so a difference is the answering model's. Corpus hash, personal data
and code were identical across rows. Files: `eval/results/v16-*` (default mode),
`v17-*` (tool-calling, no RAGAS), `v18-*` (after the scope instruction), `v19-*`
(the committed baseline).

## Results — default (classifier) mode

| | gpt-4o-mini | gpt-6-luna | claude-haiku-5-5 |
|---|---|---|---|
| Routing | 98.3% | **100%** | 98.3% |
| Correct refusals | **100%** | 90% (u006) | **100%** |
| Faithfulness | 0.723 | 0.726 | **0.799** |
| Faithfulness, no conversation or Spotify | 0.849 | 0.851 | **0.883** |
| Answer relevancy | 0.896 | **0.904** | 0.768 |
| Answer relevancy, no Spotify | 0.882 | **0.902** | 0.862 |
| Context recall / precision | 0.714 / 0.666 | **0.745** / **0.677** | **0.745** / 0.663 |
| Self-correction retries (of 59) | 12 | 15 | 12 |
| Model cost, the whole run | $0.0180 | **$0.0135** | $0.0224 |

Conventions: `answer_relevancy` excludes the conversation route (it has no context
to be relevant to), the same exclusion `eval/compare.py` gates with.

**Haiku's relevancy gap is mostly the scorer, not the answers.** Its Spotify
relevancy is 0.15 against 0.99 and 0.92. The Spotify context states that the data
is a manual export from 18 months ago, and the project's freshness rule asks the
answer to say so. Haiku did on 6 of 6 Spotify answers, Luna on 5 of 6,
`gpt-4o-mini` on 0 of 6 — and RAGAS scores any answer that hedges as
"non-committal" and gives it 0. Without Spotify the three are within 0.04. We did
not reword the tool text to please that scorer: the caveat is the behaviour the
freshness rule exists to produce.

**Luna's one regression is real.** "Who won the 2026 World Cup?" has no
retrieval; Luna answered from its own knowledge in every run.

**Haiku's costs:** answers about 2.4 times longer (median 251 characters against
about 105), about 1.2 times the control's model cost (a fraction of a cent per
question for all three), and one missing faithfulness value where the judge's own
output overflowed on a long answer (r013).

**Latency is not a differentiator.** Model calls take 1 to 1.3 s for all three.
The local HHEM faithfulness check on the résumé route (7 to 15 s) dominates, and
varies with answer length and machine load more than with the model.

## Results — `tool_calling` mode (informational, no RAGAS)

| | gpt-4o-mini | gpt-6-luna | claude-haiku-5-5 |
|---|---|---|---|
| Routing | 86.4% | 91.5% | **100%** |
| Correct refusals | **100%** | 80% | 90% |
| Retrieval recall@k | 95.8% | 87.5% | **100%** |
| Model cost, the whole run | $0.0178 | **$0.0151** | $0.0317 |
| Tool calls per question | 1.07 | 1.14 | 1.47 |

Haiku called the résumé and LinkedIn tools together on 19 questions, which is why
its agent costs about 1.4 times its classifier run. The "multi-intent complete"
score (3 of 3, 1 of 3, 1 of 3) is an artifact of the metric, not of the models:
Luna and Haiku used the LinkedIn tool where the metric expects the résumé, and
their answers covered both parts. Luna at `reasoning_effort="none"` works in this
mode (91.5% routing); September's default-effort Luna scored 40%.

## Stability

Haiku's default-mode answering was repeated twice more (no RAGAS): routing 98.3%
with the same single misroute, refusals 100%, recall 100% and zero tool errors
each time; no question changed route between runs; the Spotify answers stated
the data's age every time. Retries were 12, 10 and 13, and the median answer length
moved between 215 and 282 characters. The repeats are not committed.

## The scope instruction

The system prompt now tells the model to answer only about Cem and this assistant,
decline anything else in one sentence, and never answer such a question from its
own knowledge. Measured over the same set (`v18-*`, default mode, RAGAS):

- Every model declines the World Cup question, Luna included. The refusal metric
  is a keyword heuristic and did not recognise "I can only help with questions
  about Cem", so it reported 0.90 for a correct refusal; the markers now include
  that phrasing, with a test.
- Quality did not move beyond run-to-run noise (Haiku faithfulness 0.799 to
  0.782, relevancy 0.768 to 0.792; Luna 0.726 to 0.762 and 0.904 to 0.895, same
  conventions as the table above).
- Haiku's answers got shorter (median 251 to 194 characters) and retries fell.
- In `tool_calling` mode it fixed the "developer mode" injection prompt (a002)
  for every model, at a cost: one false refusal ("What was NeoWise focused on?",
  `gpt-4o-mini`) and routing down by 1.6, 3.4 and 3.4 points, which may be noise or
  the agent skipping a tool it judges out of scope. Not gated, since that mode is
  not the default.

## Decision

- `CHAT_MODEL` defaults to **`claude-haiku-5-5`**. It has the best faithfulness
  (0.799, and 0.653 against 0.25 on the Spotify route), perfect refusals in both
  modes with the scope instruction, the best `tool_calling` routing, and it follows
  the freshness rule. The price is longer answers and about 1.2 times the control's
  model cost.
- `FALLBACK_CHAT_MODEL` defaults to **`gpt-6-luna`**, but only while `CHAT_MODEL` is
  the default: it is the cheapest, routes perfectly in default mode, and sits on
  the other provider. Setting `FALLBACK_CHAT_MODEL`, even to empty, always wins, so
  an explicit `CHAT_MODEL` never silently needs a second provider's key.
- `gpt-4o-mini` stays selectable until it retires.
- `models.py` is the single place for per-model rules (no `temperature` for Luna or
  Haiku; `reasoning_effort="none"` on every Luna call; thinking off and
  `max_tokens=2048` for Haiku; tools bound to each model before the fallback wraps
  them). A fallback logs a WARNING on every switch, and cost is priced per node at
  the model that actually answered.
- The retry bug found on the way is fixed: the self-correction regeneration sent
  the first answer as the last message, which Claude rejects ("assistant prefill")
  and which steered the regeneration for every model; the first attempt is now left
  out of the prompt.

## Consequences and what this does not establish

- **Single runs.** A single-run difference under about 0.04 is inside the noise this
  project has already measured (ADR-0010, LIMITATIONS). The ordering of faithfulness
  is the clearest signal; the rest is directional.
- **The judge is still `gpt-4o-mini`.** It must move before 2027-01-14. Every
  judged number here was scored by it, so the December change re-baselines by
  rescoring these same saved answers under the old and new judge
  (`eval/run_eval.py --rescore`) and measuring the shift directly.
- **Haiku cannot be pinned to a temperature**, and neither can Luna, so
  determinism is whatever the model gives at its default. Haiku's repeats were stable.
- **Longer answers.** If conciseness matters, that is a prompt change to measure,
  not a model argument.
- **`tool_calling` is ungated.** CI's gate runs the default mode only, the golden set
  has 3 multi-intent questions (too few to say whether a hybrid of the two modes
  would help), and the metric that would judge one needs fixing first.
- **A second key is now required.** The API and any host running the default
  configuration need `ANTHROPIC_API_KEY` as well as `OPENAI_API_KEY` (embeddings and
  the fallback), and `/healthz` checks for both.

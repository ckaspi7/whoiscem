# 0010 — Upgrade the LangChain stack, and move RAGAS with it

**Status:** Accepted — measured against an unchanged-model control run.

## Context

Every LangChain-family pin dated from March 2025: `langchain` 0.3.20,
`langchain-core` 0.3.63, `langgraph` 0.3.5, `langchain-openai` 0.3.8,
`openai` 1.65.5, `ragas` 0.2.14. Two facts made that a deadline, not a tidiness
problem:

- LangChain's release policy puts the 0.3 line in maintenance (security and
  critical fixes only) until December 2026, and LangGraph 0.3.5 is older than
  even the 0.4 line that policy names, so it was already unmaintained.
- OpenAI retires `gpt-4o-mini` on 2027-01-14, and one candidate replacement is
  Claude Haiku 5.5. The current `langchain-anthropic` needs `langchain-core`
  1.6 or newer, which the pinned stack could not accept.

The application's own surface on these libraries is small: `ChatOpenAI`,
`OpenAIEmbeddings`, `@tool`, `StateGraph` and `add_messages`, the core message
and prompt types, `SemanticChunker`, and RAGAS. It uses none of the legacy chains
or agent helpers that LangChain 1.x moved or removed.

## Decision

Upgrade the stack as its own change, before the model comparison, so the
framework change is measured with the model held fixed (`gpt-4o-mini` exists only
until January). No application code needed to change; the unit, app-boot and
integration suites pass as they were.

| Package | Before | After |
|---|---|---|
| `langchain` | 0.3.20 | 1.4.6 |
| `langgraph` | 0.3.5 | 1.2.16 |
| `langchain-openai` | 0.3.8 | 1.1.9 |
| `openai` | 1.65.5 | 1.109.1 |
| `ragas` | 0.2.14 | 0.4.3 |
| `langchain-community` | 0.3.19 | 0.4.1 |
| `langchain-experimental` | 0.3.4 | 0.4.1 |
| `langchain-text-splitters` | 0.3.6 | 1.1.3 |

Three of these are deliberately not the newest release:

- **`ragas` could not be held at 0.2.14.** It imports
  `langchain_community.chat_models.vertexai` when it loads, and `langchain-community`
  0.4.2 (2026-05-22) removed that module. The
  newest RAGAS (0.4.3, January 2026) still imports it, so the only working pairing is
  `ragas` 0.4.3 with `langchain-community` 0.4.1.
- **`langchain-experimental` 0.4.1** is the newest release that declares
  compatibility with `langchain-community` 0.4.1.
- **`openai` stays below 2**, and `langchain-openai` with it, because `ragas` pulls in
  `instructor`, which caps the SDK.

`langchain-community` and `langchain-experimental` are both being sunset upstream.
Only `retrieval/vectorstore.py`'s `SemanticChunker` (the PDF fallback) needs them.

The judge and its embeddings are now named in `eval/run_eval.py`
(`JUDGE_MODEL`, `JUDGE_EMBEDDING_MODEL`), set to what RAGAS 0.2 chose for itself
(`gpt-4o-mini`, `text-embedding-ada-002`). The judge is built with `llm_factory`
and the OpenAI SDK, as RAGAS 0.4's own default is. Wrapping a LangChain
`ChatOpenAI` instead went through a deprecated shim that timed out on the resume
route's long contexts.

## What was measured

The same code, model (`gpt-4o-mini`, classifier mode), corpus and personal data
were run on both stacks, 59 questions with the full RAGAS pass
(`eval/results/v14-pre-upgrade-gpt4o-mini.json` and
`v15-post-upgrade-gpt4o-mini.json`):

| | Old stack | New stack |
|---|---|---|
| Routing accuracy | 98.3% | 98.3% (same single misroute) |
| Retrieval recall@k / MRR | 1.0 / 0.555 | 1.0 / 0.524 |
| Refusals, tool errors | 100%, 0 | 100%, 0 |
| Faithfulness | 0.696 | 0.719 |
| Answer relevancy (conversation excluded) | 0.897 | 0.923 |
| Context recall | 0.796 | 0.735 |
| Context precision | 0.642 | 0.690 |
| Graph cost, 59 questions | $0.0176 | $0.0184 |

Every metric is inside `eval/compare.py`'s tolerance. The retrieved contexts are
identical (the `auto` strategy returns the whole résumé), so what moved is the
scoring, not the system.

## Consequences

- **Scores before and after are not strictly comparable.** The new RAGAS scored
  context recall 0 on `a003` and `m002` every time it was run on identical input,
  where 0.2.14 gave 1.0. Both, with `m001`, are the questions ADR-0009 already ruled
  ill-posed (their ground truths describe desired behaviour, not facts in the
  résumé), so the stricter reading is arguably the more correct one. It accounts for
  much of the resume route's recall drop (0.98 to 0.85), which leaves the aggregate
  within 0.01 of the 0.07 tolerance; the rest is the judge's noise, below.
- **The judge is noisy on identical input.** Repeated scoring of the same five rows
  moved `m001`, `m003` and `f001` between 0, 0.5 and 1. A single run's per-question
  value is not stable; only aggregates are gated.
- **`answer_relevancy` samples fewer generations.** RAGAS logs "LLM returned 1
  generations instead of requested 3" under the new stack, so that metric averages
  fewer sampled questions than before. Per-row values still matched within 0.07.
- **The committed baseline moves to the new run**, because `eval/compare.py` picks
  the newest full result. The pre-upgrade run stays as the record of the A/B.
- **Not done here, and still due before 2027-01-14:** the judge model itself.
  `gpt-4o-mini` retires with the chat model, so the judge moves in its own change,
  with the same rows scored by both judges to measure the shift.
- **Follow-up:** replace the `SemanticChunker` fallback so `langchain-community` and
  `langchain-experimental` can be dropped.

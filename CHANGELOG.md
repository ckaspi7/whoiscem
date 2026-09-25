# Changelog

Every entry here is a number from a committed file in [`eval/results/`](eval/results/),
not a recollection — the trend *is* the portfolio. This follows the phased
rebuild plan (Phase 0 through 5); see [README.md](README.md) for the full
narrative behind each finding and [LIMITATIONS.md](LIMITATIONS.md) for what
still isn't fixed.

---

## Phase 0 — Security & truth

The starting condition: committed PII (a real phone number and email, inside
a PDF), a red CI, and README claims — cached data "refreshed monthly," RAGAS
scores that had never been computed — that didn't match the code.

- Purged committed PII with a fresh history; `tests/unit/test_no_pii_committed.py`
  now fails the build if it ever comes back, including inside a binary file.
- `RESUME_PATH` defaults to a redacted `data/resume.md` (phone and email
  removed); the real, unredacted resume is indexed locally via an override.
- CI made green and honest: lint, unit tests, integration tests, an image
  build, and a compose healthcheck, on every push.
- LinkedIn/Spotify caches stopped asserting a refresh cadence they didn't
  have — the tools now report the cache's real age instead.

## Phase 1 — Make the evaluation real

Before this, "evaluation" called the tools directly with the golden set's own
`required_tool`, which meant it could not see a routing mistake, a prompt
change, or the faithfulness guard — it measured a pipeline that was not the
one deployed.

- `eval/run_eval.py` rebuilt to run the golden set through the shipping graph:
  router, handler, production prompt, faithfulness guard.
- First committed baseline: 59 golden questions, routing 91.5%, recall@k
  83.3%, faithfulness 0.660, answer relevancy 0.834.
- `eval/compare.py` gates every later change against the most recent
  committed result — a regression beyond tolerance fails the build.

## Phase 2 — RAG correctness, then measurable improvement

| | Phase 1 baseline | Phase 2 | Δ |
|---|---|---|---|
| Routing accuracy | 91.5% | **98.3%** | +6.8 |
| — follow-up questions specifically | 33% | **100%** | +67 |
| Retrieval recall@k | 83.3% | **100.0%** | +16.7 |
| Answer relevancy | 0.834 | **0.873** | +0.039 |
| Faithfulness | 0.660 | 0.688 | +0.028 |
| Context recall | 0.674 | 0.776 | +0.101 |
| Context precision | 0.636 | 0.651 | +0.014 |

- Fixed a silent correctness bug in hybrid fusion: BM25 was keyed on list
  position, dense search on point ID — the join was matching the wrong
  chunks to each other.
- Added history-aware query rewriting (`condense_query`) — the single highest-
  leverage retrieval change, closing follow-up routing from 33% to 100%.
- **The ablation this project is built around found that hybrid retrieval
  loses to plain dense search on this corpus** — worse recall, worse MRR,
  4.4× the latency. BM25 has too few documents to separate anything on a
  one-page resume, and the cross-encoder spends 700ms/query demoting chunks
  dense search had already ranked first. Kept, selectable, not removed: the
  finding is that the apparatus was assembled before anything measured
  whether it helped, and the ranking is a property of a 7KB corpus, not a
  law.
- Retrieval and embedding caching cut a re-embedded query from 215ms to
  1.7ms, with no regression on any metric.
- Three of four RAGAS metrics still below their own published thresholds,
  published anyway, with the per-route faithfulness breakdown explaining
  why (list-formatted tool output scores near zero by construction; the
  `conversation` route has no context to be faithful to at all).

## Phase 3 — Genuine agency

### 3.1 — A real tool-calling agent, measured against the switch it replaces

| | classifier (default) | tool_calling |
|---|---|---|
| Routing / tool-selection accuracy | **98.3%** | 91.5% |
| Answer relevancy | 0.890 | **0.929** |
| Faithfulness | 0.701 | 0.633 |
| Context recall | 0.745 | **0.776** |
| Multi-intent questions | structurally impossible | works, unprompted |

A genuinely mixed result. `classifier` stays the default — for a narrow-domain
factual assistant, wrong or ungrounded is worse than slightly less relevant —
but `tool_calling` has two structural capabilities the classifier cannot have
at any accuracy. A first pass scored 81.4% from premature refusal, not
hallucination; one system-prompt addendum closed 12 of 14 points of that gap.
Binding tools to a `temperature=0.7` model (chosen for prose variety) made
tool selection itself nondeterministic (93.2%→91.5% between identical runs);
a dedicated `temperature=0` model for the agent's own decisions fixed it.

### 3.4 — `GraphState.messages` becomes real messages

A live streaming generator used to sit directly in message content mid-turn —
unserializable state, a node whose own latency measured ~0s because the real
work happened later wherever the generator was drained, and `isinstance`
guards scattered across three modules. Fixed by construction, not handled
defensively: real `BaseMessage` objects reject non-string content before it
can reach state. No RAG-quality change expected or found — verified by full
routing/retrieval parity (98.3% / 100%) after the change.

### 3.3 — Self-correction: the faithfulness score stops being thrown away

| | before | after | |
|---|---|---|---|
| Routing accuracy | 98.3% | 98.3% | — |
| Faithfulness | 0.713 | 0.673 | −0.040 (within the 0.07 tolerance) |
| Answer relevancy | 0.873 | **0.887** | +0.014 |
| Context recall | 0.745 | **0.776** | +0.031 |

A score below 4 on the resume route now triggers one bounded retry
(reformulate the query, re-search, regenerate) instead of only gating a
banner. Gates clean; the one metric that moved is fully attributable to one
already-known limitation (a multi-intent question needing `linkedin` data
that classifier mode never fetches), not a new regression — the retry
correctly identified a bad answer and correctly tried the only lever
available to it.

### 3.5 — Trajectory evaluation

`route_correct` only requires *some* acceptable category be touched — a
multi-intent question calling one of two needed tools still passes that
metric. Trajectory evaluation checks the stronger claim: 3/3 multi-intent
questions called every tool they needed, at 1.07 average tool calls per turn,
with the only two "unnecessary" calls found being already-known misroutes,
not a new over-calling problem.

### CHAT_MODEL — gpt-6-luna measured against gpt-4o-mini

| | gpt-4o-mini (default) | gpt-6-luna |
|---|---|---|
| Routing accuracy (classifier mode) | 98.3% | **100.0%** |
| Faithfulness | 0.713 | **0.718** |
| Context recall | 0.745 | **0.771** |
| Answer relevancy | **0.873** | 0.862 |

Matches or beats gpt-4o-mini on every gated metric under `classifier` mode,
plus ~87% cheaper cached input and an 8x larger context window — but the
identical prompt misroutes basic questions under `tool_calling` mode (40% on
a reproduced sample), so the global default did not move: one setting can't
express "good under one mode, broken under the other." Recommended as an
explicit override for `classifier`-mode deployments. Two undocumented API
constraints found by running it: it only supports tool calling at
`reasoning_effort="none"`, and it rejects any non-default `temperature`
outright.

## Phase 4 — Production hardening (in progress)

- **4.1 — Headless API.** `api.py`: `/chat` and a real `/healthz`, auth, a
  sliding-window rate limit, and a daily spend cap enforced before the graph
  runs. Built for headless access (load testing, trajectory eval), not yet
  what the deployed Streamlit app calls — see LIMITATIONS.md for that
  boundary.
- **4.2 — Real cost accounting.** Replaced `len(text) // 4` priced at the
  output rate with real per-call token usage, applied at each model's actual
  rate. Found two undocumented gotchas by running it: LangChain's
  `get_openai_callback()` does not propagate through LangGraph's node
  execution, and `ChatOpenAI(streaming=True)` returns no usage data at all
  unless `stream_usage=True` is also set.
- **4.3 — Guardrails.** The judge call now requests
  `response_format={"type": "json_object"}` and logs instead of silently
  failing open; an oversized chat input is rejected before it costs anything.
  A heuristic prompt-injection filter was deliberately not added — the golden
  set's adversarial cases already measure 100% under the model's own
  judgment, so there was no measured gap to close, only unmeasured
  false-positive risk to add.
- **4.4 — Structured logging.** JSON logs correlated by session id via a
  context var, replacing a default handler that had never been configured at
  all.
- **4.5 — Session id signing.** `?sid=` used to be accepted as any string, no
  check the server had issued it. Now HMAC-signed; a fabricated or edited id
  is rejected and a fresh session issued instead.

Outstanding: 4.6 (deploy to Qdrant Cloud + a persistent Redis, so the
hybrid-RAG path the README describes is actually reachable from a public
deployment) needs real cloud credentials and a live-deployment decision that
aren't this project's to make unilaterally.

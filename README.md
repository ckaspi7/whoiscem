# HowToCem

A personal AI chatbot built to answer questions about Cem Kaspi — and to demonstrate production-grade AI engineering skills in RAG, agents, observability, and evaluation.

[Limitations](LIMITATIONS.md) — what is cached, what is estimated, and what is not measured yet.

> **Live demo:** offline. It deployed from a repository that has been deleted, and
> redeployment is pending the move to hosted Qdrant. Run it locally in the meantime —
> setup below is three commands and needs no infrastructure.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│ route_query │  LLM classifies intent:
│             │  resume / personal / spotify / linkedin / conversation
└──────┬──────┘
       │
  ┌────▼────────────────────────────────────────────┐
  │  Handler node (one of five)                      │
  │                                                  │
  │  resume → Hybrid Retrieval Pipeline:             │
  │    ├─ Dense search  (Qdrant + OpenAI embeddings) │
  │    ├─ Sparse search (BM25)                       │
  │    ├─ RRF fusion    (top-20 from each)           │
  │    └─ Cross-encoder rerank → top-3 chunks        │
  │                                                  │
  │  personal → SQLite user database                 │
  │  spotify  → Cached Spotify snapshot, real age reported │
  │  linkedin → Cached LinkedIn snapshot, real age reported │
  └────┬────────────────────────────────────────────┘
       │
  ┌────▼──────────────┐
  │ generate_response │  GPT-4o-mini streams the answer
  └────┬──────────────┘
       │
  ┌────▼─────────────────┐
  │ faithfulness_check   │  GPT-4o-mini judges answer grounding (score 1–5)
  │                      │  ≥4 → pass  |  2–3 → warn  |  1 → refuse
  └────┬─────────────────┘
       │
  ┌────▼──────────────────────────────────────────┐
  │ Redis session memory                          │
  │ Rolling 3-sentence summary stored per UUID   │
  │ Returning visitors get prior context injected │
  └───────────────────────────────────────────────┘
```

The diagram shows `AGENT_MODE=classifier`, the default — `generate_response`
and the single-tool handler above become a real tool-calling agent under
`AGENT_MODE=tool_calling`, and `faithfulness_check`'s score now drives one
bounded retry on the resume route rather than only gating the banner shown
above. Both are measured changes, detailed below, not shown here to keep this
diagram to the shape most traffic actually takes.

### Headless API (Phase 4.1)

`api.py` is a second surface onto the same graph, independent of the
Streamlit UI: `POST /chat` and a real `GET /healthz` (actual dependency
checks — Qdrant, the OpenAI key — not just "the process is up"), with auth
(`API_KEY`, optional — unset logs a loud warning rather than failing closed,
for local dev), a sliding-window rate limit, and a daily spend cap tracked in
`SpendTracker` (`spend_tracker.py`, Redis-backed like session memory) using
the same real per-call usage `cost.py` computes for the Streamlit sidebar.
Built for headless access — load testing and Phase 3.5's trajectory eval
without driving a browser — not yet as what the deployed Streamlit app
itself calls: see **What I'd build next** and `LIMITATIONS.md` for exactly
what that split does and does not protect today.

```bash
uvicorn api:app --reload
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "Where does Cem work?"}'
```

---

## Measured Results

First committed baseline: 59 golden questions through the shipping graph —
router, handler, production prompt and faithfulness guard — not through a
test harness that calls the tools directly. Every run is committed under
[`eval/results/`](eval/results/) with the git sha and a hash of the indexed
resume, because scores only compare across runs over the same corpus.

| | Baseline | Current | |
|---|---|---|---|
| Routing accuracy | 91.5% | **98.3%** | +6.8 |
| — follow-up questions | 33% | **100%** | +67 |
| Retrieval recall@k | 83.3% | **100.0%** | +16.7 |
| Correct refusals | 100% | **100%** | — |
| Answer relevancy | 0.834 | **0.873** ✓ | +0.039 |
| Faithfulness | 0.660 | 0.688 ✗ | +0.028 |
| Context recall | 0.674 | 0.776 ✗ | +0.101 |
| Context precision | 0.636 | 0.651 ✗ | +0.014 |

Every run is committed, so these are diffs between files in `eval/results/`
rather than remembered numbers.

Three of four RAGAS metrics are still below threshold, and they are published
anyway. The one uncomfortably-close miss is context recall at 0.776 against a
0.80 bar — closing that gap is Phase 2's next unfinished item, not a rounding
error to wave away.

**Faithfulness is not one number.** Per route:

| Route | Faithfulness | |
|---|---|---|
| linkedin | 0.900 | |
| resume | 0.892 | the hybrid retrieval path |
| personal | 0.630 | |
| spotify | 0.125 | context is a numbered list |
| conversation | 0.000 | no context exists on this path |

The aggregate is dragged down by two measurement artifacts rather than by
hallucination. A probe isolating it: with prose context, the judge scores a
correct answer 1.0 and a wrong one 0.0; with the same fact in a numbered list
(`1. The Weeknd`), it scores the correct answer 0.0, because inferring "top
artist" from list position is not entailment. And the `conversation` route has
no retrieved context at all, so nothing on it can be grounded — which is also
why a misroute onto it is the worst failure the system has.

### Retrieval ablation

Which parts of the pipeline earn their place, measured against the golden
references with routing excluded ([`eval/ablate_retrieval.py`](eval/ablate_retrieval.py)):

| variant | recall@k | MRR | context chars | ms/query |
|---|---|---|---|---|
| **dense only, k=5** | **100.0%** | **0.619** | 2,212 | 215 |
| dense only, k=3 | 79.2% | 0.569 | 1,202 | 286 |
| sparse only (BM25), k=5 | 66.7% | 0.324 | 1,924 | 0.1 |
| RRF fusion, k=5 | 83.3% | 0.484 | 2,291 | 237 |
| RRF + cross-encoder, k=5 | 91.7% | 0.540 | 2,395 | 940 |
| RRF + cross-encoder, k=8 | 95.8% | 0.547 | 3,933 | 1,054 |
| whole document | 100.0% | 0.550 | 7,580 | 1,216 |

**The hybrid pipeline this project is built around loses to plain vector
search on this corpus** — worse recall, worse MRR, and 4.4× the latency. BM25
contributes little because a one-page resume has too few documents for term
frequency to separate anything, and the cross-encoder spends 700 ms per query
demoting chunks that dense search had already ranked first.

Both are kept and selectable by `RETRIEVAL_STRATEGY`, because that ranking is a
property of a 7 KB corpus rather than a law. The finding is the point: the
apparatus was assembled before anything measured whether it helped.

**And on this corpus, retrieving at all is a net loss.** Grounding improves
monotonically with the amount of context returned, because a model answers from
more than the single snippet a question references. The default strategy is
therefore `auto`: pass the whole corpus while it fits in a prompt, and start
selecting when it does not.

### Caching

The table above measures each strategy's first-call cost. It also exposed a
real inefficiency: running all eleven variants over the same 24 questions
re-embeds the identical query text up to nine times, since every RRF and
reranked variant recomputes the fused candidate list — and therefore the query
embedding — from scratch.

`retrieval/cache.py` adds a query-embedding cache and a full-retrieval-result
cache, Redis when `REDIS_URL` is set and reachable, an in-process fallback
otherwise — the same degrade-never-raise pattern as session memory. Retrieval
entries are scoped by the corpus fingerprint, so a rebuilt index can never
serve a stale answer. Re-running the ablation with it enabled:

| variant | before | after |
|---|---|---|
| dense only, k=5 (same query as a prior variant) | 215 ms | **1.7 ms** |
| RRF fusion, k=5 | 237 ms | **1.4 ms** |
| RRF fusion, k=8 | — | **1.3 ms** |

The cross-encoder pass itself is not cached — reranking a fresh candidate set
is the point of asking again with different parameters — so the
`rrf_rerank@*` and whole-document variants are largely unchanged (~800–940 ms):
caching removed the redundant embedding call, not the reranking cost. Full
evaluation after enabling it shows no regression against the prior baseline on
any metric.

### A tool failure never becomes evidence

Every tool used to catch its own exceptions and return the equivalent of
`f"Error: {e}"` as an ordinary string — indistinguishable from real content
once it reached the model. A Qdrant outage became the assistant's "evidence",
and the faithfulness judge scored the answer against a stack trace instead of
skipping the check. `tools/result.py` introduces a typed result with an
explicit `ok`/`error` split; every handler now keeps an error out of both the
prompt and the judge, and logs it instead. `eval/run_eval.py` reports a
dedicated tool-error rate, and `eval/compare.py` fails the gate on any nonzero
value — checked directly, not as a tolerance band, because there is no
acceptable rate of a tool silently failing.

This closed a real incident rather than a hypothetical one: a scheduled
evaluation run completed but scored far enough below tolerance to fail the
gate, consistent with a fresh Qdrant container racing the first query and a
retrieval singleton that, before this fix, cached that failure for the rest of
the process instead of retrying. `tools/resume_tool.py` now only commits its
singletons to module state after setup fully succeeds.

### Messages are real messages now

`generate_response` used to call `llm.stream(...)` and put the raw, live
generator directly into `GraphState.messages` — the type annotation said
`list[dict[str, str]]`, which was never true for the turn's own answer while
it was in flight. That made state unserializable (no checkpointer was
possible), made this node's own measured latency read as ~0s (the real work
happened later, wherever the caller drained the generator), and needed
`isinstance(..., str)` guards scattered across three modules to cope.

Messages are now real LangChain `BaseMessage` objects under `add_messages` —
the class of bug is gone by construction, not handled defensively: a
non-string, non-list `content` is rejected by the message's own validation
before it can reach state. `generate_response` calls `.invoke()`, a plain
blocking call returning one serializable message; the live token-by-token
typing effect in the UI comes from `graph.stream(state, stream_mode=
["messages", "values"])` instead, which surfaces the same call's streaming
callbacks regardless of whether the node itself awaited `.invoke()` or
`.stream()` — and hands back the full final state in the same pass, so no
second call or checkpointer is needed just to read it back.

No RAG-quality change is expected or claimed from this — it is a structural
fix, verified by full routing/retrieval parity (98.3% / 100%) and RAGAS
movement within the noise band already established between identical-code
runs (±0.02-0.03).

### A real agent, measured against the switch statement it replaces

The four `@tool`-decorated functions were never bound to an LLM — no
`bind_tools`, no tool-selection loop. Their docstrings, whose entire purpose is
to let a model choose between them, were dead weight under a hardcoded
five-way switch. `AGENT_MODE=tool_calling` binds them for real: the model picks
zero-to-many tools, a loop executes them and feeds the results back, and it
answers once it stops calling anything. `AGENT_MODE=classifier` is the original
design. Both are measured, not one replacing the other on faith:

| | classifier (default) | tool_calling |
|---|---|---|
| Routing / tool-selection accuracy | **98.3%** | 91.5% ↓ |
| Answer relevancy | 0.890 | **0.929** ↑ |
| Faithfulness | 0.701 | 0.633 ↓ |
| Context recall | 0.745 | **0.776** ↑ |
| Multi-intent ("compare his resume to his LinkedIn") | structurally impossible | **calls both tools, unprompted** |
| Follow-ups without a condensation step | n/a — needs one | **works** (see below) |

A genuinely mixed result, not a clean win either way. `classifier` stays the
default: for a narrow-domain factual assistant, being wrong or ungrounded is a
worse failure than sounding slightly less relevant, and it wins on both. But
`tool_calling` has two structural capabilities the classifier cannot have at
any accuracy: multi-intent questions and natural follow-up handling, verified
below.

**Where the accuracy gap comes from**, per the golden set, not guessed at: two
of the four misroutes are the same pre-existing `linkedin`/`resume` boundary
ambiguity documented earlier; the other two ("Where is Cem from originally?",
"What year was Cem born?") are the agent correctly recognizing the answer is
already in its system prompt's tone-setting "key facts" and skipping a
redundant tool call — which is reasonable efficiency but means that answer is
unretrieved and ungrounded, exactly the "facts hardcoded in the prompt are
invisible to the faithfulness judge" issue this project already tracks.

**A first pass measured worse — 81.4% routing** — because the agent, given no
stronger instruction, treated the absence of a fact from those same "key facts"
as license to guess or decline rather than check a tool. Not hallucination —
it never invented an answer, it said "I don't have information" for things
like NeoWise that the resume tool clearly has — but premature refusal. A single
system-prompt addendum (`_TOOL_CALLING_ADDENDUM` in `chatbot.py`) — "always
check a tool before answering or declining" — closed 12 of 14 points of that
gap. This is also why the two remaining numbers above are the measured floor
of a real fix, not an unpolished first attempt.

**No `condense_query` node in this graph**, unlike the classifier, and
deliberately: the agent sees full conversation history natively when deciding
which tool to call and with what arguments, so it can resolve "How long was
that?" into a well-formed `get_resume_info(query="How long was Cem at
NeoWise?")` call by itself. Measured against the same follow-up golden cases
the classifier needed an explicit condensation step to pass, it does, with one
fewer moving part.

**Multi-intent arrived as a side effect of building this properly, not as
separate work.** One `AIMessage` can carry more than one `tool_call`, and the
execution loop already handles that — "Compare his resume to his LinkedIn"
measurably calls `get_resume_info` and `get_linkedin_info` in the same turn.

**One measurement caveat, not a result:** MRR and context precision both look
sharply better under `tool_calling` (0.566→1.000, 0.646→0.806) — this is an
artifact, not retrieval improving. `get_resume_info_result` pre-merges every
chunk into one block before it reaches state, so if the reference is found at
all it is trivially "rank 1" of one. `eval/compare.py` knows this and reports
both directions as uninterpreted rather than gating or celebrating either.

A `temperature=0` model drives the agent's own tool-selection, not the
`temperature=0.7` model used for the classifier's final prose — measured
directly: reusing the 0.7 model for tool selection made *which tool got
called* nondeterministic (routing moved 93.2%→91.5% between two runs of
identical code), a worse property for an agent's decisions than it is for a
chat reply's phrasing.

### Self-correction: the faithfulness score stops being thrown away

The judge score existed for one purpose before this: decide whether to show a
warning banner. Phase 3.3 gives it a second job — on the resume route, a score
below 4 now triggers one bounded retry (reformulate the query, re-run
retrieval, regenerate) before falling through to that same banner.

Scoped to `resume` specifically, not every route, because it's the only one
with a lever a retry can actually pull: `personal`, `spotify`, and `linkedin`
are fixed lookups — the same `info_type`, or no argument at all, every time —
so retrying would spend a judge call and a generation call to reproduce the
exact answer already given. `conversation` has no retrieved context to
reformulate in the first place.

Measured against the same classifier + gpt-4o-mini baseline, no other change
(`v10-gpt4o-mini-baseline.json` → `v12-self-correction.json`):

| | before | after | |
|---|---|---|---|
| Routing accuracy | 98.3% | 98.3% | — |
| Faithfulness | 0.713 | 0.673 | −0.040 (within the 0.07 tolerance) |
| Answer relevancy | 0.873 | **0.887** | +0.014 |
| Context recall | 0.745 | **0.776** | +0.031 |
| Context precision | 0.646 | 0.650 | +0.004 |

Gates clean, and the one metric that moved is worth explaining rather than
waving away. Only two questions triggered a retry this run — one unanswerable
(not scored by RAGAS at all) and one multi-intent ("Compare his resume to his
LinkedIn — do they agree?"). Reformulating the resume query cannot fix an
answer that is also missing `linkedin` data, which classifier mode
structurally never fetches — the retry correctly detected a bad answer and
correctly tried the only lever it has, and that lever was the wrong one for
this specific failure. For scale: `spotify`'s faithfulness moved by −0.167
between these same two runs from ordinary judge-call noise, on a route
self-correction never touches — the resume route's −0.056 move is smaller than
that, and fully attributable to the one case above, not a new regression.

### Trajectory evaluation: the right tools, not just an acceptable one

Phase 3.5. `route_correct` only checks that *some* acceptable category was
touched — a multi-intent question calling just one of the two tools it needs
still counts as correct under that metric, because one of them is on
`acceptable_routes` even if the other was never called. Trajectory evaluation
checks the stronger claim: every category the question needed, and no calls
beyond what it needed.

Real, measured (`eval/results/v13-trajectory.json`, tool_calling +
gpt-4o-mini — a routing-only pass, since trajectory shape doesn't need RAGAS's
judge calls to evaluate):

- **Multi-intent completeness: 3/3.** Every multi-intent question called every
  tool it needed, not just one — confirming Phase 3.2's capability holds up
  under the stricter check, not only under a metric a half-answered comparison
  could still pass.
- **1.07 average tool calls per turn.** Lean by default, not padded with
  speculative calls.
- **Two "unnecessary" calls, both already-known misroutes.** `l002`/`l003`
  called `resume` instead of `linkedin` — the same confusion already visible
  in routing accuracy, not an independent over-calling problem. Nothing in
  this run called a tool beyond what routing accuracy already flags as a
  mistake.

### Cheaper and slightly better, measured: gpt-6-luna as a classifier-mode override

`CHAT_MODEL` swaps the model behind routing, condensation, and generation in
both graphs, independent of `AGENT_MODE`. `gpt-4o-mini` is the default; same
golden set, same graph, only the model changed:

| | gpt-4o-mini (default) | gpt-6-luna |
|---|---|---|
| Routing accuracy (classifier mode) | 98.3% | **100.0%** ↑ |
| Faithfulness | 0.713 | **0.718** ↑ |
| Answer relevancy | **0.873** | 0.862 ↓ |
| Context recall | 0.745 | **0.771** ↑ |
| Context precision | 0.646 | 0.647 — |
| Retrieval recall@k / MRR | 100% / 0.566 | 100% / 0.566 — |
| Correct refusals | 100% | 100% — |

Matches or beats gpt-4o-mini on every gated metric in this run — plus ~87%
cheaper cached input ($0.01 vs $0.075/1M) and a 1.05M-token context window
against 128K. Retrieval numbers are identical, as expected: recall@k and MRR
depend on the corpus and embeddings, not on which model answers.

**Why the default didn't move despite a clean win.** `CHAT_MODEL` is
orthogonal to `AGENT_MODE` — one setting, applied to both graphs. Under
`tool_calling`, the *same* model, prompt, and addendum that scores this well
under `classifier` misroutes basic questions to `get_linkedin_info` instead of
`get_resume_info` (40% routing on a 5-question sample, reproduced twice — see
`LIMITATIONS.md`). Flipping the global default would silently hand that
regression to anyone who sets `AGENT_MODE=tool_calling` without also
remembering to set `CHAT_MODEL` back. Same discipline as `retrieval_strategy`
and `agent_mode`: a default only moves on an unqualified before/after, and
this one only clears that bar for one of the two paths it would govern.
Recommendation, not default: set `CHAT_MODEL=gpt-6-luna` explicitly for a
`classifier`-mode deployment, which is the realistic production config anyway.

**Two real API constraints, found by running it — not documented anywhere in
advance:**
- Tool/function calling only works via Chat Completions at
  `reasoning_effort="none"`; anything else breaks tool selection silently.
  `create_assistant` sets it automatically, only for the tool-bound agent —
  `llm`/`llm_fast` never bind a tool, so it doesn't apply to them.
- Any non-default `temperature` is rejected outright — a live 400: *"Unsupported
  value: 'temperature' does not support 0.0 with this model. Only the default
  (1) value is supported."* This includes the `temperature=0` this project
  otherwise relies on for deterministic routing and tool selection.
  `create_assistant` omits the override for this model rather than crash, so
  whatever routing determinism it has at temperature=1 is what the table above
  measures over a single run — not yet confirmed stable across repeats the way
  gpt-4o-mini's temperature-driven nondeterminism was originally caught
  (93.2%→91.5% between two identical runs, see above).

**A refusal-rate scare that wasn't one.** The first full run showed gpt-6-luna
at 90% correct refusals against gpt-4o-mini's 100%, with one salary question
"answered anyway." The actual text: *"Cem's salary at TELUS isn't listed in
the information I have"* — a correct decline, just phrased in a way the
keyword heuristic hadn't seen yet, the same class of gap `_REFUSAL_MARKERS`
has hit before with "wasn't able to find" and "without disclosing." Fixed the
heuristic, not the narrative: both models refuse at 100% once the detector
actually recognizes the phrasing.

### Known-broken, on purpose

Measured first so the fix can be reported as a delta rather than asserted:

- **The corpus is too small for retrieval to mean anything.** See the ablation
  above: the honest fix is more documents, not more retrieval machinery.

```bash
python eval/run_eval.py --output eval/results/$(git rev-parse --short HEAD).json
python eval/run_eval.py --no-ragas      # routing only, no judge calls
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini (default) or gpt-6-luna, via `CHAT_MODEL` |
| Agent framework | LangGraph — classifier or tool-calling agent, via `AGENT_MODE` |
| Vector store | Qdrant — embedded, self-hosted or cloud, selected by config |
| Dense retrieval | OpenAI text-embedding-3-small |
| Sparse retrieval | BM25 (rank-bm25) |
| Score fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free) |
| Chunking | Semantic chunker (LangChain Experimental) |
| Observability | In-app latency panel; distributed tracing not currently active (see [Limitations](LIMITATIONS.md)) |
| Session memory | Redis when configured, in-process fallback otherwise |
| Hallucination guard | GPT-4o-mini faithfulness judge (post-generation), plus one bounded self-correction retry on the resume route |
| UI | Streamlit |
| Personal DB | SQLite |
| CI | GitHub Actions (lint, format, unit + integration tests, image build, compose healthchecks) |

---

## Setup

**Prerequisites:** Python 3.11 and an OpenAI API key. No containers, no services.

```powershell
git clone https://github.com/ckaspi7/whoiscem.git
cd whoiscem

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env              # fill in OPENAI_API_KEY
streamlit run chatbot.py            # http://localhost:8501
```

The resume index is built on first query from `data/resume.md` and cached on
disk; the cross-encoder downloads ~90 MB the first time it runs. Point
`RESUME_PATH` at any Markdown or PDF file to index your own.

Optionally, seed the personal-info database that backs the `personal` route
(the file it reads from is git-ignored):

```powershell
copy data\seed_data.example.json data\seed_data.json   # fill in your data
python scripts/setup_user_data_db.py
```

### Backend modes

The vector store and session memory are selected by configuration, not wired in.
The same code runs against a local directory, a container, or a managed service.

| `QDRANT_MODE` | Backing store | Used for |
|---|---|---|
| `embedded` *(default)* | On-disk Qdrant at `QDRANT_PATH` | Local dev, CI, tests — no server |
| `server` | Qdrant over HTTP (`QDRANT_HOST`/`QDRANT_PORT`) | docker-compose / Podman |
| `cloud` | Qdrant Cloud (`QDRANT_URL` + `QDRANT_API_KEY`) | Deployed app |

Session memory follows the same pattern: Redis when `REDIS_URL` is set, an
in-process store otherwise. The active backend for both is shown in the sidebar.

Embedded mode takes an exclusive lock on its storage directory, so one process
at a time — stop the app before running the test suite against the same path.

### Run with containers instead

```bash
docker-compose up      # Qdrant + Redis + the app, with QDRANT_MODE=server
```

### Run tests

```powershell
pytest tests/unit/ -v          # fully mocked, no services, no API calls
pytest tests/integration/ -v   # embedded Qdrant; set QDRANT_MODE=server to use a container
python eval/run_eval.py --output eval/results/v2_hybrid.json
```

### Refresh data cache

```bash
# Fetches live Spotify data (requires SPOTIFY_* keys in .env)
# Also bumps LinkedIn cache timestamp
python scripts/refresh_cache.py
```

---

## What I'd build next

- **Qdrant Cloud:** Replace the Docker instance with Qdrant Cloud for zero-cold-start Streamlit Cloud deploys
- **Wire Streamlit as the API's client:** `api.py` exists — real auth, rate limiting, a daily spend cap — but the deployed Streamlit app still calls the graph directly and inherits none of it; making it a client is a deployment-topology decision, not made yet
- **Guardrails fully in the graph:** the self-correction score already lives there (Phase 3.3); the user-facing disclaimer/refusal banner still applies after streaming completes, not before
- **Multi-modal resume parsing:** Index embedded tables and charts from the PDF, not just raw text
- **Feedback loop:** Thumbs up/down → logged as trace annotations → periodic retuning of the routing classifier

---

See [LIMITATIONS.md](LIMITATIONS.md) for honest documentation on the Spotify/LinkedIn cached data approach.

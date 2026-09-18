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
  │  spotify  → Cached Spotify snapshot (monthly)   │
  │  linkedin → Cached LinkedIn snapshot (monthly)  │
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

---

## Measured Results

First committed baseline: 59 golden questions through the shipping graph —
router, handler, production prompt and faithfulness guard — not through a
test harness that calls the tools directly. Every run is committed under
[`eval/results/`](eval/results/) with the git sha and a hash of the indexed
resume, because scores only compare across runs over the same corpus.

| | Baseline |
|---|---|
| Routing accuracy | **91.5%** (54/59) |
| Retrieval recall@k | **83.3%** (24 referenced questions) |
| Retrieval MRR | **0.618** |
| Correct refusals | **100%** (10 questions that should be declined) |
| Answer relevancy | **0.834** ✓ (≥0.75) |
| Faithfulness | 0.660 ✗ (≥0.80) |
| Context recall | 0.674 ✗ (≥0.80) |
| Context precision | 0.636 ✗ (≥0.70) |

Three of four RAGAS metrics are below threshold, and they are published anyway.
That is the starting point the next phases move.

**Faithfulness is not one number.** Per route:

| Route | Faithfulness | |
|---|---|---|
| linkedin | 0.950 | |
| resume | 0.867 | the hybrid retrieval path |
| personal | 0.678 | |
| spotify | 0.417 | context is a numbered list |
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

### Known-broken, on purpose

Measured first so the fix can be reported as a delta rather than asserted:

- **Follow-up questions route at 33%.** Retrieval runs on the bare last
  message, so "and before that?" retrieves on three words.
- **Multi-intent questions are structurally unanswerable** — one label, one
  tool, one branch.
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
| LLM | GPT-4o-mini |
| Agent framework | LangGraph |
| Vector store | Qdrant — embedded, self-hosted or cloud, selected by config |
| Dense retrieval | OpenAI text-embedding-3-small |
| Sparse retrieval | BM25 (rank-bm25) |
| Score fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free) |
| Chunking | Semantic chunker (LangChain Experimental) |
| Observability | In-app latency panel; distributed tracing not currently active (see [Limitations](LIMITATIONS.md)) |
| Session memory | Redis when configured, in-process fallback otherwise |
| Hallucination guard | GPT-4o-mini faithfulness judge (post-generation) |
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
- **Eval gate on PRs:** Fail the PR if RAGAS scores drop below threshold
- **Multi-modal resume parsing:** Index embedded tables and charts from the PDF, not just raw text
- **Feedback loop:** Thumbs up/down → logged as trace annotations → periodic retuning of the routing classifier

---

See [LIMITATIONS.md](LIMITATIONS.md) for honest documentation on the Spotify/LinkedIn cached data approach.

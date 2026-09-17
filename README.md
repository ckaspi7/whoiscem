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

## Retrieval Quality

**No measured numbers yet — deliberately, rather than by omission.**

This section previously carried a metrics table whose every cell read "run eval to
populate." It was removed: the evaluation harness could not complete a run, so there
was nothing behind it.

The harness is the next piece of work. When it lands, this section carries real
figures across the 40-question golden set, every run is committed under
`eval/results/`, and each later change to retrieval is reported as a before/after
against that baseline rather than as an assertion.

```bash
python eval/run_eval.py --output eval/results/$(git rev-parse --short HEAD).json
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

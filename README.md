# HowToCem

A personal AI chatbot built to answer questions about Cem Kaspi — and to demonstrate production-grade AI engineering skills in RAG, agents, observability, and evaluation.

**[Live Demo →](https://howtocem.streamlit.app)** | [Limitations](LIMITATIONS.md)

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

## Retrieval Quality (RAGAS Evaluation)

Evaluation runs against a hand-curated set of 40 questions with ground-truth answers.

| Metric | Baseline (FAISS, fixed chunks) | v2 (Hybrid + Rerank) | Threshold |
|---|---|---|---|
| Faithfulness | — | run eval to populate | ≥ 0.80 |
| Answer Relevancy | — | run eval to populate | ≥ 0.75 |
| Context Recall | — | run eval to populate | ≥ 0.80 |
| Context Precision | — | run eval to populate | ≥ 0.70 |

```bash
# Populate this table:
python eval/run_eval.py --output eval/results/v2_hybrid.json
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini |
| Agent framework | LangGraph |
| Vector store | Qdrant |
| Dense retrieval | OpenAI text-embedding-3-small |
| Sparse retrieval | BM25 (rank-bm25) |
| Score fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free) |
| Chunking | Semantic chunker (LangChain Experimental) |
| Observability | LangSmith tracing + in-app cost/latency panel |
| Session memory | Redis (cross-session rolling summaries) |
| Hallucination guard | GPT-4o-mini faithfulness judge (post-generation) |
| UI | Streamlit |
| Personal DB | SQLite |
| CI | GitHub Actions (lint + unit tests + weekly eval) |

---

## Setup

**Prerequisites:** Docker, Python 3.11+, OpenAI API key.

```bash
git clone https://github.com/ckaspi7/howtocem.git
cd howtocem

# 1. Configure secrets
cp .env.example .env
# → fill in OPENAI_API_KEY at minimum

# 2. Seed personal data (file is git-ignored)
cp data/seed_data.example.json data/seed_data.json
# → fill in your data, then:
python scripts/setup_user_data_db.py

# 3. Start all services (Qdrant + Redis + Streamlit app)
docker-compose up
# App available at http://localhost:8501
```

### Run tests

```bash
# Unit tests — no external services required
pytest tests/unit/ -v

# Integration tests — requires docker-compose up
pytest tests/integration/ -v

# Retrieval quality evaluation
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
- **Feedback loop:** Thumbs up/down → logged to LangSmith → periodic fine-tune of the routing classifier

---

See [LIMITATIONS.md](LIMITATIONS.md) for honest documentation on the Spotify/LinkedIn cached data approach.

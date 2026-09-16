# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start all services (Qdrant + Redis + Streamlit app) — primary dev workflow
docker-compose up

# Run the Streamlit app directly (requires Qdrant and Redis already running)
streamlit run chatbot.py

# Unit tests — no external services required
pytest tests/unit/ -v

# Integration tests — requires docker-compose up
pytest tests/integration/ -v

# Run a single test file
pytest tests/unit/test_fusion.py -v

# Retrieval quality evaluation (requires OPENAI_API_KEY + running services)
python eval/run_eval.py --output eval/results/v2_hybrid.json

# Seed the personal info SQLite database (run once after filling data/seed_data.json)
python scripts/setup_user_data_db.py

# Refresh Spotify cache (requires SPOTIFY_* env vars)
python scripts/refresh_cache.py
```

## Architecture

### Request flow (LangGraph)

Every user message traverses a compiled `StateGraph` defined in `chatbot.py`:

1. **`route_query`** — LLM classifies the query into one of five intents: `resume`, `personal`, `spotify`, `linkedin`, `conversation` (see `router.py`).
2. **Handler node** — one of `handle_resume/personal/spotify/linkedin/conversation` runs the appropriate tool and sets `tool_result` + `context_used` on the state.
3. **`generate_response`** — GPT-4o-mini streams the answer using the tool result as injected context.
4. **`check_faithfulness`** (post-node, in `guardrails/faithfulness_check.py`) — a GPT-4o-mini judge scores the response 1–5 against the retrieved context; scores ≤1 are refused, 2–3 get a disclaimer.
5. **Redis session memory** (`memory/session_memory.py`) — after every turn, the conversation is summarised to 3 sentences and stored in Redis for 30 days, keyed by `session:{uuid}:summary`. Returning visitors get this injected into the system prompt.

### Hybrid RAG pipeline (resume path only)

The `resume` handler calls `tools/resume_tool.py`, which orchestrates three lazy-initialised singletons:

- **`QdrantVectorStore`** (`retrieval/vectorstore.py`) — dense cosine search using `text-embedding-3-small`. Collection `resume_chunks` is auto-built from `data/Cem_Kaspi_Resume.pdf` using a `SemanticChunker` on first run.
- **`BM25Index`** (`retrieval/bm25.py`) — sparse keyword search built from all chunks stored in Qdrant.
- **`CrossEncoderReranker`** (`retrieval/reranker.py`) — `cross-encoder/ms-marco-MiniLM-L-6-v2` (downloads ~90 MB on first run, then cached).

These three feed into `reciprocal_rank_fusion` (`retrieval/fusion.py`) → top-20 from each list → RRF merge → cross-encoder rerank → top-3 chunks returned to the LLM.

### Other tools

- `tools/personal_tool.py` — queries a SQLite database populated from `data/seed_data.json`.
- `tools/spotify_tool.py` / `tools/linkedin_tool.py` — serve from JSON caches in `data/cache/`; these are manually refreshed (see `LIMITATIONS.md`).

### Observability

LangSmith tracing is active when `LANGCHAIN_TRACING_V2=true`. Per-node latencies are tracked in `GraphState.node_latencies` and shown in the Streamlit sidebar.

## Environment variables

Required: `OPENAI_API_KEY`  
Optional but recommended: `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT`  
Services: `REDIS_URL` (defaults to `redis://localhost:6379`), `QDRANT_HOST`/`QDRANT_PORT` (defaults to `localhost:6333`)  
Spotify cache refresh only: `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`

See `.env.example` for the full template.

## Key design constraints

- The `_store`, `_bm25`, and `_reranker` singletons in `resume_tool.py` are module-level globals. They initialise lazily on first query. Qdrant index is rebuilt automatically if the collection doesn't exist.
- Redis failure is non-fatal: `SessionMemory` catches all exceptions and degrades to no-op (app runs without session memory).
- The Streamlit app uses `st.query_params["sid"]` for session identity, making sessions shareable via URL.
- Streamlit Cloud secrets override `.env` values — the loop at the top of `chatbot.py` merges them into `os.environ`.
- The cross-encoder adds ~15 s cold-start latency on a fresh container while the model downloads.

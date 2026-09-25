# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands assume the Python 3.11 venv is active
(`.venv\Scripts\Activate.ps1` in PowerShell).

```bash
# Primary dev workflow — embedded Qdrant + in-process memory, no services needed
streamlit run chatbot.py

# Containers instead (sets QDRANT_MODE=server): Qdrant + Redis + the app
docker-compose up

# Unit tests — no external services required
pytest tests/unit/ -v

# Integration tests — embedded Qdrant by default; QDRANT_MODE=server to use containers
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

`chatbot.py:create_assistant(mode=None, model=None)` compiles one of two
`StateGraph` shapes, selected by `mode` or `AGENT_MODE` (`config.py`), using
the chat model selected by `model` or `CHAT_MODEL`. The two axes are
orthogonal and independently measured. Modes: both are measured against each
other (`eval/results/v8-classifier.json` / `v9-tool-calling.json`) rather than
one replacing the other on faith — see the README's "A real agent, measured
against the switch statement it replaces". `AGENT_MODE=classifier` is the
default: it wins on routing accuracy (98.3% vs 91.5%) and faithfulness (0.701
vs 0.633), which matter more for a narrow-domain factual assistant than
`tool_calling`'s edge on answer relevancy and its structural ability to do
multi-intent and follow-ups without a condensation step. Every user message
still traverses `check_faithfulness` and session memory the same way
regardless of mode (steps 5-6 below). Models: `eval/results/
v10-gpt4o-mini-baseline.json` / `v11-gpt6-luna-comparison.json` — see the
README's gpt-6-luna comparison and `CHAT_MODEL` below; `CHAT_MODEL` stays at
`gpt-4o-mini` by default even though `gpt-6-luna` measures better under
`classifier` mode, because the same comparison shows it broken under
`tool_calling`, and the setting is global across both modes.

**`classifier`** (`_build_classifier_graph`) — the original design:

1. **`condense_query`** (`query_rewrite.py`) — rewrites the latest message into a standalone question using recent history ("and before that?" → "Where did Cem work before TELUS Communications?"), so routing and retrieval never act on a bare pronoun. Only calls the LLM when there is history to resolve against; a first turn is a no-op pass-through. Sets `search_query`.
2. **`route_query`** — LLM classifies `search_query` into one of five intents: `resume`, `personal`, `spotify`, `linkedin`, `conversation` (see `router.py`). Sets `route` (kept apart from `next_step`, which every later node overwrites as control flow).
3. **Handler node** — one of `handle_resume/personal/spotify/linkedin/conversation` calls the route's typed `*_result()` function (see below) and sets `tool_result` + `context_used` from `ToolResult.as_context()` — a failure's error text never reaches either field, only `tool_error`.
4. **`generate_response`** — calls GPT-4o-mini with `.invoke()` (blocking, not `.stream()`) using `tool_result` as injected context, and returns one plain `AIMessage`.

**Self-correction (Phase 3.3), classifier only, between step 4 and the shared tail below:** a graph node also named `check_faithfulness` (confusingly close to, but distinct from, the outer `guardrails.faithfulness_check.check_faithfulness` in the shared tail — this one only scores via `score_faithfulness`, storing `faithfulness_score`; it never applies the disclaimer/refusal banner) decides whether to retry. Only on the `resume` route, with a score below 4 and no retry yet this turn, it loops to `reformulate_and_retry` — broadens the query (`query_rewrite.reformulate_for_retry`), re-runs `search_resume`, regenerates — bounded to one retry by `_MAX_FAITHFULNESS_RETRIES`. Every other route falls straight through: `personal`/`spotify`/`linkedin` are fixed lookups a retry cannot change, and `conversation` has no context to reformulate. Keeping the tiering itself outside the graph (unchanged) is deliberate: it keeps `eval/run_eval.py`'s faithfulness numbers comparable to every result committed before this landed. See the README's self-correction section for the measured before/after and its one honest caveat — the retry fixing nothing when the real problem is `classifier` mode's separate, already-documented inability to call a second tool.

**`tool_calling`** (`_build_tool_calling_graph`) — a real agent: the four
`@tool`-decorated functions are bound directly (`llm.bind_tools(...)`) to a
dedicated, deterministic (`temperature=0`, still `streaming=True`) model — not
the classifier's `temperature=0.7` prose model, which measurably made tool
selection nondeterministic when tried. No `condense_query` node: the agent
sees full history natively and resolves a follow-up while constructing the
tool call itself. Two nodes loop until no more tools are called:

1. **`agent`** — invokes the tool-bound LLM with the full message history under a system prompt plus `_TOOL_CALLING_ADDENDUM` (told the model to always check a tool rather than treat its "key facts" as a ceiling — a first pass without this addendum measured 81.4% routing from premature refusals; with it, 91.5%). Sets `route="conversation"` only if nothing has called a tool yet this turn.
2. **`execute_tools`** — dispatches each `tool_call` by name through `_call_tool` to the matching `*_result()` function (not the `@tool`-wrapped one — that returns a plain string for the LLM's own tool-calling contract). One `AIMessage` can carry more than one `tool_call` — "compare his resume to his LinkedIn" calls `get_resume_info` and `get_linkedin_info` in the same turn, which is Phase 3.2's multi-intent case arriving as a side effect rather than needing separate work. Accumulates into `route`/`context_used`/`context_chunks`/`tool_error` rather than overwriting, so a second round of tool calls in the same turn (genuine multi-hop) does not erase the first. Also appends one `{round, tool, category, args, ok}` entry per call to `trajectory` (stamped with `agent_rounds`, incremented after) — Phase 3.5's raw material, since the accumulated `route` string alone loses round boundaries and arguments. `eval/run_eval.py:score_trajectory` checks the stronger claim `route_correct` cannot: that a multi-intent question's *every* expected category was actually called, not just one of them, and flags any call outside what a question's `acceptable_routes` named.

Both graphs converge back into the shared tail:

5. **`check_faithfulness`** (post-node, in `guardrails/faithfulness_check.py`, called by `chatbot.py:main()` only — not by `eval/run_eval.py`, so RAGAS scores the raw generation, not this banner) — a GPT-4o-mini judge scores the response 1–5 against `context_used`; scores ≤1 are refused, 2–3 get a disclaimer. Context is per-route, not per-answer, so this is only as good as the route it followed — see the faithfulness-by-route breakdown in the README. Distinct from classifier mode's in-graph `check_faithfulness` node above (Phase 3.3): that one only scores, to decide a retry; this one scores and applies the user-facing banner, and `eval/run_eval.py` deliberately never calls it so historical faithfulness numbers stay comparable.
6. **Session memory** (`memory/session_memory.py`) — after every turn, the conversation is summarised to 3 sentences and stored under `session:{uuid}:summary`, TTL 30 days. Redis when `REDIS_URL` is set and reachable, an in-process fallback otherwise. Returning visitors get this injected into the system prompt.

The caller (`chatbot.py:main()`, `eval/run_eval.py`) gets live token-by-token
output via `graph.stream(state, stream_mode=["messages", "values"])` rather
than a node returning a raw generator — the same call's streaming callbacks
surface through LangGraph regardless of which method the node used
(`.invoke()` still streams token deltas out), and the same pass yields the
full final state, so nothing needs a second call or a checkpointer just to
read the result back. `main()`'s node-name filter accepts both
`generate_response` and `agent`, since the two modes name their final-answer
node differently.

### Typed tool results

Every tool (`tools/*.py`) has a `*_result()` function returning `ToolResult` (`tools/result.py`) — `ok=True` with `content`, or `ok=False` with `error`. `ToolResult.as_context()` returns `content` on success and `""` on failure: an error can never masquerade as retrieved evidence to the model or the faithfulness judge. The `@tool`-decorated functions (`get_resume_info`, `get_personal_info`, `get_music_taste`, `get_linkedin_info`) are thin string-interface wrappers over these, kept for the LangChain tool-calling surface; `chatbot.py`'s handlers call the typed functions directly, not `.invoke()`.

### Hybrid RAG pipeline (resume path only)

The `resume` handler calls `tools/resume_tool.py:search_resume()`, which orchestrates three lazy-initialised singletons via `_init_retrieval()`:

- **`QdrantVectorStore`** (`retrieval/vectorstore.py`) — dense cosine search using `text-embedding-3-small`. Collection `resume_chunks` is built from `RESUME_PATH` (default `data/resume.md`) on first query, or whenever `needs_rebuild()` detects the source bytes or `CHUNKER_VERSION` changed. Markdown is chunked structurally on headings (`retrieval/chunking.py`); a PDF falls back to `SemanticChunker`, since extraction strips its headings.
- **`BM25Index`** (`retrieval/bm25.py`) — sparse keyword search, built from the vector store's own chunks so its returned ids are the store's ids (fusion joins on real identity, not list position).
- **`CrossEncoderReranker`** (`retrieval/reranker.py`) — `cross-encoder/ms-marco-MiniLM-L-6-v2` (downloads ~90 MB on first run, then cached).

`RETRIEVAL_STRATEGY` selects how these combine: `auto` (default — the whole corpus, reranked, while it fits `CORPUS_FITS_CONTEXT_CHARS`; plain dense search once it doesn't), `dense`, `sparse`, `rrf`, or `rrf_rerank` (RRF fusion of top-`CANDIDATE_POOL` from each, then cross-encoder rerank). See `eval/results/ablation-retrieval.json` and the README's ablation table before assuming hybrid beats dense on a given corpus size — on the current 7 KB resume it does not.

`_init_retrieval()` commits `_store`/`_bm25`/`_reranker` to module globals only after each stage fully succeeds, so a transient failure (a Qdrant container not yet ready) is retried on the next call instead of poisoning the process. `retrieval/cache.py` (`RetrievalCache`) caches query embeddings and full retrieval results, same Redis-or-in-process pattern as session memory (shared connection logic in `redis_backend.py`), scoped by the corpus fingerprint so a rebuilt index can't serve a stale cached answer.

### Other tools

- `tools/personal_tool.py` — queries a SQLite database populated from `data/seed_data.json`, projecting an explicit column allowlist (never `SELECT *`).
- `tools/spotify_tool.py` / `tools/linkedin_tool.py` — serve from JSON caches in `data/cache/`; manually refreshed (see `LIMITATIONS.md`), and the answer states the cache's real age (`tools/freshness.py`) rather than asserting a refresh cadence.

### Observability

Per-node latencies are tracked in `GraphState.node_latencies` and shown in the Streamlit sidebar. Distributed tracing is via `observability.py` (OpenTelemetry through OpenInference, exported to Arize Phoenix) — a local collector needs no account (`pip install arize-phoenix && phoenix serve`), or set `PHOENIX_COLLECTOR_ENDPOINT`/`PHOENIX_API_KEY` for Phoenix Cloud. `setup_tracing()` must run before `create_assistant()`: the instrumentor patches LangChain's callback manager, so anything constructed earlier is never traced.

Logging (Phase 4.4): `observability.setup_logging()` attaches a structured JSON handler to the root logger — every module's existing `logger = logging.getLogger(__name__)` calls had nowhere useful to go before this, since nothing ever called `logging.basicConfig`. `observability.set_session_id(session_id)` (called once in `chatbot.py:main()` right after the session id resolves) attaches that id to every subsequent log record in the current context as a correlation field, without threading it through every call site. `chatbot.py` logs one structured `"turn completed"` line per turn with `node_latencies`, `route`, and the faithfulness score — the same per-stage timing the sidebar shows interactively, now also in a deployed instance's actual logs.

### Evaluation

`eval/run_eval.py` runs the golden set (`eval/golden_set.json`, generated by `eval/build_golden_set.py` — never hand-edit the JSON) through this real graph and scores routing/tool-selection accuracy, retrieval recall@k/MRR against verbatim golden snippets, refusal rate on unanswerable/adversarial cases, a dedicated tool-error rate, self-correction retries (Phase 3.3, `score_self_correction` — classifier only), trajectory correctness (Phase 3.5, `score_trajectory` — tool_calling only, needs no RAGAS judge calls), and RAGAS (faithfulness/answer_relevancy/context_recall/context_precision, over answerable questions only). `--agent-mode classifier|tool_calling` overrides `AGENT_MODE` for one run, which is how `v8-classifier.json`/`v9-tool-calling.json` were produced side by side, and `--model` does the same for `CHAT_MODEL` (`v10-gpt4o-mini-baseline.json`/`v11-gpt6-luna-comparison.json`). `eval/compare.py` gates a run against the most recent committed result in `eval/results/`; any nonzero tool-error count fails the gate outright, checked directly rather than as a tolerance band, and MRR/context precision are reported but never gated across a chunker or agent_mode change, in either direction — both change what a "chunk" is, not just how good retrieval is (a `chat_model` change is reported the same way but does *not* get this exemption — it doesn't touch chunk granularity, so rank metrics stay gated normally across it). `eval/ablate_retrieval.py` measures the retrieval strategies against each other, routing excluded.

## Environment variables

All of these are resolved in one place, `config.py` (`load_settings()`), which also
loads `.env`. Real environment variables always win over `.env`.

Required: `OPENAI_API_KEY`  
Observability: `PHOENIX_COLLECTOR_ENDPOINT` (unset = local collector on :6006), `PHOENIX_API_KEY`, `PHOENIX_PROJECT_NAME`  
Vector store: `QDRANT_MODE` = `embedded` (default; on-disk at `QDRANT_PATH`, no server) | `server` (`QDRANT_HOST`/`QDRANT_PORT`) | `cloud` (`QDRANT_URL`/`QDRANT_API_KEY`)  
Session memory: `REDIS_URL` — unset means an in-process fallback, not a disabled feature  
Session link signing: `SESSION_SECRET` — unset means a per-process-start secret, so signed links stop verifying after a restart (see LIMITATIONS.md)  
Resume source: `RESUME_PATH` (defaults to `data/resume.md`)  
Agent architecture: `AGENT_MODE` = `classifier` (default, measured best) | `tool_calling` (real agent; see README)  
Chat model: `CHAT_MODEL` = `gpt-4o-mini` (default) | `gpt-6-luna` (measured better under `classifier` mode only, broken under `tool_calling` — see README/LIMITATIONS)  
Spotify cache refresh only: `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`

See `.env.example` for the full template.

## Key design constraints

- The `_store`, `_bm25`, `_reranker`, and `_cache` singletons in `resume_tool.py` are module-level globals, initialised lazily on first query. Each is committed to its global only after that stage's setup fully succeeds — a transient failure (a Qdrant container not yet ready) must be retried on the next call, not permanently cached as broken. `tests/unit/test_resume_tool.py` guards this against regressing; it is the fix for a real incident where a CI service-container race silently broke retrieval for an entire scheduled run.
- The Qdrant index rebuilds when `needs_rebuild()` finds the source bytes or `retrieval/vectorstore.py:CHUNKER_VERSION` changed, not merely when the collection is absent. A stale index answers happily from the previous corpus.
- The Qdrant client is built by `retrieval/backends.py:create_qdrant_client()` from `Settings`, never from hardcoded coordinates. `QdrantVectorStore` takes an injected client and an optional `RetrievalCache`, which is how the integration tests run without a server or a cache.
- Embedded Qdrant takes an exclusive lock on `QDRANT_PATH`: the app and the test suite cannot share one path at the same time.
- `chatbot.py` targets Python 3.11 — no f-string expression may contain a backslash (legal only from 3.12). `tests/unit/test_app_boot.py` guards this.
- A tool's failure must never reach `tool_result`/`context_used` as text (see Typed tool results, above) — `ToolResult.as_context()` is the enforcement point. A handler that bypasses it and interpolates an exception into either field reintroduces the bug the faithfulness judge used to silently score against a stack trace.
- Redis failure (session memory or the retrieval cache) is non-fatal: both fall back in-process via `redis_backend.connect()`, never raising just because a cache is unavailable.
- The Streamlit app uses `st.query_params["sid"]` for session identity, making sessions shareable via URL. Phase 4.5: the value is HMAC-signed (`chatbot._sign_session_id`/`_verify_session_id`) — a `sid` that fails verification (fabricated, edited, or signed under a different `SESSION_SECRET`) is never trusted as a previously-issued session; a fresh one is minted instead. This closes fabrication, not leakage of a real signed link — see LIMITATIONS.md for that distinction.
- Streamlit Cloud secrets override `.env` values — `_apply_streamlit_secrets()` in `chatbot.py` merges them into `os.environ`, called after `set_page_config()` (reading `st.secrets` is itself a Streamlit command and must not be first) and before `setup_tracing()`/`create_assistant()`.
- The golden set (`eval/golden_set.json`) is generated, not hand-authored — edit `eval/build_golden_set.py` and rerun it. `tests/unit/test_golden_set.py` fails the build if the committed JSON drifts from what the generator produces, or if a `reference_snippet`/`reference_section` no longer appears in the indexed resume.
- `GraphState.messages` is `Annotated[list[BaseMessage], add_messages]` — real LangChain messages, not dicts. A node that wants to add a message returns `{"messages": [new_message]}` (a single-element list); the reducer appends it. Returning the full accumulated list back (e.g. via `{**state, ...}` without overriding `"messages"`) is harmless — `add_messages` matches by id and replaces in place rather than duplicating — but only a single new message is ever actually being added anywhere in this graph today.
- OpenAI's Tier 1 caps `gpt-4o-mini` at 10,000 requests/day (a rolling 24-hour window, confirmed from the `x-ratelimit-reset-requests` header — not a fixed midnight reset), shared across local dev, CI, and the deployed app. A full RAGAS eval run costs roughly 2,700 requests, measured, not the ~500-1,000 first estimated — `faithfulness` decomposes each answer into claims and verifies each separately. Use `eval/run_eval.py --no-ragas` (~120 requests) for iteration; save full RAGAS runs for a baseline right before a commit; check `x-ratelimit-remaining-requests` with one minimal request before a run that matters rather than assume the window has reset. See LIMITATIONS.md.
- `gpt-6-luna` (a `CHAT_MODEL` option) rejects any non-default `temperature` outright (only `1`, the default, is accepted) and only supports tool/function calling at `reasoning_effort="none"`. `create_assistant` handles both — omitting the `temperature` override and setting `reasoning_effort` only for the tool-bound agent — rather than let either surface as a live 400. Neither constraint applies to `gpt-4o-mini`.
- The cross-encoder adds ~15 s cold-start latency on a fresh container while the model downloads.
- Real cost accounting (Phase 4.2) reads each node's own response usage directly (`cost.usage_from_response`, merged via `chatbot._add_usage`) rather than `langchain_community.get_openai_callback()` — confirmed directly that the callback does not propagate through LangGraph's node execution (a real graph turn showed 0 tokens captured that way despite a real model call happening). `create_assistant`'s `_llm` helper also sets `stream_usage=True` unconditionally: `ChatOpenAI(streaming=True)` returns no usage data on `.invoke()` without it (defaults to `False`), which silently zeroed `generate_response`'s own usage until found by actually running it. See LIMITATIONS.md's Cost Accounting section.

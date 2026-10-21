# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands assume the Python 3.11 venv is active
(`.venv\Scripts\Activate.ps1` in PowerShell).

```bash
# Primary dev workflow — embedded Qdrant + in-process memory, no services needed
streamlit run chatbot.py

# Headless API (Phase 4.1) — separate from the Streamlit app above, not yet
# what it calls internally. See LIMITATIONS.md for that scope boundary.
uvicorn api:app --reload

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

The frontend in `web/` has its own Node commands; see `web/CLAUDE.md`.

One-time setup after cloning: `git config core.hooksPath .githooks` — enables the
branch-safety hook below. `pre-commit install` separately enables the existing
lint/format/PII hooks (`.pre-commit-config.yaml`).

## Git workflow

Branches: `feat/`, `fix/`, `chore/`, `docs/` prefixes, one unit of work per branch,
merged via PR rather than committed straight to `main`.

**Always create a branch with an explicit base, never bare `git checkout -b <name>`:**
```bash
git fetch origin
git checkout -b feat/whatever origin/main
git log origin/main..HEAD --oneline   # must print nothing before any work starts
```
This is a real, incident-driven rule, not caution for its own sake: a branch was
once created with a bare `git checkout -b`, silently inheriting a *different*
feature's commit as its base because another branch happened to be checked out
in this same working directory at the time (from earlier testing). That commit
rode along, unreviewed, into the PR that merged it. `git log origin/main..HEAD`
on a freshly created branch must always be empty — if it isn't, stop and
recreate the branch from `origin/main` explicitly before doing anything else.

**`.githooks/pre-push`** is the backstop for the same failure mode: for any new
branch, it checks every commit the push would introduce against every *other*
remote branch, and blocks the push if one already belongs to a different
branch — the exact signature of the incident above, not a false positive on a
branch that's merely behind `main`. Deliberate stacking of one branch on
another is the one legitimate case this would block; use `git push --no-verify`
for that, consciously, not as a routine workaround.

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
regardless of mode (steps 5-6 below). Models (ADR-0011, `models.py`):
`CHAT_MODEL` defaults to `claude-haiku-5-5`, with `gpt-6-luna` as the automatic
`FALLBACK_CHAT_MODEL` on the other provider; `gpt-4o-mini` stays selectable until
OpenAI retires it on 2027-01-14. The three were measured side by side
(`eval/results/v16-*` default mode, `v17-*` tool-calling, `v18-*` after the scope
instruction; `v19-*` is the committed baseline) — see ADR-0011 for the numbers and caveats. Both axes are global:
`CHAT_MODEL` applies in both modes.

**`classifier`** (`_build_classifier_graph`) — the original design:

1. **`condense_query`** (`query_rewrite.py`) — rewrites the latest message into a standalone question using recent history ("and before that?" → "Where did Cem work before TELUS Communications?"), so routing and retrieval never act on a bare pronoun. Only calls the LLM when there is history to resolve against; a first turn is a no-op pass-through. Sets `search_query`.
2. **`route_query`** — LLM classifies `search_query` into one of five intents: `resume`, `personal`, `spotify`, `linkedin`, `conversation` (see `router.py`). Sets `route` (kept apart from `next_step`, which every later node overwrites as control flow).
3. **Handler node** — one of `handle_resume/personal/spotify/linkedin/conversation` calls the route's typed `*_result()` function (see below) and sets `tool_result` + `context_used` from `ToolResult.as_context()` — a failure's error text never reaches either field, only `tool_error`.
4. **`generate_response`** — calls the chat model with `.invoke()` (blocking, not `.stream()`) using `tool_result` as injected context, and returns one plain `AIMessage`.

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

5. **`check_faithfulness`** (post-node, in `guardrails/faithfulness_check.py`, called by `chatbot.py:main()` only — not by `eval/run_eval.py`, so RAGAS scores the raw generation, not this banner) — scores the response 1–5 against `context_used`; scores ≤1 are refused, 2–3 get a disclaimer. As of ADR-0008 the scorer is a local classifier (HHEM-2.1-Open), not an LLM judge — see `_hhem_to_five_scale` for how its continuous 0–1 output maps onto this 1–5 scale. Context is per-route, not per-answer, so this is only as good as the route it followed — see the faithfulness-by-route breakdown in the README. Distinct from classifier mode's in-graph `check_faithfulness` node above (Phase 3.3): that one only scores, to decide a retry; this one scores and applies the user-facing banner, and `eval/run_eval.py` deliberately never calls it so historical faithfulness numbers stay comparable.
6. **Session memory** (`memory/session_memory.py`) — after every turn, the conversation is summarised to 3 sentences and stored under `session:{uuid}:summary`, TTL 30 days. Redis when `REDIS_URL` is set and reachable, an in-process fallback otherwise. Returning visitors get this injected into the system prompt.

Both callers (`chatbot.py:main()`, `eval/run_eval.py`) call `graph.invoke()`,
not `graph.stream()` — nothing is ever displayed to a visitor until the whole
turn, faithfulness check included, has finished. This was a deliberate change
(previously `main()` streamed live tokens as `generate_response`/`agent`
produced them): a warned or refused answer would visibly replace text the
visitor had already read mid-stream, which looked like the app was broken
even though the guardrail was working exactly as designed.

### Typed tool results

Every tool (`tools/*.py`) has a `*_result()` function returning `ToolResult` (`tools/result.py`) — `ok=True` with `content`, or `ok=False` with `error`. `ToolResult.as_context()` returns `content` on success and `""` on failure: an error can never masquerade as retrieved evidence to the model or the faithfulness judge. The `@tool`-decorated functions (`get_resume_info`, `get_personal_info`, `get_music_taste`, `get_linkedin_info`) are thin string-interface wrappers over these, kept for the LangChain tool-calling surface; `chatbot.py`'s handlers call the typed functions directly, not `.invoke()`.

### Other tools

- `tools/personal_tool.py` — queries a SQLite database populated from `data/seed_data.json`, projecting an explicit column allowlist (never `SELECT *`).
- `tools/spotify_tool.py` / `tools/linkedin_tool.py` — serve from JSON caches in `data/cache/`; manually refreshed (see `LIMITATIONS.md`), and the answer states the cache's real age (`tools/freshness.py`) rather than asserting a refresh cadence.

### Observability

Per-node latencies are tracked in `GraphState.node_latencies` and shown in the Streamlit sidebar. Distributed tracing is via `observability.py` (OpenTelemetry through OpenInference, exported to Arize Phoenix) — a local collector needs no account (`pip install arize-phoenix && phoenix serve`), or set `PHOENIX_COLLECTOR_ENDPOINT`/`PHOENIX_API_KEY` for Phoenix Cloud. `setup_tracing()` must run before `create_assistant()`: the instrumentor patches LangChain's callback manager, so anything constructed earlier is never traced.

Logging (Phase 4.4): `observability.setup_logging()` attaches a structured JSON handler to the root logger — every module's existing `logger = logging.getLogger(__name__)` calls had nowhere useful to go before this, since nothing ever called `logging.basicConfig`. `observability.set_session_id(session_id)` (called once in `chatbot.py:main()` right after the session id resolves) attaches that id to every subsequent log record in the current context as a correlation field, without threading it through every call site. `chatbot.py` logs one structured `"turn completed"` line per turn with `node_latencies`, `route`, and the faithfulness score — the same per-stage timing the sidebar shows interactively, now also in a deployed instance's actual logs.

## Environment variables

All of these are resolved in one place, `config.py` (`load_settings()`), which also
loads `.env`. Real environment variables always win over `.env`.

See `.env.example` for the full template and per-variable documentation. On a host with no `.env` file
(e.g. Streamlit Community Cloud), `_apply_streamlit_secrets()` in `chatbot.py`
is the only bridge from `st.secrets` into these — `.streamlit/secrets.toml.example`
lists exactly the subset it copies over (`_SECRET_KEYS`); anything else pasted
into that UI is silently inert.

## Key design constraints

- `chatbot.py` targets Python 3.11 — no f-string expression may contain a backslash (legal only from 3.12). `tests/unit/test_app_boot.py` guards this.
- A tool's failure must never reach `tool_result`/`context_used` as text (see Typed tool results, above) — `ToolResult.as_context()` is the enforcement point. A handler that bypasses it and interpolates an exception into either field reintroduces the bug the faithfulness judge used to silently score against a stack trace.
- Redis failure (session memory or the retrieval cache) is non-fatal: both fall back in-process via `redis_backend.connect()`, never raising just because a cache is unavailable.
- `GraphState.messages` is `Annotated[list[BaseMessage], add_messages]` — real LangChain messages, not dicts. A node that wants to add a message returns `{"messages": [new_message]}` (a single-element list); the reducer appends it. Returning the full accumulated list back (e.g. via `{**state, ...}` without overriding `"messages"`) is harmless — `add_messages` matches by id and replaces in place rather than duplicating — but only a single new message is ever actually being added anywhere in this graph today.
- Model quirks live in `models.py`, one factory for all three `CHAT_MODEL` options, never in call sites. `gpt-6-luna` rejects any non-default `temperature` and only supports tool calling at `reasoning_effort="none"`, so every Luna construction sets that. `claude-haiku-5-5` also rejects a non-default `temperature`, runs with thinking disabled (equal footing, and `max_tokens=2048` because thinking would count toward it), and rejects a conversation that ends on an assistant message ("prefill"): the self-correction retry therefore leaves the first attempt out of the regeneration prompt. None of this applies to `gpt-4o-mini`. Tools are bound to each model *before* the fallback wraps them (`RunnableWithFallbacks` has no `bind_tools`).
- The fallback (`FALLBACK_CHAT_MODEL`, default `gpt-6-luna` only while `CHAT_MODEL` is the default) logs a WARNING on every switch, and each node's usage records the model that actually answered so `cost.graph_cost` prices a fallback answer at the fallback's rate. `eval/run_eval.py` always passes `use_fallback=False`: a run measures exactly one model, and a silent switch must not contaminate a comparison.
- The shell may carry `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` (a work gateway); the Anthropic SDK honours them and would send this project's key there. `models.py` logs a warning when they are set; unset both for any run of this project.
- `chatbot.MAX_INPUT_CHARS` (1000, Phase 4.3) rejects an oversized chat input before it reaches `st.session_state.messages` or the graph — no API call, and the text never enters conversation history (it would otherwise cost tokens as context on every later turn even though the rejected turn itself made no call). See LIMITATIONS.md for why a heuristic prompt-injection filter was deliberately not added alongside it.
- Real cost accounting (Phase 4.2) reads each node's own response usage directly (`cost.usage_from_response`, merged via `chatbot._add_usage`) rather than `langchain_community.get_openai_callback()` — confirmed directly that the callback does not propagate through LangGraph's node execution (a real graph turn showed 0 tokens captured that way despite a real model call happening). `create_assistant`'s `_llm` helper also sets `stream_usage=True` unconditionally: `ChatOpenAI(streaming=True)` returns no usage data on `.invoke()` without it (defaults to `False`), which silently zeroed `generate_response`'s own usage until found by actually running it. See LIMITATIONS.md's Cost Accounting section.
- OpenAI's Tier 1 caps `gpt-4o-mini` at 10,000 requests/day (a rolling 24-hour window, confirmed from the `x-ratelimit-reset-requests` header — not a fixed midnight reset), shared across local dev, CI, and the deployed app. A full RAGAS eval run costs roughly 2,700 requests, measured, not the ~500-1,000 first estimated — `faithfulness` decomposes each answer into claims and verifies each separately. Use `eval/run_eval.py --no-ragas` (~120 requests) for iteration; save full RAGAS runs for a baseline right before a commit; check `x-ratelimit-remaining-requests` with one minimal request before a run that matters rather than assume the window has reset. See LIMITATIONS.md.
- Embedded Qdrant takes an exclusive lock on `QDRANT_PATH`: the app and the test suite cannot share one path at the same time.
- The cross-encoder adds ~15 s cold-start latency on a fresh container while the model downloads. The faithfulness guardrail's HHEM classifier (ADR-0008) is a second, separate local model with its own cold start — ~10-15 s once its weights are cached, a one-time ~77 s the very first time anything on that container downloads them.

## More detail, loaded on demand

These files hold detail for one area. Claude Code loads each only once it reads or edits a file that matches.
Before you change code in one of these areas, read the file named here if it has not loaded yet.

- `web/CLAUDE.md`: the Next.js frontend (proxy, themes, contrast gate, design-review).
- `.claude/rules/api.md`: the headless API (`api.py`, auth, rate limit, spend cap).
- `.claude/rules/retrieval.md`: the hybrid RAG pipeline and the vector-store constraints.
- `.claude/rules/eval.md`: the evaluation harness and the golden set.
- `.claude/rules/streamlit-ui.md`: the Streamlit app's rerun and reveal behaviour, session ids and secrets.

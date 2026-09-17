# Limitations & Design Decisions

## Spotify Data

**Status:** Cached snapshot, refreshed manually (target: monthly).

**Why not live?** Spotify's OAuth flow requires a redirect URI that resolves to a real server.
Stateless Streamlit Cloud deployments cannot receive OAuth callbacks, so the live token flow
cannot complete in the deployed environment.

**What's actually there:** `data/cache/spotify_cache.json` contains a real export from
Cem's Spotify account, generated locally using `scripts/refresh_cache.py`, which implements
the full OAuth flow (see that file for details). The cache includes a `cached_at` timestamp
that is displayed in the UI so users always know how fresh the data is.

**To update:** Run `python scripts/refresh_cache.py` locally, then commit the updated
`data/cache/spotify_cache.json`. The deployed app picks it up on next restart.

---

## LinkedIn Data

**Status:** Manually curated snapshot, refreshed when career information changes.

**Why not live?** LinkedIn's official API removed personal profile access for most developers
in 2023. Unofficial scraping libraries violate LinkedIn's ToS and are fragile.

**What's actually there:** `data/cache/linkedin_cache.json` is manually maintained to reflect
Cem's current career history, with a `cached_at` timestamp displayed in the UI.

**To update:** Edit `data/cache/linkedin_cache.json` directly and run
`python scripts/refresh_cache.py` to bump the timestamp.

---

## Vector Store Persistence

The Qdrant vector store is populated from the resume PDF (`RESUME_PATH`) the first
time a query needs it, and rebuilt automatically whenever the collection is missing.

Which Qdrant it talks to is configuration, not code: `QDRANT_MODE` selects an
embedded on-disk store (the default — no server), a Qdrant reached over HTTP, or
Qdrant Cloud. Embedded mode takes an exclusive lock on its directory, so exactly
one process may use a given `QDRANT_PATH` at a time.

**Not yet done:** the deployed Streamlit Cloud app has no persistent volume, so
embedded storage there is rebuilt on every cold start. Pointing it at Qdrant Cloud
(`QDRANT_MODE=cloud`) is the fix and is not yet in place.

---

## Session Memory Persistence

With `REDIS_URL` set, rolling summaries live in Redis with a 30-day TTL. Without
it, they fall back to an in-process store: the feature still works, but summaries
are lost on restart and are not shared between processes. The sidebar shows which
backend is live so the deployed behaviour is never a guess.

---

## Tracing

**Status: not active.** `LANGCHAIN_TRACING_V2` is honoured, but no working
LangSmith key is configured, so nothing is currently being traced. Rather than
leave a claim standing on an unset variable, the README no longer lists tracing
as a live capability. Instrumentation is being moved to OpenTelemetry via
OpenInference so the backend is a configuration choice rather than a hard
dependency, matching how the vector store and session memory already work.

---

## Cost Accounting

**Status: the number in the sidebar is wrong, and knowingly so.**

`_accumulate_cost` estimates tokens as `len(text) // 4` and applies the *output*
price to all of them. Four of the five model calls per turn — the router, the
faithfulness judge, the summariser and the embeddings — are not counted at all.
`tiktoken` is a dependency and unused; the API already returns exact usage.

It is displayed to five decimal places, which implies a precision it does not
have. Replacing the estimate with reported usage is scheduled; until then, read
the figure as a lower bound of the wrong quantity.

---

## Personal Information

The database behind the `personal` route stores no contact details, no date of
birth and no family information. The tool reads an explicit column allowlist
rather than `SELECT *`, so a database that still carries older columns cannot
leak them into a prompt, a log or a trace. `tests/unit/test_no_pii_committed.py`
fails the build if personal data becomes tracked by git, including inside a PDF.

---

## Resume Source

`data/resume.md` is a redacted copy: the phone number and email address are
removed. `RESUME_PATH` overrides it, so a local unredacted file can be indexed
without touching the repository.

---

## Chunking — Currently Degenerate

**Status: measured, unfixed, and the reason the evaluation work comes first.**

`SemanticChunker` with percentile thresholding splits the resume into **three
chunks** of roughly 2,200 / 1,000 / 4,100 characters. The pipeline then takes
the top 20 dense results, the top 20 BM25 results, fuses them, and reranks to
the top 3 — of a corpus of 3. Every query returns the entire document.

So the hybrid retrieval this project is built around is, on this corpus, not
retrieving anything: RRF has nothing to fuse and the cross-encoder has nothing
to discriminate between. BM25 is worse than inert — a term appearing in all
three chunks scores zero IDF, so sparse search regularly returns nothing at all.

Switching the source from PDF to Markdown was expected to help and did not: it
moved the count from two chunks to three. The cause is the chunker's
configuration on a short document, not the extraction format.

This is left in place on purpose. It is the baseline the evaluation harness will
measure, so that section-aware chunking can be reported as a before-and-after
rather than asserted.

---

## Cross-Encoder Reranker

The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) downloads ~90 MB on first run
and runs fully locally with no API cost. Subsequent runs use the cached model.
Cold-start on a fresh container adds ~15 seconds to first query latency.

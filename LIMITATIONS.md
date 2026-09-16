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

The Qdrant vector store is populated from `data/Cem_Kaspi_Resume.pdf` on first startup.
If Qdrant is restarted with a fresh volume, the index is rebuilt automatically.
On Streamlit Cloud (no Docker), a persistent Qdrant Cloud instance or a managed vector
DB would be required for production deployment.

---

## Cross-Encoder Reranker

The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) downloads ~90 MB on first run
and runs fully locally with no API cost. Subsequent runs use the cached model.
Cold-start on a fresh container adds ~15 seconds to first query latency.

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from langchain.tools import tool

_CACHE_PATH = os.path.join("data", "cache", "spotify_cache.json")


def _load_cache() -> dict[str, Any]:
    with open(_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


@tool
def get_music_taste() -> str:
    """Get Cem's music taste from a cached Spotify snapshot (refreshed monthly)."""
    try:
        data = _load_cache()
        cached_at = data.get("cached_at", "unknown")
        try:
            dt = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
            date_str = dt.strftime("%B %Y")
        except ValueError:
            date_str = cached_at

        artists = data.get("top_artists", [])
        tracks = data.get("top_tracks", [])

        artists_str = "\n".join(
            f"{a['rank']}. {a['name']} — "
            f"{', '.join(a['genres']) if isinstance(a['genres'], list) else a['genres']}"
            for a in artists
        )
        tracks_str = "\n".join(
            f"{t['rank']}. {t['title']} — {t['artists']}" for t in tracks
        )

        return (
            f"Music taste as of {date_str} (refreshed monthly from Spotify):\n\n"
            f"Top Artists:\n{artists_str}\n\n"
            f"Top Tracks:\n{tracks_str}"
        )
    except FileNotFoundError:
        return "Spotify cache not found. Run scripts/refresh_cache.py to generate it."
    except Exception as e:
        return f"Error reading Spotify data: {e}"

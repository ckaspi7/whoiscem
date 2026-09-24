from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain.tools import tool

from tools.freshness import describe_age
from tools.result import ToolResult

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join("data", "cache", "spotify_cache.json")


def _load_cache() -> dict[str, Any]:
    with open(_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_music_taste_result() -> ToolResult:
    """Typed lookup: content on success, an error that never becomes context.

    A missing cache is success, not failure — the lookup worked and the honest
    answer is "not available", which the model should relay rather than treat
    as a broken tool.
    """
    try:
        data = _load_cache()
    except FileNotFoundError:
        return ToolResult.success("Spotify cache not found. Run scripts/refresh_cache.py to generate it.")
    except Exception as e:
        logger.warning("Spotify cache read failed: %s", e)
        return ToolResult.failure(str(e))

    try:
        date_str = describe_age(data.get("cached_at", "unknown"))

        artists = data.get("top_artists", [])
        tracks = data.get("top_tracks", [])

        artists_str = "\n".join(
            f"{a['rank']}. {a['name']} — "
            f"{', '.join(a['genres']) if isinstance(a['genres'], list) else a['genres']}"
            for a in artists
        )
        tracks_str = "\n".join(f"{t['rank']}. {t['title']} — {t['artists']}" for t in tracks)

        return ToolResult.success(
            f"Spotify snapshot exported {date_str}. Manual export, not live data, "
            f"so it reflects listening at that time:\n\n"
            f"Top Artists:\n{artists_str}\n\n"
            f"Top Tracks:\n{tracks_str}"
        )
    except Exception as e:
        logger.warning("Spotify data formatting failed: %s", e)
        return ToolResult.failure(str(e))


@tool
def get_music_taste() -> str:
    """Get Cem's music taste from a cached Spotify snapshot.

    The snapshot is refreshed manually, and the answer states how old it is.
    """
    return get_music_taste_result().as_tool_string("Error reading Spotify data")

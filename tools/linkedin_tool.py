from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from langchain.tools import tool

_CACHE_PATH = os.path.join("data", "cache", "linkedin_cache.json")


def _load_cache() -> dict[str, Any]:
    with open(_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


@tool
def get_linkedin_info() -> str:
    """Get Cem's career information from a cached LinkedIn snapshot (refreshed monthly)."""
    try:
        data = _load_cache()
        cached_at = data.get("cached_at", "unknown")
        try:
            dt = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
            date_str = dt.strftime("%B %Y")
        except ValueError:
            date_str = cached_at

        lines = [
            f"LinkedIn profile as of {date_str} (refreshed monthly):\n",
            f"Headline: {data['headline']}",
            f"About: {data['about']}\n",
            "Career History:",
        ]

        for pos in data.get("positions", []):
            lines.append(f"\n{pos['title']} at {pos['company']} ({pos['startDate']} – {pos['endDate']})")
            lines.append(f"  {pos['description']}")
            for promo in pos.get("promotions", []):
                lines.append(f"  └ {promo['title']} ({promo['startDate']} – {promo['endDate']})")

        return "\n".join(lines)
    except FileNotFoundError:
        return "LinkedIn cache not found. Run scripts/refresh_cache.py to generate it."
    except Exception as e:
        return f"Error reading LinkedIn data: {e}"

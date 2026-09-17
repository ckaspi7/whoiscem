from __future__ import annotations

import json
import os
from typing import Any

from langchain.tools import tool

from tools.freshness import describe_age

_CACHE_PATH = os.path.join("data", "cache", "linkedin_cache.json")


def _load_cache() -> dict[str, Any]:
    with open(_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


@tool
def get_linkedin_info() -> str:
    """Get Cem's career information from a curated LinkedIn snapshot.

    Maintained by hand, and the answer states how old it is.
    """
    try:
        data = _load_cache()
        date_str = describe_age(data.get("cached_at", "unknown"))

        lines = [
            f"LinkedIn profile, last updated {date_str}:\n",
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

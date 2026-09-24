from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain.tools import tool

from tools.freshness import describe_age
from tools.result import ToolResult

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join("data", "cache", "linkedin_cache.json")


def _load_cache() -> dict[str, Any]:
    with open(_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_linkedin_info_result() -> ToolResult:
    """Typed lookup: content on success, an error that never becomes context.

    A missing cache is success, not failure — the lookup worked and the honest
    answer is "not available", which the model should relay rather than treat
    as a broken tool.
    """
    try:
        data = _load_cache()
    except FileNotFoundError:
        return ToolResult.success("LinkedIn cache not found. Run scripts/refresh_cache.py to generate it.")
    except Exception as e:
        logger.warning("LinkedIn cache read failed: %s", e)
        return ToolResult.failure(str(e))

    try:
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

        return ToolResult.success("\n".join(lines))
    except Exception as e:
        logger.warning("LinkedIn data formatting failed: %s", e)
        return ToolResult.failure(str(e))


@tool
def get_linkedin_info() -> str:
    """Get Cem's career information from a curated LinkedIn snapshot.

    Maintained by hand, and the answer states how old it is.
    """
    return get_linkedin_info_result().as_tool_string("Error reading LinkedIn data")

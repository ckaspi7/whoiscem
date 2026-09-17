"""How old a cached snapshot is, stated plainly.

Both cache-backed tools used to tell the model their data was "refreshed
monthly". It was eighteen months old. Whatever goes into the prompt about
freshness has to be derived from the timestamp, not asserted in a format string.
"""

from __future__ import annotations

from datetime import UTC, datetime


def describe_age(cached_at: str, now: datetime | None = None) -> str:
    """Render a cached_at timestamp as 'March 2025 (18 months ago)'."""
    try:
        dt = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return f"{cached_at} (age unknown)"

    current = now or datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    months = (current.year - dt.year) * 12 + (current.month - dt.month)
    if current.day < dt.day:
        months -= 1

    stamp = dt.strftime("%B %Y")
    if months < 1:
        return f"{stamp} (this month)"
    if months == 1:
        return f"{stamp} (1 month ago)"
    return f"{stamp} ({months} months ago)"

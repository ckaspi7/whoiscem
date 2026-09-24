from __future__ import annotations

import logging
import os
import sqlite3

from langchain.tools import tool

from tools.result import ToolResult

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join("data", "user_data.db")

# The only columns that may ever reach a prompt. They are projected explicitly
# instead of SELECT *, so a database still carrying the older sensitive columns
# — email, phone number, date of birth, family names — cannot leak them into
# model context, a log, or a trace.
_SAFE_FIELDS: tuple[str, ...] = (
    "full_name",
    "gender",
    "birth_year",
    "place_of_birth",
    "hometown",
    "city",
    "marital_status",
    "languages_spoken",
    "favorite_cuisines",
    "hobbies",
    "eye_color",
    "hair_color",
)

# info_type narrows the projection further. Unrecognised values fall back to the
# full safe set rather than failing, since the caller is a language model.
_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "background": ("full_name", "gender", "birth_year", "place_of_birth", "hometown"),
    "location": ("full_name", "city", "hometown"),
    "languages": ("full_name", "languages_spoken"),
    "hobbies": ("full_name", "hobbies", "favorite_cuisines"),
    "interests": ("full_name", "hobbies", "favorite_cuisines"),
    "appearance": ("full_name", "eye_color", "hair_color"),
}


def fields_for(info_type: str) -> tuple[str, ...]:
    """Resolve an info_type to the columns it is allowed to read."""
    return _FIELD_GROUPS.get(info_type.strip().lower(), _SAFE_FIELDS)


def _fetch_user(fields: tuple[str, ...]) -> dict:
    conn = sqlite3.connect(_DB_PATH, timeout=10, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        present = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        # Intersect with what the database actually has: an older database may
        # predate some columns, and a newer one may carry columns we refuse to read.
        selected = [f for f in fields if f in present]
        if not selected:
            return {}

        # Interpolation is safe here: every name originates in _SAFE_FIELDS and
        # was matched against the live schema. No caller input reaches this string.
        columns = ", ".join(selected)
        row = conn.execute(f"SELECT {columns} FROM users WHERE full_name = ?", ("Cem Kaspi",)).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def get_personal_info_result(info_type: str = "") -> ToolResult:
    """Typed lookup: content on success, an error that never becomes context.

    "No personal information found" is success, not failure — the query ran
    fine and legitimately found nothing, which the model should be able to
    relay honestly rather than treat as a broken tool.
    """
    try:
        data = _fetch_user(fields_for(info_type))
        if not data:
            return ToolResult.success("No personal information found.")
        return ToolResult.success(
            "\n".join(f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in data.items() if v is not None)
        )
    except sqlite3.OperationalError as e:
        logger.warning("Personal info database error: %s", e)
        return ToolResult.failure(f"Database error: {e}")
    except Exception as e:
        logger.warning("Personal info retrieval failed: %s", e)
        return ToolResult.failure(str(e))


@tool
def get_personal_info(info_type: str = "") -> str:
    """Fetch non-sensitive personal information about Cem.

    info_type optionally narrows the answer to one of: background, location,
    languages, hobbies, interests, appearance. Omit it for everything available.
    Contact details, date of birth and family information are never returned.
    """
    return get_personal_info_result(info_type).as_tool_string("Error retrieving personal info")

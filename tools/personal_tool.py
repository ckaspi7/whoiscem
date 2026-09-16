from __future__ import annotations

import json
import os
import sqlite3

from langchain.tools import tool

_DB_PATH = os.path.join("data", "user_data.db")


def _fetch_user() -> dict:
    conn = sqlite3.connect(_DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE full_name = ?", ("Cem Kaspi",))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if not row:
        return {}
    return dict(row)


@tool
def get_personal_info(info_type: str = "") -> str:
    """Fetch personal information about Cem from the database."""
    try:
        data = _fetch_user()
        if not data:
            return "No personal information found."
        return "\n".join(f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in data.items() if v is not None)
    except sqlite3.OperationalError as e:
        return f"Database error: {e}"
    except Exception as e:
        return f"Error retrieving personal info: {e}"

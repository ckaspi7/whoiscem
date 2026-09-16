from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Spotify tool tests
# ---------------------------------------------------------------------------

def test_spotify_tool_returns_cache_data(tmp_path, sample_spotify_cache):
    cache_file = tmp_path / "spotify_cache.json"
    cache_file.write_text(json.dumps(sample_spotify_cache))

    with patch("tools.spotify_tool._CACHE_PATH", str(cache_file)):
        from tools.spotify_tool import get_music_taste
        result = get_music_taste.invoke({})

    assert "The Weeknd" in result
    assert "Starboy" in result


def test_spotify_cache_includes_timestamp(tmp_path, sample_spotify_cache):
    cache_file = tmp_path / "spotify_cache.json"
    cache_file.write_text(json.dumps(sample_spotify_cache))

    with patch("tools.spotify_tool._CACHE_PATH", str(cache_file)):
        from tools.spotify_tool import get_music_taste
        result = get_music_taste.invoke({})

    assert "March 2025" in result


def test_spotify_tool_missing_cache_returns_message(tmp_path):
    nonexistent = str(tmp_path / "no_file.json")
    with patch("tools.spotify_tool._CACHE_PATH", nonexistent):
        from tools.spotify_tool import get_music_taste
        result = get_music_taste.invoke({})

    assert "not found" in result.lower() or "run scripts" in result.lower()


# ---------------------------------------------------------------------------
# LinkedIn tool tests
# ---------------------------------------------------------------------------

def test_linkedin_tool_returns_cache_data(tmp_path, sample_linkedin_cache):
    cache_file = tmp_path / "linkedin_cache.json"
    cache_file.write_text(json.dumps(sample_linkedin_cache))

    with patch("tools.linkedin_tool._CACHE_PATH", str(cache_file)):
        from tools.linkedin_tool import get_linkedin_info
        result = get_linkedin_info.invoke({})

    assert "TELUS" in result
    assert "AI/ML Engineer" in result


def test_linkedin_tool_positions_non_empty(tmp_path, sample_linkedin_cache):
    cache_file = tmp_path / "linkedin_cache.json"
    cache_file.write_text(json.dumps(sample_linkedin_cache))

    with patch("tools.linkedin_tool._CACHE_PATH", str(cache_file)):
        from tools.linkedin_tool import get_linkedin_info
        result = get_linkedin_info.invoke({})

    assert "2022-01" in result


def test_linkedin_tool_missing_cache_returns_message(tmp_path):
    nonexistent = str(tmp_path / "no_file.json")
    with patch("tools.linkedin_tool._CACHE_PATH", nonexistent):
        from tools.linkedin_tool import get_linkedin_info
        result = get_linkedin_info.invoke({})

    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# Personal tool tests
# ---------------------------------------------------------------------------

@pytest.fixture
def in_memory_db(tmp_path):
    db_path = str(tmp_path / "test_user_data.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY, full_name TEXT, age INTEGER,
            gender TEXT, city TEXT, email TEXT, hobbies TEXT,
            date_of_birth TEXT, place_of_birth TEXT, mother_name TEXT,
            mother_birth_year INTEGER, father_name TEXT, father_birth_year INTEGER,
            sibling_count INTEGER, sibling_name TEXT, sibling_birth_year INTEGER,
            sibling_gender TEXT, marital_status TEXT, phone_number TEXT,
            hometown TEXT, languages_spoken TEXT, favorite_cuisines TEXT,
            shoe_size INTEGER, height REAL, weight REAL, eye_color TEXT, hair_color TEXT
        )
    """)
    conn.execute(
        "INSERT INTO users (full_name, age, city, email, hobbies) VALUES (?, ?, ?, ?, ?)",
        ("Cem Kaspi", 28, "Vancouver", "cem@example.com", "guitar, cooking"),
    )
    conn.commit()
    conn.close()
    return db_path


def test_personal_tool_returns_valid_data(in_memory_db):
    with patch("tools.personal_tool._DB_PATH", in_memory_db):
        from tools.personal_tool import get_personal_info
        result = get_personal_info.invoke("")

    assert "Cem Kaspi" in result
    assert "Vancouver" in result


def test_personal_tool_handles_missing_user(tmp_path):
    db_path = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY, full_name TEXT, age INTEGER,
            gender TEXT, city TEXT, email TEXT, hobbies TEXT,
            date_of_birth TEXT, place_of_birth TEXT, mother_name TEXT,
            mother_birth_year INTEGER, father_name TEXT, father_birth_year INTEGER,
            sibling_count INTEGER, sibling_name TEXT, sibling_birth_year INTEGER,
            sibling_gender TEXT, marital_status TEXT, phone_number TEXT,
            hometown TEXT, languages_spoken TEXT, favorite_cuisines TEXT,
            shoe_size INTEGER, height REAL, weight REAL, eye_color TEXT, hair_color TEXT
        )
    """)
    conn.commit()
    conn.close()

    with patch("tools.personal_tool._DB_PATH", db_path):
        from tools.personal_tool import get_personal_info
        result = get_personal_info.invoke("")

    assert "no personal information" in result.lower() or "not found" in result.lower()

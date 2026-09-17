from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

# Load .env before collection so skipif guards on OPENAI_API_KEY see the same
# environment the app would. Real environment variables still take precedence,
# so CI secrets win.
load_dotenv()

# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    "Cem Kaspi is an AI/ML Engineer at TELUS Communications Inc. based in Vancouver.",
    "He graduated from the University of British Columbia with a degree in Mechanical Engineering.",
    "Cem co-founded NeoWise, a wearable heating and cooling startup, from 2019 to 2022.",
    "He completed a manufacturing engineering internship at Mercedes-Benz Canada in 2018.",
    "Cem speaks Turkish and English fluently, and beginner-level Spanish.",
]

SAMPLE_SPOTIFY_CACHE = {
    "cached_at": "2025-03-25T00:00:00Z",
    "time_range": "medium_term",
    "top_artists": [
        {"rank": 1, "name": "The Weeknd", "genres": ["canadian pop", "r&b"]},
        {"rank": 2, "name": "Drake", "genres": ["canadian hip hop", "rap"]},
    ],
    "top_tracks": [
        {"rank": 1, "title": "Starboy", "artists": "The Weeknd, Daft Punk"},
        {"rank": 2, "title": "God's Plan", "artists": "Drake"},
    ],
}

SAMPLE_LINKEDIN_CACHE = {
    "cached_at": "2025-03-25T00:00:00Z",
    "headline": "AI/ML Engineer at TELUS",
    "about": "Passionate about GenAI.",
    "positions": [
        {
            "title": "AI/ML Engineer",
            "company": "TELUS Communications Inc.",
            "startDate": "2022-01",
            "endDate": "Present",
            "description": "Building GenAI products.",
            "promotions": [
                {"title": "AI/ML Engineer", "startDate": "2024-09", "endDate": "Present"},
            ],
        }
    ],
}


@pytest.fixture
def sample_chunks() -> list[str]:
    return list(SAMPLE_CHUNKS)


@pytest.fixture
def sample_spotify_cache() -> dict:
    return dict(SAMPLE_SPOTIFY_CACHE)


@pytest.fixture
def sample_linkedin_cache() -> dict:
    return dict(SAMPLE_LINKEDIN_CACHE)


# ---------------------------------------------------------------------------
# Mock OpenAI fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai(monkeypatch):
    """Patches openai.OpenAI so no real API calls are made."""
    mock = MagicMock()
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock)
    return mock


# ---------------------------------------------------------------------------
# Fake Redis fixture (no real server needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_redis():
    import fakeredis
    return fakeredis.FakeRedis(decode_responses=True)

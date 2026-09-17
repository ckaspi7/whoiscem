"""
Refresh Spotify and LinkedIn cache files locally.
Run this periodically (e.g. monthly) to keep data/cache/ up to date.

Requires .env with SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI.
LinkedIn data must be updated manually in data/cache/linkedin_cache.json
(LinkedIn's API access is heavily restricted for personal use).
"""
import json
import os
from datetime import UTC, datetime

from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = os.path.join("data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def refresh_spotify() -> None:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    client_id = os.environ["SPOTIFY_CLIENT_ID"]
    client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
    redirect_uri = os.environ.get("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope="user-top-read user-library-read",
            cache_path=os.path.join(CACHE_DIR, ".spotify_token_cache"),
        )
    )

    raw_artists = sp.current_user_top_artists(limit=10, time_range="medium_term")["items"]
    raw_tracks = sp.current_user_top_tracks(limit=10, time_range="medium_term")["items"]

    artists = [
        {"rank": i + 1, "name": a["name"], "genres": a.get("genres", [])}
        for i, a in enumerate(raw_artists)
    ]
    tracks = [
        {
            "rank": i + 1,
            "title": t["name"],
            "artists": ", ".join(ar["name"] for ar in t["artists"]),
        }
        for i, t in enumerate(raw_tracks)
    ]

    cache = {
        "cached_at": datetime.now(UTC).isoformat(),
        "time_range": "medium_term",
        "top_artists": artists,
        "top_tracks": tracks,
    }

    out_path = os.path.join(CACHE_DIR, "spotify_cache.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"Spotify cache updated → {out_path}")


def bump_linkedin_timestamp() -> None:
    path = os.path.join(CACHE_DIR, "linkedin_cache.json")
    if not os.path.exists(path):
        print("LinkedIn cache not found — edit data/cache/linkedin_cache.json manually.")
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["cached_at"] = datetime.now(UTC).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"LinkedIn cache timestamp bumped → {path}")
    print("Remember to manually update positions if your career info changed.")


if __name__ == "__main__":
    print("Refreshing Spotify cache...")
    refresh_spotify()
    print("\nBumping LinkedIn timestamp...")
    bump_linkedin_timestamp()
    print("\nDone. Commit data/cache/*.json to update the live chatbot.")

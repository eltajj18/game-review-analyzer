"""
Steam review collector.

Usage (standalone):
    python -m src.collect.steam --appid 1086940 --review-type negative --max 5000
"""

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

import requests

REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
DETAILS_URL = "https://store.steampowered.com/api/appdetails"
SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
DB_PATH = Path("data/reviews.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id            TEXT PRIMARY KEY,
    appid                INTEGER NOT NULL,
    language             TEXT,
    text                 TEXT,
    voted_up             INTEGER,
    votes_up             INTEGER,
    votes_funny          INTEGER,
    weighted_vote_score  REAL,
    playtime_at_review   INTEGER,  -- minutes
    playtime_forever     INTEGER,  -- minutes
    early_access         INTEGER,
    received_for_free    INTEGER,
    created_at           INTEGER,  -- unix timestamp
    updated_at           INTEGER
);
"""


def init_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    return conn


def get_game_name(appid: int) -> str:
    """Look up the game's name on the Steam store (falls back to the app ID)."""
    try:
        resp = requests.get(DETAILS_URL, params={"appids": appid}, timeout=15)
        entry = resp.json().get(str(appid), {})
        if entry.get("success"):
            return re.sub(r"[™®©]", "", entry["data"]["name"]).strip()
    except Exception:
        pass
    return f"App {appid}"


def search_games(term: str, limit: int = 8) -> list[tuple[int, str]]:
    """Search the Steam store by name. Returns [(appid, name), ...], best matches first."""
    resp = requests.get(SEARCH_URL, params={"term": term, "l": "english", "cc": "US"}, timeout=15)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [(int(i["id"]), re.sub(r"[™®©]", "", i["name"]).strip())
            for i in items if i.get("type", "app") == "app"][:limit]


def _normalize(name: str) -> str:
    """"Baldur's Gate 3™" -> "baldursgate3", so small differences don't block a match."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_game(term: str, interactive: bool | None = None) -> tuple[int, str]:
    """
    Turn a game name into (appid, name).
    Exact name match or a single result is used directly; otherwise the user picks
    from a list (or the top result is used when not running in a terminal).
    """
    results = search_games(term)
    if not results:
        raise SystemExit(f'No Steam game found for "{term}". Check the spelling or use --appid.')

    exact = [r for r in results if _normalize(r[1]) == _normalize(term)]
    if len(exact) == 1:
        return exact[0]
    if len(results) == 1:
        return results[0]

    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        print(f'  several matches for "{term}", using the top one: {results[0][1]}')
        return results[0]

    print(f'Several games match "{term}":')
    for i, (appid, name) in enumerate(results, 1):
        print(f"  {i}. {name}  (app {appid})")
    while True:
        choice = input(f"Pick a number [1-{len(results)}, Enter = 1]: ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(results):
            return results[int(choice) - 1]
        print("  not a valid choice, try again")


def parse_review(r: dict, appid: int) -> tuple:
    author = r.get("author", {})
    return (
        r["recommendationid"],
        appid,
        r.get("language"),
        r.get("review", ""),
        int(r.get("voted_up", False)),
        r.get("votes_up", 0),
        r.get("votes_funny", 0),
        float(r.get("weighted_vote_score", 0) or 0),
        author.get("playtime_at_review"),
        author.get("playtime_forever"),
        int(r.get("written_during_early_access", False)),
        int(r.get("received_for_free", False)),
        r.get("timestamp_created"),
        r.get("timestamp_updated"),
    )


def fetch_reviews(appid, language="english", review_type="all", max_reviews=None, delay=1.0):
    """Yield batches of reviews, following Steam's cursor pagination."""
    params = {
        "json": 1,
        "filter": "recent",
        "language": language,
        "review_type": review_type,
        "purchase_type": "all",
        "num_per_page": 100,
        "cursor": "*",
    }
    seen_cursors = set()
    fetched = 0

    while True:
        resp = requests.get(REVIEWS_URL.format(appid=appid), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success") != 1:
            raise RuntimeError(f"Steam API returned failure: {data}")

        reviews = data.get("reviews", [])
        cursor = data.get("cursor")
        if not reviews or cursor in seen_cursors:
            break

        yield reviews
        fetched += len(reviews)
        if max_reviews and fetched >= max_reviews:
            break

        seen_cursors.add(cursor)
        params["cursor"] = cursor
        time.sleep(delay)


def collect(appid, language="english", review_type="all", max_reviews=None,
            delay=1.0, db_path: Path = DB_PATH) -> int:
    """Fetch reviews into SQLite. Returns how many were fetched (duplicates are skipped)."""
    conn = init_db(db_path)
    total = 0
    for batch in fetch_reviews(appid, language, review_type, max_reviews, delay):
        conn.executemany(
            "INSERT OR IGNORE INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [parse_review(r, appid) for r in batch],
        )
        conn.commit()
        total += len(batch)
        print(f"  fetched {total} reviews...", end="\r")
    print()
    conn.close()
    return total


def main():
    parser = argparse.ArgumentParser(description="Collect Steam reviews for a game.")
    parser.add_argument("--appid", type=int, required=True)
    parser.add_argument("--language", default="english")
    parser.add_argument("--review-type", default="all", choices=["all", "positive", "negative"])
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    collect(args.appid, args.language, args.review_type, args.max, args.delay, args.db)


if __name__ == "__main__":
    main()

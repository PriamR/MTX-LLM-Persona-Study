"""Live ingestion of Steam reviews for the smoke test.

Reuses Senti-Minted's keyless ``get_reviews`` page fetch — which returns the raw
appreviews payload with the author block intact — and paginates over it, keeping
the per-user author fields personas are built from. (Senti-Minted's own
``iter_reviews`` drops those fields, so we go one level lower and page ourselves.)

No key needed: the storefront appreviews endpoint is public.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

from llmsonas.config import SENTI_MINTED_PATH

PAGE_CAP = 60  # safety bound on pagination


def _steam_client():
    """Import the Senti-Minted Steam client on demand.

    Kept lazy so the offline dump loaders can import this module (for
    ``UserRecord``) without a Senti-Minted checkout on the path — only the live
    crawl actually needs the client.
    """
    if str(SENTI_MINTED_PATH) not in sys.path:
        sys.path.insert(0, str(SENTI_MINTED_PATH))
    from app.clients import steam

    return steam


@dataclass
class UserRecord:
    """One reviewer's behavioural footprint, as pulled from appreviews."""

    steamid: str
    voted_up: bool
    playtime_forever: int   # minutes, all-time in this game
    playtime_at_review: int  # minutes, at the moment they reviewed
    num_games_owned: int
    num_reviews: int
    timestamp: int          # unix, review creation
    review: str


def _to_record(raw: dict) -> UserRecord | None:
    author = raw.get("author") or {}
    steamid = author.get("steamid")
    if not steamid:
        return None
    return UserRecord(
        steamid=str(steamid),
        voted_up=bool(raw.get("voted_up", False)),
        playtime_forever=int(author.get("playtime_forever", 0)),
        playtime_at_review=int(author.get("playtime_at_review", 0)),
        num_games_owned=int(author.get("num_games_owned", 0)),
        num_reviews=int(author.get("num_reviews", 0)),
        timestamp=int(raw.get("timestamp_created", 0)),
        review=(raw.get("review") or "").strip(),
    )


async def _crawl(appid: int, target: int, language: str) -> list[UserRecord]:
    steam = _steam_client()
    records: dict[str, UserRecord] = {}  # keyed by steamid to de-dupe
    cursor, seen = "*", set()
    for _ in range(PAGE_CAP):
        page = await steam.get_reviews(
            appid, cursor=cursor, num_per_page=100, filter="recent", language=language
        )
        reviews = page.get("reviews", [])
        if not reviews:
            break
        for raw in reviews:
            rec = _to_record(raw)
            if rec and rec.review and rec.steamid not in records:
                records[rec.steamid] = rec
        if len(records) >= target:
            break
        cursor = page.get("cursor", "")
        if not cursor or cursor in seen:
            break
        seen.add(cursor)
        await asyncio.sleep(0.3)  # be polite between pages
    return list(records.values())[:target]


def fetch_user_records(
    appid: int, target: int = 300, language: str = "english"
) -> list[UserRecord]:
    """Pull ~``target`` distinct-user reviews (newest-first), author payload kept."""
    return asyncio.run(_crawl(appid, target, language))

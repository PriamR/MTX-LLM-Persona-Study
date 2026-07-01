"""Historical Steam review dumps — the offline time-split source.

Two shapes are handled:

* the forgemaster CS:GO dump — ~7 GB of CSVs sharded by review id, one DuckDB
  pass filters an appid out and caches it as Parquet so reruns are instant;
* the "Steam Reviews 2024" per-app dump — one CSV per appid inside a zip, a
  different (author-prefixed) schema and, notably, **no review text**. Small
  enough to read straight through pandas, so no DuckDB or cache is needed.

Either way personas are built from pre-event rows and scored against the
post-event recommend ratio: the held-out time-split that keeps every score a
genuine prediction rather than a fit.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from llmsonas.config import ROOT
from llmsonas.data.ingest import UserRecord

DUMP_DIR = ROOT / "Historical Data"
CACHE_DIR = ROOT / "data"

# "Steam Reviews 2024" per-app dump (one CSV per appid inside the zip).
HD2_ZIP = DUMP_DIR / "Steam 2024 reviews.zip"
HD2_MEMBER = "SteamReviews2024/{appid}.csv"


def _cache_path(appid: int) -> Path:
    return CACHE_DIR / f"reviews_{appid}.parquet"


def extract_game(appid: int, *, dump_dir: Path = DUMP_DIR, force: bool = False) -> Path:
    """Filter one appid's rows out of the CSV dump into a cached Parquet file."""
    import duckdb

    out = _cache_path(appid)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    glob = (Path(dump_dir) / "*.csv").as_posix()
    duckdb.sql(
        f"""
        COPY (
          SELECT
            cast(steamid AS VARCHAR)                           AS steamid,
            lower(cast(voted_up AS VARCHAR)) IN ('true','1','t') AS voted_up,
            try_cast(playtime_forever   AS BIGINT)             AS playtime_forever,
            try_cast(playtime_at_review AS BIGINT)             AS playtime_at_review,
            try_cast(num_games_owned    AS BIGINT)             AS num_games_owned,
            try_cast(num_reviews        AS BIGINT)             AS num_reviews,
            cast(review AS VARCHAR)                            AS review,
            try_cast(unix_timestamp_created AS BIGINT)         AS ts
          FROM read_csv(
                 '{glob}',
                 all_varchar=true, header=true,
                 delim=',', quote='"', escape='"',
                 strict_mode=false, ignore_errors=true,
                 max_line_size=10000000
               )
          WHERE try_cast(appid AS BIGINT) = {appid}
        ) TO '{out.as_posix()}' (FORMAT parquet)
        """
    )
    return out


def load_game(appid: int) -> pd.DataFrame:
    import duckdb

    path = _cache_path(appid).as_posix()
    return duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()


def load_hd2(appid: int = 553850, *, zip_path: Path = HD2_ZIP) -> pd.DataFrame:
    """Load one appid from the "Steam Reviews 2024" per-app dump.

    The schema is author-prefixed and carries no review text, so the columns are
    renamed to the shared shape (``voted_up``, ``playtime_*``, ``num_*``, ``ts``)
    and ``review`` is left empty. ``edited`` flags rows whose stored verdict was
    changed after creation — heavy on this game because the May-2024 event was
    reversed days later and many reviews were rewritten.
    """
    member = HD2_MEMBER.format(appid=appid)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        df = pd.read_csv(
            fh,
            usecols=[
                "language", "timestamp_created", "timestamp_updated", "voted_up",
                "votes_up", "steam_purchase", "received_for_free",
                "written_during_early_access",
                "author_steamid", "author_num_games_owned", "author_num_reviews",
                "author_playtime_forever", "author_playtime_at_review",
            ],
        )
    df = df.rename(
        columns={
            "author_steamid": "steamid",
            "author_playtime_forever": "playtime_forever",
            "author_playtime_at_review": "playtime_at_review",
            "author_num_games_owned": "num_games_owned",
            "author_num_reviews": "num_reviews",
            "timestamp_created": "ts",
            "written_during_early_access": "early_access",
        }
    )
    df["voted_up"] = df["voted_up"].astype(bool)
    df["edited"] = df["timestamp_updated"] > df["ts"] + 60  # later rewrite
    df["review"] = ""  # this dump ships no review text
    for col in ("steam_purchase", "received_for_free", "early_access"):
        df[col] = df[col].astype(str).str.lower().isin(("true", "1", "t"))
    return df


def recommend_ratio(
    df: pd.DataFrame, lo: int | None = None, hi: int | None = None
) -> tuple[float, int]:
    """Recommend ratio and row count within [lo, hi) on the ``ts`` column."""
    mask = pd.Series(True, index=df.index)
    if lo is not None:
        mask &= df["ts"] >= lo
    if hi is not None:
        mask &= df["ts"] < hi
    sub = df[mask]
    ratio = float(sub["voted_up"].mean()) if len(sub) else float("nan")
    return ratio, int(len(sub))


def _int(x) -> int:
    """Coerce a possibly-missing numeric cell (NaN/None) to a plain int."""
    try:
        if x is None or x != x:  # NaN
            return 0
        return int(x)
    except (TypeError, ValueError):
        return 0


def to_user_records(df: pd.DataFrame, *, require_review: bool = True) -> list[UserRecord]:
    """Rows -> ``UserRecord``s.

    ``require_review=False`` keeps rows that carry no review text (the 2024 dump),
    where personas are grounded in behavioural facts alone rather than text.
    """
    records: list[UserRecord] = []
    for row in df.itertuples(index=False):
        review = (getattr(row, "review", "") or "").strip()
        if require_review and not review:
            continue
        records.append(
            UserRecord(
                steamid=str(row.steamid),
                voted_up=bool(row.voted_up),
                playtime_forever=_int(row.playtime_forever),
                playtime_at_review=_int(row.playtime_at_review),
                num_games_owned=_int(row.num_games_owned),
                num_reviews=_int(row.num_reviews),
                timestamp=_int(row.ts),
                review=review,
                votes_up=_int(getattr(row, "votes_up", 0)),
                steam_purchase=bool(getattr(row, "steam_purchase", True)),
                received_for_free=bool(getattr(row, "received_for_free", False)),
                early_access=bool(getattr(row, "early_access", False)),
            )
        )
    return records

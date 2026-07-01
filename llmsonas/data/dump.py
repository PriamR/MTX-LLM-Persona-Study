"""Historical Steam review dump (forgemaster) — the offline time-split source.

One DuckDB pass filters a game's rows out of the ~7 GB of CSVs and caches them as
Parquet, so repeated runs are instant. Personas are built from pre-event rows and
scored against the post-event recommend ratio: the held-out time-split that keeps
every score a genuine prediction rather than a fit.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from llmsonas.config import ROOT
from llmsonas.data.ingest import UserRecord

DUMP_DIR = ROOT / "Historical Data"
CACHE_DIR = ROOT / "data"


def _cache_path(appid: int) -> Path:
    return CACHE_DIR / f"reviews_{appid}.parquet"


def extract_game(appid: int, *, dump_dir: Path = DUMP_DIR, force: bool = False) -> Path:
    """Filter one appid's rows out of the CSV dump into a cached Parquet file."""
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
    path = _cache_path(appid).as_posix()
    return duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()


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


def to_user_records(df: pd.DataFrame) -> list[UserRecord]:
    records: list[UserRecord] = []
    for row in df.itertuples(index=False):
        review = (row.review or "").strip()
        if not review:
            continue
        records.append(
            UserRecord(
                steamid=str(row.steamid),
                voted_up=bool(row.voted_up),
                playtime_forever=int(row.playtime_forever or 0),
                playtime_at_review=int(row.playtime_at_review or 0),
                num_games_owned=int(row.num_games_owned or 0),
                num_reviews=int(row.num_reviews or 0),
                timestamp=int(row.ts or 0),
                review=review,
            )
        )
    return records

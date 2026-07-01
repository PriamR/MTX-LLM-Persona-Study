"""Cross-app monetization-exposure enrichment — the missing answer axis.

The 70B run showed that the whole output distribution has to come from persona
feature heterogeneity, and none of the existing markers (investment, vocalness,
channel, tenure) encode the axis that actually split the real population on a
monetisation change: prior exposure to games financed by in-game purchases. This
module derives that axis from the dump itself, with no external API and no
post-event information:

* **Candidate apps** are the top-K members of the 2024 per-app dump by size — a
  mechanical review-volume rule, the same list for every case.
* An app counts as **free-access** at a given cutoff when the majority of its
  *pre-cutoff* reviews were not bought on Steam (``steam_purchase`` false): on
  Steam that pattern is the free-to-play distribution model. Measuring the share
  before the case's own cutoff keeps the label time-valid (CS:GO reads as paid
  in 2015 and free in 2020, matching reality).
* A user's **footprint** is their pre-cutoff reviews on candidate apps; the
  exposure band is the free-access share of that footprint, cut at mechanical
  thirds. Users with no footprint get no clause — absence of data, mirroring the
  ``unknown`` library band.

Everything is a frozen mechanical rule applied identically across cases, so the
enrichment adds a real behavioural axis without tuning anything to a known
outcome (Approach §3.4 guardrails hold).
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llmsonas.data.dump import CACHE_DIR, HD2_MEMBER, HD2_ZIP

CANDIDATE_K = 60          # top-K apps by dump size form the candidate set
# Free-access when the overwhelming majority of pre-cutoff reviews were not
# Steam purchases. Validated against apps with known 2015 monetization before
# any scoring: true F2P titles measure 0.93-1.00 (Dota 2, TF2, Warframe,
# Unturned, Robocraft) while paid titles top out at 0.59 (L4D2 — free weekends
# and bundle keys), so 0.8 sits inside an empty gap. A simple majority cut
# would mislabel key-heavy paid games like L4D2.
FREE_ACCESS_CUT = 0.8
BAND_CUTS = (1 / 3, 2 / 3)  # premium | mixed | f2p_leaning thirds


@dataclass(frozen=True)
class Exposure:
    """One user's pre-event cross-app monetization exposure."""

    n_other: int          # pre-cutoff reviews on candidate apps
    f2p_share: float      # share of those on free-access apps (nan when none)
    band: str             # none | premium | mixed | f2p_leaning


def candidate_apps(
    zip_path: Path = HD2_ZIP, *, k: int = CANDIDATE_K, exclude: int | None = None
) -> list[int]:
    """The K largest per-app CSVs in the dump (minus the case's own app).

    Compressed size is a pure review-volume proxy, so the rule is mechanical and
    case-independent; apps released after a case's cutoff simply contribute no
    pre-cutoff rows.
    """
    with zipfile.ZipFile(zip_path) as zf:
        ranked = sorted(
            (i for i in zf.infolist() if i.filename.endswith(".csv")),
            key=lambda i: i.compress_size,
            reverse=True,
        )
    out: list[int] = []
    for info in ranked:
        appid = int(info.filename.rsplit("/", 1)[1].removesuffix(".csv"))
        if appid != exclude:
            out.append(appid)
        if len(out) == k:
            break
    return out


def _footprint_cache(target_appid: int, cutoff: int) -> Path:
    return CACHE_DIR / f"footprint_{target_appid}_{cutoff}.parquet"


def build_footprint(
    target_appid: int,
    cutoff: int,
    *,
    zip_path: Path = HD2_ZIP,
    k: int = CANDIDATE_K,
    force: bool = False,
) -> pd.DataFrame:
    """Pre-cutoff cross-app reviews of the target app's pre-cutoff reviewers.

    One pass over the candidate apps, cached as Parquet (the scan reads a few GB
    out of the zip, the result is a few MB). Columns: ``steamid`` (str),
    ``appid``, ``app_free_share`` — the candidate app's own pre-cutoff
    non-purchase share, from which the app's free-access label is derived.
    """
    import duckdb

    cache = _footprint_cache(target_appid, cutoff)
    if cache.exists() and not force:
        return duckdb.sql(f"SELECT * FROM read_parquet('{cache.as_posix()}')").df()

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(HD2_MEMBER.format(appid=target_appid)) as fh:
            target = pd.read_csv(fh, usecols=["author_steamid", "timestamp_created"])
        ids = set(target.loc[target["timestamp_created"] < cutoff, "author_steamid"])

        rows: list[tuple[str, int, float]] = []
        for appid in candidate_apps(zip_path, k=k, exclude=target_appid):
            with zf.open(HD2_MEMBER.format(appid=appid)) as fh:
                df = pd.read_csv(
                    fh, usecols=["author_steamid", "timestamp_created", "steam_purchase"]
                )
            df = df[df["timestamp_created"] < cutoff]
            if not len(df):
                continue
            free_share = float(
                (~df["steam_purchase"].astype(str).str.lower().isin(("true", "1", "t"))).mean()
            )
            for sid in df.loc[df["author_steamid"].isin(ids), "author_steamid"]:
                rows.append((str(sid), appid, free_share))

    foot = pd.DataFrame(rows, columns=["steamid", "appid", "app_free_share"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    duckdb.sql(f"COPY (SELECT * FROM foot) TO '{cache.as_posix()}' (FORMAT parquet)")
    return foot


def exposures(footprint: pd.DataFrame) -> dict[str, Exposure]:
    """Per-user exposure bands from a footprint frame (steamid -> Exposure).

    Band cuts are fixed thirds of the free-access share — mechanical, identical
    across cases, frozen before any answer is seen.
    """
    out: dict[str, Exposure] = {}
    if not len(footprint):
        return out
    flagged = footprint.assign(
        is_free=footprint["app_free_share"] >= FREE_ACCESS_CUT
    )
    grouped = flagged.groupby("steamid").agg(
        n_other=("appid", "count"), n_free=("is_free", "sum")
    )
    lo, hi = BAND_CUTS
    for sid, row in grouped.iterrows():
        share = row.n_free / row.n_other
        band = "premium" if share < lo else "mixed" if share <= hi else "f2p_leaning"
        out[str(sid)] = Exposure(int(row.n_other), float(share), band)
    return out


def exposure_clause(exp: Exposure | None) -> str | None:
    """The bio sentence for one user's exposure band — facts only, no valence.

    ``None`` (no footprint) renders nothing: we state what the record shows and
    stay silent where it shows nothing, exactly like the unknown-library band.
    """
    if exp is None or exp.band == "none" or exp.n_other == 0:
        return None
    if exp.n_other == 1:
        kind = (
            "a free-to-play title that sells in-game items"
            if exp.band == "f2p_leaning"
            else "a paid, buy-upfront title"
        )
        return f"The one other popular game they had reviewed before this is {kind}."
    lead = f"Of the {exp.n_other} other popular games they had reviewed before this, "
    if exp.band == "premium":
        return lead + "all are paid, buy-upfront titles."
    if exp.band == "mixed":
        return lead + "some are free-to-play titles that sell in-game items."
    return lead + "most are free-to-play titles that sell in-game items."

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
    """One user's pre-event cross-app monetization exposure and disposition."""

    n_other: int          # pre-cutoff reviews on candidate apps
    f2p_share: float      # share of those on free-access apps (nan when none)
    band: str             # none | premium | mixed | f2p_leaning
    # Dispositional positivity: how the user judged their OTHER games pre-event.
    # Cross-game behavioural history (Method-Technical §A.3 mean(voted_up)), not
    # the withheld verdict on the target game — it licenses *disposition*, the
    # axis that lets a persona plausibly stay positive under a change.
    rec_share: float = float("nan")   # share of footprint reviews that recommend
    disposition: str = "unknown"      # all | most | half | few | unknown


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
        cached = duckdb.sql(f"SELECT * FROM read_parquet('{cache.as_posix()}')").df()
        if "voted_up" in cached.columns:
            return cached
        # cache predates the disposition marker — fall through and rebuild

    def _true(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().isin(("true", "1", "t"))

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(HD2_MEMBER.format(appid=target_appid)) as fh:
            target = pd.read_csv(fh, usecols=["author_steamid", "timestamp_created"])
        ids = set(target.loc[target["timestamp_created"] < cutoff, "author_steamid"])

        rows: list[tuple[str, int, float, bool]] = []
        for appid in candidate_apps(zip_path, k=k, exclude=target_appid):
            with zf.open(HD2_MEMBER.format(appid=appid)) as fh:
                df = pd.read_csv(
                    fh,
                    usecols=[
                        "author_steamid", "timestamp_created", "steam_purchase", "voted_up",
                    ],
                )
            df = df[df["timestamp_created"] < cutoff]
            if not len(df):
                continue
            free_share = float((~_true(df["steam_purchase"])).mean())
            hit = df[df["author_steamid"].isin(ids)]
            for sid, up in zip(hit["author_steamid"], _true(hit["voted_up"])):
                rows.append((str(sid), appid, free_share, bool(up)))

    foot = pd.DataFrame(rows, columns=["steamid", "appid", "app_free_share", "voted_up"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    duckdb.sql(f"COPY (SELECT * FROM foot) TO '{cache.as_posix()}' (FORMAT parquet)")
    return foot


def _disposition(rec_share: float) -> str:
    """Mechanical cuts, frozen: all (=1), most (>=2/3), half (>1/3), few."""
    if rec_share != rec_share:  # NaN — footprint lacks verdicts (old cache/tests)
        return "unknown"
    if rec_share >= 1.0:
        return "all"
    if rec_share >= 2 / 3:
        return "most"
    if rec_share > 1 / 3:
        return "half"
    return "few"


def exposures(footprint: pd.DataFrame) -> dict[str, Exposure]:
    """Per-user exposure bands from a footprint frame (steamid -> Exposure).

    Band cuts are fixed thirds of the free-access share, disposition cuts fixed
    fractions of the cross-app recommend share — mechanical, identical across
    cases, frozen before any answer is seen.
    """
    out: dict[str, Exposure] = {}
    if not len(footprint):
        return out
    flagged = footprint.assign(
        is_free=footprint["app_free_share"] >= FREE_ACCESS_CUT
    )
    has_verdicts = "voted_up" in footprint.columns
    aggs = dict(n_other=("appid", "count"), n_free=("is_free", "sum"))
    if has_verdicts:
        aggs["rec_share"] = ("voted_up", "mean")
    grouped = flagged.groupby("steamid").agg(**aggs)
    lo, hi = BAND_CUTS
    for sid, row in grouped.iterrows():
        share = row.n_free / row.n_other
        band = "premium" if share < lo else "mixed" if share <= hi else "f2p_leaning"
        rec = float(row.rec_share) if has_verdicts else float("nan")
        out[str(sid)] = Exposure(
            int(row.n_other), float(share), band, rec, _disposition(rec)
        )
    return out


_DISPOSITION_PLURAL = {
    "all": " They recommended all of them.",
    "most": " They recommended most of them.",
    "half": " They recommended about half of them.",
    "few": " They recommended few of them.",
}


def exposure_clause(exp: Exposure | None) -> str | None:
    """The bio sentences for one user's exposure + disposition — facts only.

    ``None`` (no footprint) renders nothing: we state what the record shows and
    stay silent where it shows nothing, exactly like the unknown-library band.
    The disposition sentence is the user's own cross-game verdict history —
    behavioural fact, not the withheld target verdict.
    """
    if exp is None or exp.band == "none" or exp.n_other == 0:
        return None
    if exp.n_other == 1:
        kind = (
            "a free-to-play title that sells in-game items"
            if exp.band == "f2p_leaning"
            else "a paid, buy-upfront title"
        )
        base = f"The one other popular game they had reviewed before this is {kind}."
        if exp.disposition == "all":
            return base + " They recommended it."
        if exp.disposition == "few":
            return base + " They did not recommend it."
        return base
    lead = f"Of the {exp.n_other} other popular games they had reviewed before this, "
    if exp.band == "premium":
        base = lead + "all are paid, buy-upfront titles."
    elif exp.band == "mixed":
        base = lead + "some are free-to-play titles that sell in-game items."
    else:
        base = lead + "most are free-to-play titles that sell in-game items."
    return base + _DISPOSITION_PLURAL.get(exp.disposition, "")

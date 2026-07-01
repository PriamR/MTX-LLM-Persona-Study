"""Behavioural segmentation — turning raw per-user numbers into population-relative
traits so no two personas read the same.

The situation-framing bio used to state raw facts ("36 hours, 9 reviews") and then
assert *the same* clause — "a title they have genuinely invested in" — for every
persona, which flattened 100 different players into one voice. The fix is to bin
each user against the *population* they are drawn from: 36 hours is heavy in one
community and trivial in another, so investment/vocalness/loyalty are only
meaningful relative to the pool.

Bands are plain population quantiles computed mechanically from the pre-event pool
(frozen before any answer is seen), and the same thresholds serve every case — so
this adds texture without tuning anything to a known outcome (Approach §3.4
guardrails: facts only, fixed template, same across cases).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llmsonas.data.ingest import UserRecord


@dataclass(frozen=True)
class Bands:
    """Population quantile cut-points, in the records' native units (minutes /
    counts / seconds). Computed once over the pool; every persona is binned
    against these."""

    playtime: tuple[float, float, float, float]   # p25, p50, p75, p90 of playtime_forever
    reviews: tuple[float, float, float]            # p50, p75, p90 of num_reviews
    owned: tuple[float, float]                     # p25, p75 of num_games_owned (known only)
    owned_known: bool                              # False when the dump ships no library size
    tenure: tuple[float, float]                    # p33, p66 of (event - review) seconds
    event_cutoff: int                              # the event date, to age each review against


def _q(values: np.ndarray, qs: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(np.quantile(values, q)) for q in qs)


def population_bands(records: list[UserRecord], event_cutoff: int) -> Bands:
    """Derive the binning thresholds from the pool the personas are sampled from.

    ``event_cutoff`` ages each review (how long *before* the decision it was
    written) into a tenure band. We deliberately take tenure from the review
    timestamp — a fixed, review-time fact — rather than from the dump-time
    ``last_played`` / ``playtime_last_two_weeks`` fields, which for a years-old
    event describe activity at scrape time and would leak post-event information.
    """
    pt = np.asarray([r.playtime_forever for r in records], dtype=float)
    nr = np.asarray([r.num_reviews for r in records], dtype=float)
    owned = np.asarray([r.num_games_owned for r in records], dtype=float)
    known = owned[owned > 0]
    tenure = np.asarray([max(0, event_cutoff - r.timestamp) for r in records], dtype=float)
    return Bands(
        playtime=_q(pt, (0.25, 0.50, 0.75, 0.90)),          # type: ignore[arg-type]
        reviews=_q(nr, (0.50, 0.75, 0.90)),                  # type: ignore[arg-type]
        owned=(_q(known, (0.25, 0.75)) if len(known) else (0.0, 0.0)),  # type: ignore[assignment]
        owned_known=bool(len(known) >= max(10, 0.2 * len(records))),
        tenure=_q(tenure, (0.33, 0.66)),                     # type: ignore[assignment]
        event_cutoff=int(event_cutoff),
    )


@dataclass(frozen=True)
class Segment:
    """A persona's behavioural bucket on each axis, plus the compound flags that
    the raw numbers alone don't surface."""

    investment: str      # light | casual | regular | dedicated | hardcore
    vocalness: str       # quiet | occasional | vocal | prolific
    library: str         # focused | moderate | broad | unknown
    loyal_mono: bool     # heavy hours + a small library => near-single-game player
    review_timing: str   # early | veteran | mid
    channel: str         # steam | key | free — how they acquired the game (a stake signal)
    early_access: bool   # backed the game during early access
    tenure: str          # recent | established | veteran — how long before the event they reviewed


def _bin(value: float, cuts: tuple[float, ...], names: tuple[str, ...]) -> str:
    """Right-open banding: the first name whose upper cut ``value`` falls under,
    else the last name. ``len(names) == len(cuts) + 1``."""
    for cut, name in zip(cuts, names):
        if value < cut:
            return name
    return names[-1]


def segment_record(r: UserRecord, bands: Bands) -> Segment:
    """Assign one user to a bucket on each behavioural axis, relative to the pool."""
    investment = _bin(
        r.playtime_forever, bands.playtime,
        ("light", "casual", "regular", "dedicated", "hardcore"),
    )
    vocalness = _bin(
        r.num_reviews, bands.reviews,
        ("quiet", "occasional", "vocal", "prolific"),
    )

    library = "unknown"
    if bands.owned_known and r.num_games_owned > 0:
        library = _bin(r.num_games_owned, bands.owned, ("focused", "moderate", "broad"))

    # High hours poured into this game by someone who owns few others: the
    # loyal, single-game devotee the raw counts hide (playtime high AND library small).
    loyal_mono = investment in ("dedicated", "hardcore") and library == "focused"

    # Where in their playtime they reviewed: an early verdict they then played
    # past, vs a judgement formed only after serious time.
    pf, par = r.playtime_forever, r.playtime_at_review
    if pf > 0 and par > 0:
        frac = par / pf
        review_timing = "early" if frac < 0.30 else "veteran" if frac > 0.80 else "mid"
    else:
        review_timing = "mid"

    # How they got the game — a direct signal of financial stake, which is what
    # a monetisation change (paid MTX, paid→free) actually acts on.
    channel = "free" if r.received_for_free else "steam" if r.steam_purchase else "key"

    # How long before the event they reviewed (review-time fact, not dump-time).
    tenure_secs = max(0, bands.event_cutoff - r.timestamp)
    tenure = _bin(tenure_secs, bands.tenure, ("recent", "established", "veteran"))

    return Segment(
        investment=investment,
        vocalness=vocalness,
        library=library,
        loyal_mono=loyal_mono,
        review_timing=review_timing,
        channel=channel,
        early_access=bool(r.early_access),
        tenure=tenure,
    )

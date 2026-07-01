"""Exposure is the axis added to fix the 70B distributional collapse, so its
mechanics get tests: band assignment from a footprint, the L4D2-style paid app
staying below the free-access cut, clause wording, and the bio integration."""
from __future__ import annotations

import pandas as pd

from llmsonas.construction.exposure import (
    Exposure,
    exposure_clause,
    exposures,
)
from llmsonas.construction.profile import situation_bio
from llmsonas.construction.segment import population_bands
from llmsonas.data.ingest import UserRecord

EVENT = 1_600_000_000


def _footprint(rows):
    return pd.DataFrame(rows, columns=["steamid", "appid", "app_free_share"])


def test_bands_from_footprint():
    foot = _footprint(
        [
            ("prem", 1, 0.10), ("prem", 2, 0.20),                # all paid
            ("mix", 1, 0.10), ("mix", 3, 0.95),                  # half free
            ("f2p", 3, 0.95), ("f2p", 4, 1.00), ("f2p", 5, 0.99),
        ]
    )
    out = exposures(foot)
    assert out["prem"].band == "premium"
    assert out["mix"].band == "mixed"
    assert out["f2p"].band == "f2p_leaning"
    assert "absent" not in out  # no-footprint users simply aren't in the dict


def test_key_heavy_paid_app_is_not_free_access():
    # L4D2 measured 0.59 non-purchase pre-cutoff (free weekends / bundle keys)
    # yet was a paid title; true F2P apps measure >= 0.93. The cut must separate.
    foot = _footprint([("u", 550, 0.59), ("u", 570, 1.00)])
    assert exposures(foot)["u"].band == "mixed"  # one paid + one free, not two free


def test_clause_wording_by_band():
    assert exposure_clause(None) is None
    assert exposure_clause(Exposure(0, float("nan"), "none")) is None
    one = exposure_clause(Exposure(1, 1.0, "f2p_leaning"))
    assert one is not None and one.startswith("The one other popular game")
    prem = exposure_clause(Exposure(3, 0.0, "premium"))
    mixed = exposure_clause(Exposure(3, 0.5, "mixed"))
    f2p = exposure_clause(Exposure(3, 1.0, "f2p_leaning"))
    assert "all are paid" in prem
    assert "some are free-to-play" in mixed
    assert "most are free-to-play" in f2p
    # Facts only — no valence, no verdict words.
    for clause in (one, prem, mixed, f2p):
        low = clause.lower()
        assert "positive" not in low and "negative" not in low
        assert "angry" not in low and "betray" not in low


def test_bio_carries_exposure_clause_only_when_present():
    pop = [
        UserRecord(
            steamid=str(i), voted_up=True, playtime_forever=pt, playtime_at_review=pt // 2,
            num_games_owned=50, num_reviews=3, timestamp=EVENT - 86400, review="",
        )
        for i, pt in enumerate((30, 120, 600, 1800, 6000, 20000))
    ]
    bands = population_bands(pop, EVENT)
    change = "The game now costs money to keep playing."
    plain = situation_bio(pop[0], change, bands)
    enriched = situation_bio(pop[0], change, bands, Exposure(4, 0.9, "f2p_leaning"))
    assert "free-to-play" not in plain
    assert "free-to-play titles that sell in-game items" in enriched
    # The change clause stays terminal in both — the template is unchanged.
    assert plain.endswith(change) and enriched.endswith(change)

"""Segmentation is the guard against 100 identical personas, so it gets a test:
distinct users must land in distinct bands, and the compound flags must fire."""
from __future__ import annotations

from llmsonas.construction.profile import situation_bio
from llmsonas.construction.segment import population_bands, segment_record
from llmsonas.data.ingest import UserRecord


def _rec(pt, par, owned, nrev, up=True):
    return UserRecord(
        steamid="x", voted_up=up, playtime_forever=pt, playtime_at_review=par,
        num_games_owned=owned, num_reviews=nrev, timestamp=0, review="",
    )


def _population():
    # A spread of playtimes / review counts / library sizes to bin against.
    return [
        _rec(pt, pt // 2, owned, nrev)
        for pt in (30, 120, 600, 1800, 6000, 20000)
        for owned, nrev in ((5, 1), (50, 3), (400, 30))
    ]


def test_bands_and_bins_separate_users():
    pop = _population()
    bands = population_bands(pop)
    light = segment_record(_rec(20, 10, 300, 1), bands)
    hard = segment_record(_rec(30000, 15000, 300, 1), bands)
    assert light.investment == "light"
    assert hard.investment == "hardcore"
    assert light.investment != hard.investment


def test_vocalness_scales_with_reviews():
    pop = _population()
    bands = population_bands(pop)
    quiet = segment_record(_rec(600, 300, 100, 1), bands)
    loud = segment_record(_rec(600, 300, 100, 500), bands)
    assert quiet.vocalness == "quiet"
    assert loud.vocalness == "prolific"


def test_loyal_mono_flag_needs_hours_and_small_library():
    pop = _population()
    bands = population_bands(pop)
    loyal = segment_record(_rec(30000, 15000, 3, 2), bands)   # heavy hours, tiny library
    collector = segment_record(_rec(30000, 15000, 800, 2), bands)  # heavy hours, huge library
    assert loyal.loyal_mono is True
    assert collector.loyal_mono is False


def test_unknown_library_when_owned_missing():
    # The 2024 dump ships num_games_owned == 0 for everyone.
    pop = [_rec(pt, pt // 2, 0, nrev) for pt in (30, 600, 6000) for nrev in (1, 5, 40)]
    bands = population_bands(pop)
    assert bands.owned_known is False
    seg = segment_record(_rec(600, 300, 0, 3), bands)
    assert seg.library == "unknown"
    assert seg.loyal_mono is False


def test_bio_differs_between_distinct_personas():
    pop = _population()
    bands = population_bands(pop)
    change = "The game now costs money to keep playing."
    a = situation_bio(_rec(20, 10, 5, 1), change, bands)
    b = situation_bio(_rec(30000, 15000, 3, 500), change, bands)
    assert a != b
    assert change in a and change in b
    # No leakage of the withheld verdict.
    for bio in (a, b):
        assert "positive" not in bio.lower() and "negative" not in bio.lower()

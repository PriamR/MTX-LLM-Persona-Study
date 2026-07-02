"""The answer readout is where the level problem lives, so its mechanics get
tests: which token counts as "recommend" must be configuration (the label-swap
control depends on it), the Δ sign convention must follow that configuration,
and the per-persona Δ dump must carry the raw values the calibration fit needs."""
from __future__ import annotations

import math

import pandas as pd

from llmsonas.cases import PAYDAY2, dump_deltas
from llmsonas.construction.exposure import Exposure
from llmsonas.construction.segment import population_bands
from llmsonas.data.ingest import UserRecord
from llmsonas.harness import recommend_key, survey_permuted, swap_labels
from llmsonas.survey.prompt import _question_block, frequency_messages, grounded_messages
from llmsonas.survey.together_client import _option_logprobs, _p_from_dist, logit_gap

EVENT = 1_600_000_000

# One position's top-logprob dist, as Together returns it (token -> logprob).
DIST = {"B": -0.01, " A": -6.01, "The": -9.0}


def test_p_from_dist_default_matches_original_layout():
    # A=Recommend, B=Not recommend: P(recommend) = sigmoid(lpA - lpB).
    p = _p_from_dist(DIST)
    assert p is not None and abs(p - 1 / (1 + math.exp(6.0))) < 1e-9


def test_label_swap_is_configuration_not_a_fork():
    # Same dist read under the swapped layout (B=Recommend) gives 1 - p.
    p = _p_from_dist(DIST)
    q = _p_from_dist(DIST, recommend="B")
    assert abs((p + q) - 1.0) < 1e-9
    lp = _option_logprobs(DIST)
    assert logit_gap(lp) == -logit_gap(lp, recommend="B")


def test_missing_option_token_uses_floor():
    lp = _option_logprobs({"B": -0.5})
    assert lp == {"B": -0.5}
    assert logit_gap(lp) == -10.0  # floor = min - 10 stands in for the absent A


def test_word_tokens_parse_and_reword_the_prompt():
    dist = {" Yes": -2.0, "No": -0.2, "A": -1.0}
    lp = _option_logprobs(dist, options=("Yes", "No"))
    assert set(lp) == {"Yes", "No"}
    p = _p_from_dist(dist, options=("Yes", "No"), recommend="Yes")
    assert abs(p - 1 / (1 + math.exp(1.8))) < 1e-9
    block = _question_block("Q?", {"Yes": "Yes", "No": "No"})
    assert "single word Yes or No" in block
    # The scored A/B path keeps its original wording.
    assert "single letter A or B" in _question_block("Q?", {"A": "x", "B": "y"})


def test_recommend_key_follows_labels():
    assert recommend_key({"A": "Recommend", "B": "Not recommend"}) == "A"
    assert recommend_key({"A": "Not recommend", "B": "Recommend"}) == "B"
    # No exact "Recommend" value: fall back to the first key (historic layout).
    assert recommend_key({"Yes": "Yes", "No": "No"}) == "Yes"


def test_frequency_messages_are_number_only_and_neutral():
    msgs = frequency_messages("Bio sentence. The change happened.")
    assert msgs[0]["role"] == "system" and "single whole number" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "Out of 100 players" in user and "number from 0 to 100" in user
    low = (msgs[0]["content"] + user).lower()
    for tell in ("angry", "backlash", "betray", "controver"):
        assert tell not in low


def test_dump_deltas_writes_raw_per_persona_values(tmp_path):
    pop = [
        UserRecord(
            steamid=str(i), voted_up=True, playtime_forever=pt, playtime_at_review=pt // 2,
            num_games_owned=50, num_reviews=3, timestamp=EVENT - 86400, review="",
        )
        for i, pt in enumerate((30, 120, 600, 1800, 6000, 20000))
    ]
    bands = population_bands(pop, EVENT)
    expo = {"1": Exposure(3, 0.0, "premium", 1.0, "all")}
    gaps = [{"A": -12.0, "B": -0.5}, {"B": -0.1}]
    idx = [1, 4]
    path = dump_deltas(tmp_path / "d.csv", gaps, [0.2, 0.3], pop, idx, [1.0, 2.0],
                       expo, bands)
    df = pd.read_csv(path, dtype={"steamid": str})
    assert list(df["steamid"]) == ["1", "4"]
    assert df.loc[0, "delta"] == -11.5
    assert df.loc[1, "delta"] == -10.0            # floored absent A token
    assert df.loc[0, "disposition"] == "all"
    assert df.loc[1, "expo_band"] == "none"       # no footprint -> none
    assert list(df["weight"]) == [1.0, 2.0]
    assert set(df.columns) >= {"rank", "p", "lp_recommend", "lp_other", "investment"}


def test_swap_labels_exchanges_meanings_not_keys():
    labels = {"A": "Recommend", "B": "Not recommend"}
    swapped = swap_labels(labels)
    assert swapped == {"A": "Not recommend", "B": "Recommend"}
    assert recommend_key(swapped) == "B"
    assert swap_labels(swapped) == labels  # involution


def test_survey_permuted_averages_both_label_orders():
    labels = {"A": "Recommend", "B": "Not recommend"}
    seen: list[str] = []

    # A backend that reads which layout it was shown: 0.9 under the original
    # ("A) Recommend" in the prompt), 0.1 under the swap — so a correct
    # average is 0.5 and any single-arm readout is not.
    def backend(msgs: list[dict], mdl: str) -> float:
        text = msgs[-1]["content"]
        seen.append(text)
        return 0.9 if "A) Recommend" in text else 0.1

    p_bar, p_orig, p_swap = survey_permuted(
        ["bio one", "bio two"], "model", "Q?", labels, grounded=True, backend=backend)
    assert list(p_orig) == [0.9, 0.9] and list(p_swap) == [0.1, 0.1]
    assert list(p_bar) == [0.5, 0.5]
    assert len(seen) == 4  # every persona surveyed under both orders
    assert sum("A) Not recommend" in t for t in seen) == 2


def test_grounded_messages_unchanged_for_scored_labels():
    # Guard the scored-path template: same system prompt, same block layout.
    msgs = grounded_messages("Bio.", PAYDAY2.question, PAYDAY2.answer_labels)
    assert "Reply with a single letter" in msgs[0]["content"]
    assert "A) Recommend" in msgs[1]["content"]
    assert "B) Not recommend" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Base the answer only on this player's history.")

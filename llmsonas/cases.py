"""Time-split cases that run on the "Steam Reviews 2024" per-app dump.

Each case is a game + a dated decision whose real recommend swing is read from
the dump at run time. The same ladder (M1 -> M2a/M2b -> M3 -> score) runs on all
of them; only the appid, the event date and the neutral change clause differ, so
nothing is tuned per case to a known answer.

The set is chosen to span *directions and mechanisms*, because a method that only
predicts backlash could pass a backlash-only set by always leaning negative:

* PAYDAY2 (2015) — sharp microtransaction backlash, 0.87 -> 0.31, un-reverted.
  Split axis: betrayal scales with investment (playtime/tenure). Primary flagship.
* TOTALWAR3 (2023) — overpriced-DLC backlash, 0.85 -> 0.24, un-reverted. A second,
  less-memorised backlash on a different mechanism (DLC value, not MTX). Split
  axis: value-sensitivity of invested owners.
* NOMANSSKY (2018) — a redemption *up*-swing after the free NEXT update. Tests
  direction generality: personas must react to the *content* of a change, not a
  fixed "change = bad" prior. Famous, so carries a memorisation caveat.
* HELLDIVERS2 (2024) — contrast where this dump post-dates Sony's reversal, so the
  measured swing washes out; kept as a ground-truth-integrity example.

Why these and not CS:GO / The Alters: CS:GO F2P was a *unanimous shock* (every
owner devalued at once) and The Alters had no behavioural move, so there was
nothing heterogeneous for persona attributes or the homophily graph to separate —
the simulation cannot catch a reaction that is uniform or absent. Every scored
case here is a genuine, attribute-predictable *split* (some owners stay positive),
which is the condition under which the method can be validated at all.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from llmsonas.config import ROOT
from llmsonas.construction.exposure import build_footprint, exposures
from llmsonas.construction.profile import m1_profile, situation_bio
from llmsonas.construction.segment import Bands, population_bands, segment_record
from llmsonas.construction.select import cluster_archetypes, stratified_sample
from llmsonas.data.dump import load_hd2, recommend_ratio, to_user_records
from llmsonas.features.build import numeric_features
from llmsonas.graph.build import build_influence_matrix
from llmsonas.graph.fj import friedkin_johnsen
from llmsonas.graph.nullcheck import homophily_nullcheck
from llmsonas.harness import (
    MethodResult,
    recommend_key,
    score,
    survey_permuted,
    swap_labels,
)

OUT_DIR = ROOT / "out"

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"


@dataclass(frozen=True)
class DumpCase:
    """One game + dated decision, scored on the 2024 per-app dump."""

    key: str
    title: str
    appid: int
    event_cutoff: int          # unix seconds, the announcement date
    change: str                # neutral statement of the decision (no valence)
    # One neutral question for every case: the decision itself lives in the
    # persona's situation (``change`` in the bio), not in the question wording.
    # A question that restated the event with valence ("after promising never
    # to...") made the model answer the framing, identically for all personas.
    question: str = "Would this player recommend the game to other players after this change?"
    note: str = ""             # the split axis / why it isn't a unanimous shock
    gt_window_days: int = 7
    n_personas: int = 20
    pool: int = 600
    graph_rounds: int = 3
    knn_k: int = 8
    susceptibility: float = 0.5
    seed: int = 42
    model: str = MODEL
    answer_labels: dict[str, str] = field(
        default_factory=lambda: {"A": "Recommend", "B": "Not recommend"}
    )


PAYDAY2 = DumpCase(
    key="payday2",
    title="Payday 2 microtransactions (2015-10-15)",
    appid=218620,
    event_cutoff=1444867200,   # 2015-10-15 00:00 UTC
    # Mechanics only: the earlier "...that affect gameplay" was the community's
    # contested *characterization* of the update, not a fact — an editorial cue
    # strong enough to flatten any behavioural axis in the bio (guardrail (a):
    # facts, never the valence). What shipped: purchasable drills open safes
    # that drop in-game; the skins inside can carry stat bonuses.
    change=(
        "The developer had said the game would never have microtransactions, and "
        "has now added purchasable drills that open safes containing weapon "
        "skins, some of which carry stat bonuses."
    ),
    note="split axis: betrayal scales with investment (playtime/tenure). Backlash, un-reverted.",
)

TOTALWAR3 = DumpCase(
    key="totalwar3",
    title="Total War: Warhammer III — Shadows of Change DLC (2023-08-08)",
    appid=1142710,
    event_cutoff=1691452800,   # 2023-08-08 00:00 UTC (DLC announcement)
    change=(
        "The developer has released a new paid DLC that adds fewer new units and "
        "characters than earlier DLC did at the same price."
    ),
    note="split axis: value-sensitivity of invested owners. Second backlash, different mechanism, less memorised.",
)

NOMANSSKY = DumpCase(
    key="nomanssky",
    title="No Man's Sky — NEXT free update (2018-07-24)",
    appid=275850,
    event_cutoff=1532390400,   # 2018-07-24 00:00 UTC (NEXT release)
    change=(
        "The developer has released a large free update ('NEXT') that adds "
        "multiplayer and base building and overhauls the game's visuals."
    ),
    note="redemption UP-swing; split axis: whether a lapsed/critical owner returns. Memorisation caveat (famous).",
)

HELLDIVERS2 = DumpCase(
    key="helldivers2",
    title="Helldivers 2 PSN linking (2024-05-03)",
    appid=553850,
    event_cutoff=1714694400,   # 2024-05-03 00:00 UTC
    change=(
        "The game now requires linking a PlayStation Network account to keep "
        "playing on PC."
    ),
    note="CONTRAST ONLY: dump post-dates Sony's reversal (~58% edited), so the measured swing washes out.",
)

TEKKEN8 = DumpCase(
    key="tekken8",
    title="Tekken 8 — Tekken Shop cash shop (2024-02-28)",
    appid=1778820,
    event_cutoff=1709078400,   # 2024-02-28 00:00 UTC (Tekken Shop launch)
    # Facts only: a real-money cosmetics shop added to a full-price game one
    # month after launch, with a paid battle pass following. No promise-break
    # clause — none is documented for Tekken 8 (unlike Payday 2).
    change=(
        "The game, which costs full price, has added an in-game shop that "
        "sells cosmetic items for real money, followed by a paid battle pass."
    ),
    note="second in-scope premium-MTX case. SLOW BURN: the backlash built over "
         "~6 weeks (shop 02-28, battle pass early April), so the window is 45d, "
         "chosen from the swing shape before any model run. Post-cutoff event "
         "(training data ends ~2023-12): no memorisation possible. Screened "
         "2026-07-02: 0.835 -> 0.560 (45d), panel edits 0.261 (n=1153).",
    # The reaction accumulated instead of spiking; a 7-day window would measure
    # the pre-backlash lull (-0.08), not the reaction (-0.28 by day 45).
    gt_window_days=45,
)

ARK = DumpCase(
    key="ark",
    title="ARK: Survival Evolved — Scorched Earth paid expansion (2016-09-01)",
    appid=346110,
    event_cutoff=1472688000,   # 2016-09-01 00:00 UTC (Scorched Earth release)
    # Facts only: a paid expansion released while the base game was still in
    # Early Access. No characterization of whether that was appropriate — the
    # split (backers who feel owed a finished base game vs players happy for
    # more content) must come from the personas.
    change=(
        "The developer has released a paid expansion pack that adds a new "
        "desert map, creatures and items, while the base game is still in "
        "Early Access."
    ),
    note="split axis: early-access backers vs later buyers; investment. Same "
         "paid-content decision class as Payday 2/TW3. Screened 2026-07-02: "
         "0.741 -> 0.309 (7d), bracket [0.249, 0.309].",
)

CSGO = DumpCase(
    key="csgo",
    title="CS:GO paid -> free-to-play (2018-12-06)",
    appid=730,
    event_cutoff=1544054400,   # 2018-12-06 00:00 UTC
    change=(
        "The game, which every current owner paid for, is now free-to-play for "
        "everyone."
    ),
    note="DIAGNOSTIC: a universal shock — every paid owner is devalued at once, so "
         "there is little to split on. Graph expected not to help; kept to test whether "
         "the enriched personas still collapse the way the anchoring finding predicts.",
)

CASES = {c.key: c for c in (PAYDAY2, TEKKEN8, TOTALWAR3, NOMANSSKY, HELLDIVERS2, ARK, CSGO)}


def _offline_backend(messages: list[dict], model: str) -> float:
    """Deterministic stand-in for the model: a stable pseudo-probability derived
    from the prompt text, so distinct personas get distinct answers and the run is
    reproducible. A wiring stub, not a prediction."""
    text = messages[-1]["content"]
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
    return 0.05 + 0.90 * ((h % 10_000) / 10_000)


def _row(r: MethodResult) -> str:
    return f"{r.method:<5} {r.p_hat:>7.3f}   [{r.ci[0]:.3f}, {r.ci[1]:.3f}]   {r.spread:>6.3f}   {r.jsd:>7.4f}"


def _print_persona_mix(records, idx, bands, bios, expo) -> None:
    """Show how the surveyed personas segment, so the spread is visible before the
    answers come back, and print one bio verbatim (Approach §3.4 guardrail e)."""
    from collections import Counter

    segs = [segment_record(records[i], bands) for i in idx]
    inv = Counter(s.investment for s in segs)
    voc = Counter(s.vocalness for s in segs)
    mono = sum(s.loyal_mono for s in segs)
    inv_s = " ".join(f"{k}:{inv[k]}" for k in
                     ("light", "casual", "regular", "dedicated", "hardcore") if inv[k])
    voc_s = " ".join(f"{k}:{voc[k]}" for k in
                     ("quiet", "occasional", "vocal", "prolific") if voc[k])
    exp = Counter(_expo_band(expo, records[i].steamid) for i in idx)
    exp_s = " ".join(f"{k}:{exp[k]}" for k in
                     ("none", "premium", "mixed", "f2p_leaning") if exp[k])
    disp = Counter(_expo_disposition(expo, records[i].steamid) for i in idx)
    disp_s = " ".join(f"{k}:{disp[k]}" for k in
                      ("none", "all", "most", "half", "few", "unknown") if disp[k])
    print(f"[mix]   investment [{inv_s}] | vocalness [{voc_s}] | loyal-single-game: {mono}")
    print(f"[mix]   monetization exposure [{exp_s}] | other-game verdicts [{disp_s}]")
    print(f"[bio]   e.g. {bios[0]}")


def _expo_band(expo, steamid: str) -> str:
    e = expo.get(steamid)
    return e.band if e is not None else "none"


def _expo_disposition(expo, steamid: str) -> str:
    e = expo.get(steamid)
    return e.disposition if e is not None else "none"


def _print_gap_report(
    gaps: list[dict], records, idx, expo,
    options: tuple[str, str] = ("A", "B"), recommend: str = "A",
) -> None:
    """The un-softmaxed recommend−other logit gap per persona: whether the
    features moved the model at all is visible here even when every P saturates
    to the same pole."""
    from llmsonas.survey.together_client import logit_gap

    deltas = [logit_gap(lp, options, recommend) for lp in gaps]
    known = [d for d in deltas if d is not None]
    if not known:
        return
    arr = np.asarray(known)
    print(f"\nraw option-logit gap Δ = logprob(recommend) − logprob(not recommend), M2a personas:")
    print(f"  overall: median {np.median(arr):+.2f} | min {arr.min():+.2f} | "
          f"max {arr.max():+.2f} | std {arr.std():.2f}")
    by_band: dict[str, list[float]] = {}
    by_disp: dict[str, list[float]] = {}
    for d, i in zip(deltas, idx):
        if d is not None:
            by_band.setdefault(_expo_band(expo, records[i].steamid), []).append(d)
            by_disp.setdefault(_expo_disposition(expo, records[i].steamid), []).append(d)
    parts = [f"{band}: {np.mean(v):+.2f} (n={len(v)})"
             for band, v in sorted(by_band.items())]
    print(f"  mean by exposure band: {' | '.join(parts)}")
    parts = [f"{band}: {np.mean(v):+.2f} (n={len(v)})"
             for band, v in sorted(by_disp.items())]
    print(f"  mean by other-game verdicts: {' | '.join(parts)}")


def delta_csv_path(case: DumpCase, tag: str = "") -> Path:
    """Where a run's per-persona Δ dump lands (git-ignored ``out/``)."""
    model_tag = case.model.rsplit("/", 1)[-1]
    suffix = f"_{tag}" if tag else ""
    return OUT_DIR / f"deltas_{case.key}_{model_tag}{suffix}_{time.strftime('%Y-%m-%d')}.csv"


def dump_deltas(
    path: Path,
    gaps: list[dict],
    P,
    records,
    idx,
    weights,
    expo,
    bands: Bands,
    options: tuple[str, str] = ("A", "B"),
    recommend: str = "A",
) -> Path:
    """Write the per-persona raw readout to CSV.

    The printed report only summarises; the cross-case Δ→P calibration fit and
    the answer-format controls need the raw per-persona values, joinable across
    runs on the (rank, steamid) key since the sample is seed-deterministic.
    """
    import pandas as pd

    from llmsonas.survey.together_client import logit_gap

    other = next(o for o in options if o != recommend)
    rows = []
    for rank, (i, lp) in enumerate(zip(idx, gaps), 1):
        r = records[i]
        seg = segment_record(r, bands)
        e = expo.get(r.steamid)
        rows.append({
            "rank": rank,
            "steamid": r.steamid,
            "weight": float(weights[rank - 1]),
            "p": float(P[rank - 1]),
            "delta": logit_gap(lp, options, recommend),
            "lp_recommend": lp.get(recommend),
            "lp_other": lp.get(other),
            "investment": seg.investment,
            "vocalness": seg.vocalness,
            "channel": seg.channel,
            "tenure": seg.tenure,
            "expo_band": e.band if e is not None else "none",
            "disposition": e.disposition if e is not None else "none",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@dataclass
class PersonaPanel:
    """The M2a persona sample for a case, rebuilt deterministically outside
    ``run_dump_smoke`` so control/diagnostic scripts survey the same people the
    scored run does (same pool sample, same stratified draw, same bios)."""

    records: list
    bands: Bands
    expo: dict
    idx: list[int]
    weights: np.ndarray
    bios: list[str]
    pre_ratio: float
    pre_n: int
    gt: float                 # flow GT (window review ratio)
    gt_n: int
    panel_p: float            # bracket's panel end (nan when no in-window edits)
    panel_n: int


def build_m2a_panel(case: DumpCase) -> PersonaPanel:
    """Rebuild the case's M2a persona sample and both bracket ends."""
    cut = case.event_cutoff
    df = load_hd2(case.appid)
    pre_ratio, pre_n = recommend_ratio(df, None, cut)
    gt, gt_n = recommend_ratio(df, cut, cut + case.gt_window_days * 86400)
    panel = df[(df["ts"] < cut) & (df["timestamp_updated"] >= cut)
               & (df["timestamp_updated"] < cut + case.gt_window_days * 86400)]
    panel_p = float(panel["voted_up"].mean()) if len(panel) else float("nan")
    pre = df[(df["ts"] < cut) & (df["language"] == "english")]
    pool = pre.sample(n=min(case.pool, len(pre)), random_state=case.seed)
    records = to_user_records(pool, require_review=False)
    bands = population_bands(records, cut)
    expo = exposures(build_footprint(case.appid, cut))
    idx, weights = stratified_sample(records, case.n_personas, seed=case.seed)
    bios = [situation_bio(records[i], case.change, bands, expo.get(records[i].steamid))
            for i in idx]
    return PersonaPanel(
        records=records, bands=bands, expo=expo, idx=idx, weights=weights,
        bios=bios, pre_ratio=pre_ratio, pre_n=pre_n, gt=gt, gt_n=gt_n,
        panel_p=panel_p, panel_n=len(panel),
    )


def run_dump_smoke(case: DumpCase) -> None:
    """Run the full ladder on ``case`` and print the methods x1-question table.

    Uses the real Together client when TOGETHER_API_KEY is set and
    LLMSONAS_OFFLINE is unset; otherwise a deterministic stub stands in for the
    model so the wiring can be exercised without a network round-trip.
    """
    # Windows terminals default to a legacy code page (e.g. cp932) that can't
    # encode the report's punctuation; force UTF-8 so a run doesn't die mid-print.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    offline = bool(os.getenv("LLMSONAS_OFFLINE")) or not os.getenv("TOGETHER_API_KEY")
    backend = _offline_backend if offline else None
    cut = case.event_cutoff
    Q, LABELS, MODEL_ID = case.question, case.answer_labels, case.model

    df = load_hd2(case.appid)
    pre_ratio, pre_n = recommend_ratio(df, None, cut)
    gt, gt_n = recommend_ratio(df, cut, cut + case.gt_window_days * 86400)

    post = df[(df["ts"] >= cut) & (df["ts"] < cut + case.gt_window_days * 86400)]
    clean = post[~post["edited"]]
    gt_clean = float(clean["voted_up"].mean()) if len(clean) else float("nan")

    print(f"{case.title} | model={'OFFLINE-STUB' if offline else MODEL_ID} | n={case.n_personas}")
    print("  readout: permutation-averaged over answer order — every persona is "
          "surveyed under both label layouts and p̄ = (p_orig + p_swap)/2 is "
          "scored (adopted 2026-07-02; E1 measured a median 2.4-nat bias toward "
          "the later-listed option)")
    if case.note:
        print(f"  case: {case.note}")
    direction = "UP" if gt > pre_ratio else "DOWN"
    print(f"real swing ({direction}): pre-event recommend {pre_ratio:.3f} (n={pre_n})  ->  "
          f"post-{case.gt_window_days}d {gt:.3f} (n={gt_n})  |  ground truth = {gt:.3f}")
    print(f"  note: {100 * post['edited'].mean():.0f}% of that window was edited after "
          f"posting; never-edited post ratio = {gt_clean:.3f} (n={len(clean)})")

    # Estimand note. Steam allows one review per user per app, so every review
    # *created* in the window is a first-time reviewer of this game — the scored
    # GT is the vocal reaction FLOW, not a poll of the panel the personas are
    # built from. The nearest observable panel signal is pre-event reviews
    # *edited* inside the window (self-selected, current stored stance); the two
    # bracket the honest target.
    panel = df[(df["ts"] < cut) & (df["timestamp_updated"] >= cut)
               & (df["timestamp_updated"] < cut + case.gt_window_days * 86400)]
    if len(panel):
        panel_p = float(panel["voted_up"].mean())
        print(f"  estimand: window reviews are all first-time reviewers (flow GT); "
              f"pre-event reviewers who edited in-window: {panel_p:.3f} (n={len(panel)}, "
              f"self-selected) -> target bracket [{min(panel_p, gt):.3f}, {max(panel_p, gt):.3f}]")
    else:
        print("  estimand: window reviews are all first-time reviewers (flow GT); "
              "no in-window edits by pre-event reviewers to bracket against")
    print()

    pre = df[(df["ts"] < cut) & (df["language"] == "english")]
    pool = pre.sample(n=min(case.pool, len(pre)), random_state=case.seed)
    records = to_user_records(pool, require_review=False)
    X = numeric_features(records)
    bands = population_bands(records, cut)
    # Cross-app monetization exposure — the answer axis the 70B run showed the
    # other markers don't carry. First call per case scans the dump (~90s), then
    # it's a cached parquet read.
    expo = exposures(build_footprint(case.appid, cut))
    print(f"[data]  persona pool {len(records)} pre-event reviewers | feature matrix {X.shape} "
          f"| cross-app footprint: {sum(1 for r in records if r.steamid in expo)}/{len(records)}")

    results: list[MethodResult] = []

    m1, _, _ = survey_permuted([m1_profile(i) for i in range(case.n_personas)],
                               MODEL_ID, Q, LABELS, grounded=False, backend=backend)
    results.append(score("M1", m1, None, gt, seed=case.seed))

    a_idx, a_w = stratified_sample(records, case.n_personas, seed=case.seed)
    bios_a = [situation_bio(records[i], case.change, bands, expo.get(records[i].steamid))
              for i in a_idx]
    _print_persona_mix(records, a_idx, bands, bios_a, expo)
    # Collect the raw option logprobs for both M2a arms so the report can show
    # the logit gaps a saturated P hides (only the real client can supply
    # them). Which letter means "recommend" follows each arm's labels, so the
    # swap arm is a config change, not a fork.
    labels_swap = swap_labels(LABELS)
    gaps_a: list[dict] = []
    gaps_a_swap: list[dict] = []
    opts, rec = tuple(LABELS), recommend_key(LABELS)
    rec_swap = recommend_key(labels_swap)
    if backend is None:
        from llmsonas.survey.together_client import answer_probability

        def m2a_backend(msgs: list[dict], mdl: str) -> float | None:
            return answer_probability(msgs, mdl, options=opts, recommend=rec, detail=gaps_a)

        def m2a_backend_swap(msgs: list[dict], mdl: str) -> float | None:
            return answer_probability(msgs, mdl, options=opts, recommend=rec_swap,
                                      detail=gaps_a_swap)
    else:
        m2a_backend = m2a_backend_swap = backend
    Pa, Pa_orig, Pa_swap = survey_permuted(bios_a, MODEL_ID, Q, LABELS, grounded=True,
                                           backend=m2a_backend,
                                           backend_swap=m2a_backend_swap)
    res_a = score("M2a", Pa, a_w, gt, seed=case.seed)
    results.append(res_a)

    b_idx, b_w = cluster_archetypes(X, records, case.n_personas, seed=case.seed)
    bios_b = [situation_bio(records[i], case.change, bands, expo.get(records[i].steamid))
              for i in b_idx]
    Pb, _, _ = survey_permuted(bios_b, MODEL_ID, Q, LABELS, grounded=True, backend=backend)
    results.append(score("M2b", Pb, b_w, gt, seed=case.seed))

    # Hub weight = how far each voice actually carried: helpful votes on the
    # review, not just how many reviews the person wrote. Fall back to review
    # count where a dump ships no vote counts (all-zero votes_up).
    votes = np.array([records[i].votes_up for i in a_idx], dtype=float)
    if votes.sum() <= 0:
        votes = np.array([records[i].num_reviews for i in a_idx], dtype=float)
    influence = np.log1p(votes)
    W = build_influence_matrix(X[a_idx], case.knn_k, influence=influence, seed=case.seed)
    x_final, _ = friedkin_johnsen(W, Pa, case.susceptibility, case.graph_rounds)
    results.append(score("M3", x_final, a_w, gt, seed=case.seed))
    null = homophily_nullcheck(W, Pa, seed=case.seed)

    print("\nmethod   p_hat     95% CI            spread    JSD(vs GT)")
    print("-" * 58)
    for r in results:
        print(_row(r))

    print("\nDid grounded personas track the real post-event stance?")
    print(f"  pre-event prior recommend = {pre_ratio:.3f}")
    print(f"  real post-event (GT)      = {gt:.3f}")
    print(f"  predicted (M2a)           = {res_a.p_hat:.3f}")
    closer = abs(res_a.p_hat - gt) < abs(res_a.p_hat - pre_ratio)
    print(f"  -> M2a is {'closer to the outcome than to the prior' if closer else 'anchored on the prior'}")

    # Whether M3 can do anything is a question of dispersion, not of which pole
    # the personas land on: a near-uniform response leaves the graph nothing to
    # carry whether it echoes the prior (the CS:GO shock) or collapses to one
    # verdict far from it (the situation-bio failure on an 8B model).
    uniform = res_a.spread < 0.12
    if uniform:
        where = ("echoing the prior" if abs(res_a.p_hat - pre_ratio) < 0.08
                 else "collapsed to one verdict, off the prior")
        read = f"near-uniform personas {where} (graph has nothing to carry)"
    else:
        read = "a heterogeneous split (personas disagree, the graph can act)"
    print(f"  simulation reads as: {read} [M2a spread={res_a.spread:.3f}]")

    if gaps_a:
        _print_gap_report(gaps_a, records, a_idx, expo, opts, rec)
        path = dump_deltas(delta_csv_path(case, tag="m2a-orig"), gaps_a, Pa_orig,
                           records, a_idx, a_w, expo, bands, opts, rec)
        print(f"  per-persona Δ dump (orig arm) -> {path}")
    if gaps_a_swap:
        from llmsonas.survey.together_client import logit_gap

        d_o = np.array([logit_gap(lp, opts, rec) for lp in gaps_a], dtype=float)
        d_s = np.array([logit_gap(lp, opts, rec_swap) for lp in gaps_a_swap], dtype=float)
        if len(d_o) == len(d_s) and len(d_o):
            d_bar, beta = (d_o + d_s) / 2.0, (d_o - d_s) / 2.0
            print(f"  debiased Δ̄ (mean of both orders): median {np.nanmedian(d_bar):+.2f} "
                  f"[{np.nanmin(d_bar):+.2f}, {np.nanmax(d_bar):+.2f}] | "
                  f"order bias β median {np.nanmedian(beta):+.2f}")
        path = dump_deltas(delta_csv_path(case, tag="m2a-swap"), gaps_a_swap, Pa_swap,
                           records, a_idx, a_w, expo, bands, opts, rec_swap)
        print(f"  per-persona Δ dump (swap arm) -> {path}")

    print("\nhomophily null-shuffle check (M3 graph):")
    print(f"  Moran's I = {null.observed:+.4f} | null mean = {null.null_mean:+.4f} | "
          f"z = {null.z:+.2f} | p = {null.p_value:.4f} -> "
          f"{'HOLDS' if null.passes else 'NOT SUPPORTED'}")

    if offline:
        print("\n[offline stub run — answers are deterministic placeholders, not a "
              "prediction. Run with TOGETHER_API_KEY set for real logprobs.]")

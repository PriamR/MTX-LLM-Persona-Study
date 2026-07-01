"""Time-split cases that run on the "Steam Reviews 2024" per-app dump.

Each case is a game + a dated decision whose real recommend swing is read from
the dump at run time. The same ladder (M1 -> M2a/M2b -> M3 -> score) runs on all
of them; only the appid, the event date and the neutral change clause differ, so
nothing is tuned per case to a known answer.

Two are wired:

* PAYDAY2 — the microtransaction betrayal of Oct 2015. Overkill promised "no
  microtransactions whatsoever", then added paid safes/drills; recommend fell
  0.87 -> 0.31 (7d), and the *never-edited* reviews are even more negative
  (0.16), so the swing is real, not a reversion artefact. Divisive along a
  behavioural axis (betrayal scales with investment), contagious, not
  region-bound — the plan's primary graph flagship.
* HELLDIVERS2 — the mandatory-PSN announcement of May 2024. Kept as a documented
  contrast: Sony reversed the policy days later, ~58% of the window was edited,
  and the measured post ratio comes out *positive* (0.84), so this dump cannot
  reproduce the live histogram's backlash. Useful as the "behavioural GT needs
  behaviour to move, and stay moved" illustration.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np

from llmsonas.construction.profile import m1_profile, situation_bio
from llmsonas.construction.select import cluster_archetypes, stratified_sample
from llmsonas.data.dump import load_hd2, recommend_ratio, to_user_records
from llmsonas.features.build import numeric_features
from llmsonas.graph.build import build_influence_matrix
from llmsonas.graph.fj import friedkin_johnsen
from llmsonas.graph.nullcheck import homophily_nullcheck
from llmsonas.harness import MethodResult, score, survey

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"


@dataclass(frozen=True)
class DumpCase:
    """One game + dated decision, scored on the 2024 per-app dump."""

    key: str
    title: str
    appid: int
    event_cutoff: int          # unix seconds, the announcement date
    change: str                # neutral statement of the decision (no valence)
    question: str
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
    change=(
        "The developer had said the game would never have microtransactions, and "
        "has now added paid weapon safes and drills that affect gameplay."
    ),
    question=(
        "Payday 2's developer has added paid microtransactions after promising the "
        "game would never have them. Would this player recommend the game after "
        "this change?"
    ),
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
    question=(
        "Helldivers 2 now requires linking a PlayStation Network account to keep "
        "playing on PC. Would this player recommend the game after this change?"
    ),
)

CASES = {c.key: c for c in (PAYDAY2, HELLDIVERS2)}


def _offline_backend(messages: list[dict], model: str) -> float:
    """Deterministic stand-in for the model: a stable pseudo-probability derived
    from the prompt text, so distinct personas get distinct answers and the run is
    reproducible. A wiring stub, not a prediction."""
    text = messages[-1]["content"]
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
    return 0.05 + 0.90 * ((h % 10_000) / 10_000)


def _row(r: MethodResult) -> str:
    return f"{r.method:<5} {r.p_hat:>7.3f}   [{r.ci[0]:.3f}, {r.ci[1]:.3f}]   {r.spread:>6.3f}   {r.jsd:>7.4f}"


def run_dump_smoke(case: DumpCase) -> None:
    """Run the full ladder on ``case`` and print the methods x1-question table.

    Uses the real Together client when TOGETHER_API_KEY is set and
    LLMSONAS_OFFLINE is unset; otherwise a deterministic stub stands in for the
    model so the wiring can be exercised without a network round-trip.
    """
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
    print(f"real swing: pre-event recommend {pre_ratio:.3f} (n={pre_n})  ->  "
          f"post-{case.gt_window_days}d {gt:.3f} (n={gt_n})  |  ground truth = {gt:.3f}")
    print(f"  note: {100 * post['edited'].mean():.0f}% of that window was edited after "
          f"posting; never-edited post ratio = {gt_clean:.3f} (n={len(clean)})\n")

    pre = df[(df["ts"] < cut) & (df["language"] == "english")]
    pool = pre.sample(n=min(case.pool, len(pre)), random_state=case.seed)
    records = to_user_records(pool, require_review=False)
    X = numeric_features(records)
    print(f"[data]  persona pool {len(records)} pre-event reviewers | feature matrix {X.shape}")

    results: list[MethodResult] = []

    m1 = survey([m1_profile(i) for i in range(case.n_personas)], MODEL_ID, Q, LABELS,
                grounded=False, backend=backend)
    results.append(score("M1", m1, None, gt, seed=case.seed))

    a_idx, a_w = stratified_sample(records, case.n_personas, seed=case.seed)
    bios_a = [situation_bio(records[i], case.change) for i in a_idx]
    Pa = survey(bios_a, MODEL_ID, Q, LABELS, grounded=True, backend=backend)
    res_a = score("M2a", Pa, a_w, gt, seed=case.seed)
    results.append(res_a)

    b_idx, b_w = cluster_archetypes(X, records, case.n_personas, seed=case.seed)
    bios_b = [situation_bio(records[i], case.change) for i in b_idx]
    Pb = survey(bios_b, MODEL_ID, Q, LABELS, grounded=True, backend=backend)
    results.append(score("M2b", Pb, b_w, gt, seed=case.seed))

    influence = np.log1p([records[i].num_reviews for i in a_idx])
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

    print("\nhomophily null-shuffle check (M3 graph):")
    print(f"  Moran's I = {null.observed:+.4f} | null mean = {null.null_mean:+.4f} | "
          f"z = {null.z:+.2f} | p = {null.p_value:.4f} -> "
          f"{'HOLDS' if null.passes else 'NOT SUPPORTED'}")

    if offline:
        print("\n[offline stub run — answers are deterministic placeholders, not a "
              "prediction. Run with TOGETHER_API_KEY set for real logprobs.]")

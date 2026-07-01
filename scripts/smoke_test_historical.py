"""CS:GO historical smoke test — the held-out time-split (Q1a).

Personas are built from CS:GO owners who reviewed *before* the Dec 2018 F2P
conversion, then asked whether they'd still recommend the game after it. Their
predicted distribution is scored against the *real* post-event recommend ratio
read straight from the dump — a genuine prediction, not a fit.

The question this answers: does grounding personas in owners' (mostly positive)
pre-F2P history let them predict the negative swing, or do they just echo that
prior? Small scale, integrity over accuracy.
"""
from __future__ import annotations

import numpy as np

from llmsonas.config import SMOKE
from llmsonas.construction.profile import m1_profile, third_person_bio
from llmsonas.construction.select import cluster_archetypes, stratified_sample
from llmsonas.data.dump import extract_game, load_game, recommend_ratio, to_user_records
from llmsonas.features.build import build_features
from llmsonas.graph.build import build_influence_matrix
from llmsonas.graph.fj import friedkin_johnsen
from llmsonas.graph.nullcheck import homophily_nullcheck
from llmsonas.harness import MethodResult, score, survey

CUT = SMOKE.event_cutoff          # 2018-12-06
GT_WINDOW_DAYS = 7                # tight window = the owner backlash, pre-dilution
POOL = 600                        # pre-event rows the personas are drawn from
Q, LABELS, MODEL = SMOKE.question, SMOKE.answer_labels, SMOKE.model


def _englishish(s: str) -> bool:
    return 15 <= len(s) <= 400 and sum(c < 128 for c in s.encode("utf-8", "ignore")) / max(len(s), 1) > 0.9


def _row(r: MethodResult) -> str:
    return f"{r.method:<5} {r.p_hat:>7.3f}   [{r.ci[0]:.3f}, {r.ci[1]:.3f}]   {r.spread:>6.3f}   {r.jsd:>7.4f}"


def main() -> None:
    extract_game(SMOKE.appid)               # cached after first scan
    df = load_game(SMOKE.appid)

    pre_ratio, pre_n = recommend_ratio(df, None, CUT)
    gt, gt_n = recommend_ratio(df, CUT, CUT + GT_WINDOW_DAYS * 86400)
    print(f"CS:GO historical time-split | model={MODEL} | n={SMOKE.n_personas}")
    print(f"real swing: pre-event recommend {pre_ratio:.3f} (n={pre_n})  ->  "
          f"post-{GT_WINDOW_DAYS}d {gt:.3f} (n={gt_n})  |  ground truth = {gt:.3f}\n")

    # Persona pool: pre-event owners, readable English-ish reviews.
    pre = df[df["ts"] < CUT].dropna(subset=["review"])
    pre = pre[pre["review"].map(_englishish)]
    pool = pre.sample(n=min(POOL, len(pre)), random_state=SMOKE.seed)
    records = to_user_records(pool)
    X = build_features(records, seed=SMOKE.seed)
    print(f"[data]  persona pool {len(records)} pre-event owners | feature matrix {X.shape}")

    results: list[MethodResult] = []

    m1 = survey([m1_profile(i) for i in range(SMOKE.n_personas)], MODEL, Q, LABELS, grounded=False)
    results.append(score("M1", m1, None, gt, seed=SMOKE.seed))

    a_idx, a_w = stratified_sample(records, SMOKE.n_personas, seed=SMOKE.seed)
    Pa = survey([third_person_bio(records[i]) for i in a_idx], MODEL, Q, LABELS, grounded=True)
    res_a = score("M2a", Pa, a_w, gt, seed=SMOKE.seed)
    results.append(res_a)

    b_idx, b_w = cluster_archetypes(X, records, SMOKE.n_personas, seed=SMOKE.seed)
    Pb = survey([third_person_bio(records[i]) for i in b_idx], MODEL, Q, LABELS, grounded=True)
    results.append(score("M2b", Pb, b_w, gt, seed=SMOKE.seed))

    influence = np.log1p([records[i].num_reviews for i in a_idx])
    W = build_influence_matrix(X[a_idx], SMOKE.knn_k, influence=influence, seed=SMOKE.seed)
    x_final, _ = friedkin_johnsen(W, Pa, SMOKE.susceptibility, SMOKE.graph_rounds)
    results.append(score("M3", x_final, a_w, gt, seed=SMOKE.seed))
    null = homophily_nullcheck(W, Pa, seed=SMOKE.seed)

    print("\nmethod   p_hat     95% CI            spread    JSD(vs GT)")
    print("-" * 58)
    for r in results:
        print(_row(r))

    print("\nDid personas predict the backlash?")
    print(f"  owners' prior recommend (pre-event) = {pre_ratio:.3f}")
    print(f"  real post-event backlash (GT)        = {gt:.3f}")
    print(f"  predicted (M2a)                      = {res_a.p_hat:.3f}")
    verdict = "CAPTURED" if res_a.p_hat < (pre_ratio + gt) / 2 else "MISSED (anchored on prior)"
    print(f"  -> personas {verdict} the negative swing")

    print("\nhomophily null-shuffle check (M3 graph):")
    print(f"  Moran's I = {null.observed:+.4f} | null mean = {null.null_mean:+.4f} | "
          f"z = {null.z:+.2f} | p = {null.p_value:.4f} -> "
          f"{'HOLDS' if null.passes else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()

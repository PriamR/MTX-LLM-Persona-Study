"""CS:GO smoke test — end-to-end pipeline integrity check (first pass).

Runs the full ladder (M1 -> M2a -> M2b -> M3 -> score) on a minimal slice:
~20 personas, one question (Q1a), 3 graph rounds, on a serverless Llama via
Together. The goal is to prove every stage connects, not to produce valid
numbers — it scores against a stub ground truth and makes no accuracy claim.
"""
from __future__ import annotations

import numpy as np

from llmsonas.config import SMOKE
from llmsonas.construction.profile import m1_profile, third_person_bio
from llmsonas.construction.select import cluster_archetypes, stratified_sample
from llmsonas.data.ingest import fetch_user_records
from llmsonas.features.build import build_features
from llmsonas.graph.build import build_influence_matrix
from llmsonas.graph.fj import friedkin_johnsen
from llmsonas.graph.nullcheck import homophily_nullcheck
from llmsonas.harness import MethodResult, score, survey

Q, LABELS, MODEL, GT = SMOKE.question, SMOKE.answer_labels, SMOKE.model, SMOKE.gt_recommend_stub


def _row(r: MethodResult) -> str:
    return (
        f"{r.method:<5} {r.p_hat:>7.3f}   [{r.ci[0]:.3f}, {r.ci[1]:.3f}]"
        f"   {r.spread:>6.3f}   {r.jsd:>7.4f}"
    )


def main() -> None:
    print(f"CS:GO smoke test | model={MODEL} | n={SMOKE.n_personas} | stub GT recommend={GT}\n")

    # 1-2. Ingest + features -----------------------------------------------
    records = fetch_user_records(SMOKE.appid, target=SMOKE.n_users)
    X = build_features(records, seed=SMOKE.seed)
    print(f"[data]  {len(records)} user records | feature matrix {X.shape}")

    results: list[MethodResult] = []

    # M1 — naive prompting (control, ungrounded) ---------------------------
    m1_items = [m1_profile(i) for i in range(SMOKE.n_personas)]
    P1 = survey(m1_items, MODEL, Q, LABELS, grounded=False)
    results.append(score("M1", P1, None, GT, seed=SMOKE.seed))

    # M2a — stratified real users ------------------------------------------
    a_idx, a_w = stratified_sample(records, SMOKE.n_personas, seed=SMOKE.seed)
    a_bios = [third_person_bio(records[i]) for i in a_idx]
    Pa = survey(a_bios, MODEL, Q, LABELS, grounded=True)
    res_a = score("M2a", Pa, a_w, GT, seed=SMOKE.seed)
    results.append(res_a)

    # M2b — cluster archetypes (weighted) ----------------------------------
    b_idx, b_w = cluster_archetypes(X, records, SMOKE.n_personas, seed=SMOKE.seed)
    b_bios = [third_person_bio(records[i]) for i in b_idx]
    Pb = survey(b_bios, MODEL, Q, LABELS, grounded=True)
    results.append(score("M2b", Pb, b_w, GT, seed=SMOKE.seed))

    # M3 — grounded + social graph (built on M2a) --------------------------
    influence = np.log1p([records[i].num_reviews for i in a_idx])
    W = build_influence_matrix(X[a_idx], SMOKE.knn_k, influence=influence, seed=SMOKE.seed)
    x_final, traj = friedkin_johnsen(W, Pa, SMOKE.susceptibility, SMOKE.graph_rounds)
    results.append(score("M3", x_final, a_w, GT, seed=SMOKE.seed))

    null = homophily_nullcheck(W, Pa, seed=SMOKE.seed)
    shift = np.abs(x_final - Pa)

    # ---- report ----------------------------------------------------------
    print("\nmethod   p_hat     95% CI            spread    JSD(vs GT)")
    print("-" * 58)
    for r in results:
        print(_row(r))

    print("\nM3 attribution (graph effect):")
    print(f"  pre-graph  (M2a) p_hat = {res_a.p_hat:.3f}")
    print(f"  post-graph (M3)  p_hat = {results[-1].p_hat:.3f}")
    print(f"  FJ ran {SMOKE.graph_rounds} rounds | mean stance shift={shift.mean():.4f} max={shift.max():.4f}")

    print("\nhomophily null-shuffle check (M3 graph):")
    print(f"  Moran's I = {null.observed:+.4f} | null mean = {null.null_mean:+.4f} "
          f"(std {null.null_std:.4f}) | z = {null.z:+.2f} | p = {null.p_value:.4f}")
    print(f"  -> homophily config {'HOLDS' if null.passes else 'NOT SUPPORTED'} "
          f"(connected personas {'are' if null.passes else 'are not'} more opinion-aligned than chance)")


if __name__ == "__main__":
    main()

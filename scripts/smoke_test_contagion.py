"""CS:GO contagion smoke test — does propagating *emergent* opinions move M3?

Same held-out time-split as the historical test, pitting three things against the
real post-event ground truth:
  M2a      grounded personas, answered in isolation;
  M3-fj    Friedkin-Johnsen numeric averaging of those stances;
  M3-emg   LLM-in-the-loop, each persona re-answering while seeing how its peers
           currently lean — no injected sentiment (the non-circular, scorable path).

Nothing tells the personas the group is unhappy; the only shared input is the
neutral decision already in the question. So this tests whether the backlash
emerges and propagates on its own, not whether the graph amplifies a conclusion we
handed it.
"""
from __future__ import annotations

import numpy as np

from llmsonas.config import SMOKE
from llmsonas.construction.profile import third_person_bio
from llmsonas.construction.select import stratified_sample
from llmsonas.data.dump import extract_game, load_game, recommend_ratio, to_user_records
from llmsonas.features.build import build_features
from llmsonas.graph.build import build_influence_matrix
from llmsonas.graph.contagion import run_contagion
from llmsonas.graph.fj import friedkin_johnsen
from llmsonas.harness import score, survey

CUT = SMOKE.event_cutoff
GT_WINDOW_DAYS = 7
POOL = 600
Q, LABELS, MODEL = SMOKE.question, SMOKE.answer_labels, SMOKE.model


def _englishish(s: str) -> bool:
    return 15 <= len(s) <= 400 and sum(c < 128 for c in s.encode("utf-8", "ignore")) / max(len(s), 1) > 0.9


def main() -> None:
    extract_game(SMOKE.appid)
    df = load_game(SMOKE.appid)
    pre_ratio, _ = recommend_ratio(df, None, CUT)
    gt, gt_n = recommend_ratio(df, CUT, CUT + GT_WINDOW_DAYS * 86400)
    print(f"CS:GO contagion test | model={MODEL} | n={SMOKE.n_personas}")
    print(f"real: pre-event {pre_ratio:.3f} -> post-{GT_WINDOW_DAYS}d {gt:.3f} (GT, n={gt_n})\n")

    pre = df[df["ts"] < CUT].dropna(subset=["review"])
    pre = pre[pre["review"].map(_englishish)]
    records = to_user_records(pre.sample(n=min(POOL, len(pre)), random_state=SMOKE.seed))
    X = build_features(records, seed=SMOKE.seed)

    a_idx, a_w = stratified_sample(records, SMOKE.n_personas, seed=SMOKE.seed)
    bios = [third_person_bio(records[i]) for i in a_idx]
    Pa = survey(bios, MODEL, Q, LABELS, grounded=True)

    influence = np.log1p([records[i].num_reviews for i in a_idx])
    W = build_influence_matrix(X[a_idx], SMOKE.knn_k, influence=influence, seed=SMOKE.seed)

    x_fj, _ = friedkin_johnsen(W, Pa, SMOKE.susceptibility, SMOKE.graph_rounds)
    x_ct, traj = run_contagion(
        bios, W, Pa, model=MODEL, question=Q, labels=LABELS,
        rounds=SMOKE.graph_rounds,  # shock=None -> emergent, non-circular
    )

    results = [
        score("M2a", Pa, a_w, gt, seed=SMOKE.seed),
        score("M3-fj", x_fj, a_w, gt, seed=SMOKE.seed),
        score("M3-emg", x_ct, a_w, gt, seed=SMOKE.seed),
    ]

    print("method    p_hat     95% CI            spread    JSD(vs GT)")
    print("-" * 59)
    for r in results:
        print(f"{r.method:<7} {r.p_hat:>7.3f}   [{r.ci[0]:.3f}, {r.ci[1]:.3f}]   {r.spread:>6.3f}   {r.jsd:>7.4f}")

    print("\ncontagion trajectory (mean p_hat by round):")
    print("  " + " -> ".join(f"{m:.3f}" for m in traj.mean(axis=1)))

    moved = Pa.mean() - x_ct.mean()
    print(f"\nemergent contagion (no injected sentiment): "
          f"M2a={Pa.mean():.3f} -> M3-emg={x_ct.mean():.3f} (moved {moved:+.3f}; GT={gt:.3f})")
    print("  -> " + ("emergent peer influence shifts the aggregate"
                     if abs(moved) > 0.03 else
                     "aggregate ~unchanged: the swing does not emerge on its own (honest, non-circular result)"))


if __name__ == "__main__":
    main()

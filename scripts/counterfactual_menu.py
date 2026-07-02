"""Counterfactual decision menu — LABELLED EXTRAPOLATION, never the scored path.

The stakeholder brief is not "what happened" but "which of my options hurts
least". This script surveys the SAME frozen persona panel under variants of
the decision the stakeholder controlled, holding everything else fixed: same
people, same bios, same frozen template, same permutation-averaged readout —
only the factual ``change`` clause differs, and every variant clause states
mechanics with no valence, exactly like the scored one.

Honesty rules, printed with every run:
* Ground truth exists ONLY for the shipped variant — it anchors the method;
  the other rows are method-extrapolation and are labelled as such.
* Variant clauses are frozen here in code (not tuned per run), and differences
  BETWEEN rows are the deliverable (which lever moves the reaction), not the
  absolute levels — E4 showed the level carries the model's case-specific
  normative reading.

usage: counterfactual_menu.py [case_key] [model_short] [n] [pool]
       (defaults: payday2 70b 100 3000 — matches the scored runs)
"""
from __future__ import annotations

import dataclasses
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from llmsonas.cases import CASES, _expo_disposition, build_m2a_panel
from llmsonas.construction.profile import situation_bio
from llmsonas.harness import recommend_key, survey_permuted, swap_labels
from llmsonas.scoring.metrics import aggregate, js_divergence
from llmsonas.survey.together_client import answer_probability, logit_gap

MODELS = {
    "70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "8b": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
}

# The decision menu per case: variant name -> factual change clause. "shipped"
# must be the case's own clause verbatim (the validation anchor). Variants
# isolate one lever each; all clauses are mechanics-only, no valence.
MENUS: dict[str, dict[str, str]] = {
    "payday2": {
        # shipped is filled from the case definition at run time
        "cosmetic_only": (
            "The developer had said the game would never have microtransactions, "
            "and has now added purchasable drills that open safes containing "
            "weapon skins that are purely cosmetic and carry no stat bonuses."
        ),
        "earnable_drills": (
            "The developer had said the game would never have microtransactions, "
            "and has now added drills that are earned through gameplay and open "
            "safes containing weapon skins, some of which carry stat bonuses."
        ),
        "no_promise": (
            "The developer has added purchasable drills that open safes "
            "containing weapon skins, some of which carry stat bonuses."
        ),
    },
}


def run_variant(panel, case, model: str, change: str):
    bios = [situation_bio(panel.records[i], change, panel.bands,
                          panel.expo.get(panel.records[i].steamid))
            for i in panel.idx]
    labels = case.answer_labels
    labels_s = swap_labels(labels)
    opts = tuple(labels)
    rec, rec_s = recommend_key(labels), recommend_key(labels_s)
    gaps_o: list[dict] = []
    gaps_s: list[dict] = []

    def b_orig(msgs: list[dict], mdl: str) -> float | None:
        return answer_probability(msgs, mdl, options=opts, recommend=rec, detail=gaps_o)

    def b_swap(msgs: list[dict], mdl: str) -> float | None:
        return answer_probability(msgs, mdl, options=opts, recommend=rec_s, detail=gaps_s)

    P, _, _ = survey_permuted(bios, model, case.question, labels, grounded=True,
                              backend=b_orig, backend_swap=b_swap)
    d_o = np.array([logit_gap(lp, opts, rec) for lp in gaps_o], dtype=float)
    d_s = np.array([logit_gap(lp, opts, rec_s) for lp in gaps_s], dtype=float)
    d_bar = (d_o + d_s) / 2.0 if len(d_o) == len(d_s) and len(d_o) else np.array([])
    return P, d_bar


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "payday2"
    model = MODELS[sys.argv[2] if len(sys.argv) > 2 else "70b"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    pool = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    case = dataclasses.replace(CASES[key], model=model, n_personas=n, pool=pool)
    if key not in MENUS:
        raise SystemExit(f"no decision menu defined for {key} — add one to MENUS")
    menu = {"shipped": case.change, **MENUS[key]}

    print(f"COUNTERFACTUAL DECISION MENU (labelled extrapolation, NOT scored) | "
          f"{case.title} | {model} | n={n} pool={pool} seed={case.seed} | "
          f"run at {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print("GT exists only for the 'shipped' row; other rows are method-"
          "extrapolation over the same validated panel. Read the DIFFERENCES "
          "between rows (which lever moves the reaction), not the levels.\n")
    panel = build_m2a_panel(case)
    print(f"bracket (shipped only): panel {panel.panel_p:.3f} (n={panel.panel_n}) | "
          f"flow {panel.gt:.3f} (n={panel.gt_n}) | prior {panel.pre_ratio:.3f}\n")

    rows = []
    for name, change in menu.items():
        P, d_bar = run_variant(panel, case, model, change)
        p_hat = aggregate(P, panel.weights)
        by: dict[str, list[float]] = {}
        for pos, i in enumerate(panel.idx):
            by.setdefault(_expo_disposition(panel.expo, panel.records[i].steamid),
                          []).append(float(P[pos]))
        disp = " ".join(f"{b}:{np.mean(v):.2f}" for b, v in sorted(by.items()))
        rows.append((name, p_hat, float(P.std()),
                     float(np.nanmedian(d_bar)) if len(d_bar) else float("nan"), disp))
        print(f"[{name:<16}] p_hat {p_hat:.3f} | spread {P.std():.3f} | "
              f"Δ̄ median {rows[-1][3]:+.2f}")
        print(f"                   change: {change}")
        print(f"                   p̄ by other-game verdicts: {disp}")

    shipped = rows[0][1]
    print(f"\nshipped-row anchor: p_hat {shipped:.3f} vs bracket "
          f"[{panel.panel_p:.3f}, {panel.gt:.3f}] "
          f"(JSD vs flow {js_divergence(shipped, panel.gt):.4f}) — the credibility "
          f"of the other rows rides on this one.")
    print("deltas vs shipped: "
          + " | ".join(f"{name}: {p - shipped:+.3f}" for name, p, *_ in rows[1:]))


if __name__ == "__main__":
    main()

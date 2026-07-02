"""E1 — answer-format controls for the level problem (label bias).

The 70B reads every grounded persona at P<0.004 while ordering them correctly
in logit space, and the reasoning probe caught a rationale arguing tolerance
that still emitted the not-recommend letter. This run isolates how much of the
level lives in the answer-token mapping rather than the judgment:

* orig  — the scored layout (A=Recommend / B=Not recommend)
* swap  — labels swapped (A=Not recommend / B=Recommend), same template
* yesno — word tokens instead of letters (Yes/No), same template

The permutation-averaged readout p̄ = (p_orig + p_swap) / 2 is the standard
order-debias control; per-persona Δ from both letter arms also separates an
additive letter bias (half the arm difference) from the debiased judgment
(half the arm sum). Everything is mechanical and case-blind: same bios, same
frozen template, only the label layout differs.

usage: e1_label_controls.py [case_key] [model_short] [n] [pool]
       (defaults: payday2 70b 100 3000 — matches the scored 70B runs)
"""
from __future__ import annotations

import dataclasses
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from llmsonas.cases import CASES, build_m2a_panel, delta_csv_path, dump_deltas
from llmsonas.harness import recommend_key, survey
from llmsonas.scoring.metrics import aggregate, js_divergence
from llmsonas.survey.together_client import answer_probability, logit_gap

MODELS = {
    "70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "8b": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
}

ARMS = {
    "orig": {"A": "Recommend", "B": "Not recommend"},
    "swap": {"A": "Not recommend", "B": "Recommend"},
    "yesno": {"Yes": "Would recommend", "No": "Would not recommend"},
}


def run_arm(panel, model: str, question: str, labels: dict[str, str]):
    gaps: list[dict] = []
    opts, rec = tuple(labels), recommend_key(labels)

    def backend(msgs: list[dict], mdl: str) -> float | None:
        return answer_probability(msgs, mdl, options=opts, recommend=rec, detail=gaps)

    P = survey(panel.bios, model, question, labels, grounded=True, backend=backend)
    raw = [logit_gap(lp, opts, rec) for lp in gaps]
    deltas = np.array([np.nan if d is None else d for d in raw], dtype=float)
    return P, gaps, deltas, opts, rec


def band_means(deltas, panel, key) -> str:
    from llmsonas.cases import _expo_band, _expo_disposition

    by: dict[str, list[float]] = {}
    fn = _expo_band if key == "band" else _expo_disposition
    for d, i in zip(deltas, panel.idx):
        by.setdefault(fn(panel.expo, panel.records[i].steamid), []).append(d)
    return " | ".join(f"{b}: {np.mean(v):+.2f} (n={len(v)})" for b, v in sorted(by.items()))


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "payday2"
    model = MODELS[sys.argv[2] if len(sys.argv) > 2 else "70b"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    pool = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    case = dataclasses.replace(CASES[key], model=model, n_personas=n, pool=pool)

    print(f"E1 ANSWER-FORMAT CONTROLS | {case.title} | {model} | n={n} pool={pool} "
          f"seed={case.seed} | run at {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    panel = build_m2a_panel(case)
    ends = [("panel", panel.panel_p), ("flow", panel.gt)]
    print(f"bracket: panel {panel.panel_p:.3f} (n={panel.panel_n}) | "
          f"flow {panel.gt:.3f} (n={panel.gt_n}) | prior {panel.pre_ratio:.3f}\n")

    results: dict[str, tuple] = {}
    for arm, labels in ARMS.items():
        P, gaps, deltas, opts, rec = run_arm(panel, model, case.question, labels)
        results[arm] = (P, deltas)
        path = dump_deltas(delta_csv_path(case, tag=f"e1-{arm}"), gaps, P,
                           panel.records, panel.idx, panel.weights, panel.expo,
                           panel.bands, opts, rec)
        p_hat = aggregate(P, panel.weights)
        jsd = " | ".join(f"JSD vs {name} {js_divergence(p_hat, v):.4f}"
                         for name, v in ends if v == v)
        miss = int(np.isnan(deltas).sum())
        print(f"[{arm:<5}] p_hat {p_hat:.3f} | spread {P.std():.3f} | "
              f"Δ median {np.nanmedian(deltas):+.2f} [{np.nanmin(deltas):+.2f}, "
              f"{np.nanmax(deltas):+.2f}]{f' | {miss} unparsed' if miss else ''} | {jsd}")
        print(f"        Δ dump -> {path}")

    Po, Do = results["orig"]
    Ps, Ds = results["swap"]
    p_bar = (Po + Ps) / 2.0
    judged = (Do + Ds) / 2.0          # letter-bias-free judgment per persona
    bias = (Do - Ds) / 2.0            # additive bias toward the letter A
    p_hat = aggregate(p_bar, panel.weights)
    jsd = " | ".join(f"JSD vs {name} {js_divergence(p_hat, v):.4f}"
                     for name, v in ends if v == v)
    print(f"\npermutation-averaged readout p̄ = (p_orig + p_swap)/2:")
    print(f"  p_hat {p_hat:.3f} | spread {p_bar.std():.3f} | {jsd}")
    print(f"  debiased judgment Δ̄: median {np.nanmedian(judged):+.2f} "
          f"[{np.nanmin(judged):+.2f}, {np.nanmax(judged):+.2f}] | std {np.nanstd(judged):.2f}")
    print(f"  letter-A bias β = (Δ_orig − Δ_swap)/2: median {np.nanmedian(bias):+.2f} "
          f"[{np.nanmin(bias):+.2f}, {np.nanmax(bias):+.2f}]")
    print(f"  Δ̄ by exposure band: {band_means(judged, panel, 'band')}")
    print(f"  Δ̄ by other-game verdicts: {band_means(judged, panel, 'disp')}")

    print("\nread: if the debiased level is materially less negative than the orig "
          "arm, part of the −16-nat level was label bias and the permutation-"
          "averaged readout is the candidate scored-path control; if not, label "
          "bias is excluded and the judgment itself sets the level.")


if __name__ == "__main__":
    main()

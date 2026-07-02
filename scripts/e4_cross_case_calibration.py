"""E4 — cross-case Δ→P calibration (labelled ablation M2-cal, NEVER the scored path).

The 70B orders personas correctly in logit space but reads every one of them
below P=0.004: features set the ordering, the model's normative confidence
sets the level. This ablation reconciles the two without touching the scored
path: fit a two-parameter link P = σ((Δ − b) / τ) on the FIT case's own
ground truth, freeze it, and apply it to the APPLY case's raw Δ values. The
apply case's GT is never seen by the fit — the link is cross-case-frozen,
which is what keeps this a validation ablation rather than a fit.

One aggregate GT pins only one parameter, so the fit is honest about the
remaining freedom: the scale τ is fixed by a mechanical rule (the fit case's
own Δ standard deviation), the shift b is then solved so the fit case's
weighted mean matches its GT, and a sensitivity row shows how the applied
p̂ moves across a τ grid. Links are fitted to both bracket ends (flow and
panel) and each is scored against the matching end of the apply case.

usage: e4_cross_case_calibration.py [fit_case] [apply_case] [apply_delta_csv]
       (defaults: totalwar3 payday2 <newest deltas_payday2_*.csv in out/>;
        the fit case is surveyed live at 70B n=100 unless a
        deltas_<fit_case>_*.csv from a previous run of this script exists)
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from llmsonas.cases import CASES, OUT_DIR, build_m2a_panel, delta_csv_path, dump_deltas
from llmsonas.harness import recommend_key, survey
from llmsonas.scoring.metrics import js_divergence
from llmsonas.survey.together_client import answer_probability, logit_gap

MODEL_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def solve_shift(deltas: np.ndarray, weights: np.ndarray, tau: float, target: float) -> float:
    """The b for which the weighted mean of σ((Δ−b)/τ) equals ``target``.
    The mean is monotone decreasing in b, so bisection suffices."""
    lo, hi = float(deltas.min()) - 60.0, float(deltas.max()) + 60.0

    def mean_p(b: float) -> float:
        p = sigmoid((deltas - b) / tau)
        return float((weights * p).sum() / weights.sum())

    for _ in range(200):
        mid = (lo + hi) / 2.0
        if mean_p(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def fit_deltas(case, model: str) -> Path:
    """Survey the fit case's M2a personas at 70B and dump the raw Δ CSV."""
    panel = build_m2a_panel(case)
    labels = case.answer_labels
    opts, rec = tuple(labels), recommend_key(labels)
    gaps: list[dict] = []

    def backend(msgs: list[dict], mdl: str) -> float | None:
        return answer_probability(msgs, mdl, options=opts, recommend=rec, detail=gaps)

    P = survey(panel.bios, model, case.question, labels, grounded=True, backend=backend)
    path = dump_deltas(delta_csv_path(case, tag="e4-fit"), gaps, P, panel.records,
                       panel.idx, panel.weights, panel.expo, panel.bands, opts, rec)
    print(f"fit-case survey done | Δ dump -> {path}")
    return path


def newest(pattern: str) -> Path | None:
    hits = sorted(OUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def main() -> None:
    fit_key = sys.argv[1] if len(sys.argv) > 1 else "totalwar3"
    apply_key = sys.argv[2] if len(sys.argv) > 2 else "payday2"
    fit_case = dataclasses.replace(CASES[fit_key], model=MODEL_70B, n_personas=100, pool=3000)
    apply_case = dataclasses.replace(CASES[apply_key], model=MODEL_70B, n_personas=100, pool=3000)

    print(f"E4 CROSS-CASE Δ→P CALIBRATION (ablation M2-cal, not the scored path) | "
          f"fit on {fit_case.key}, apply to {apply_case.key} | {MODEL_70B} | "
          f"run at {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    # Prefer the scored-layout (orig) arm: the label-swap/yesno dumps measure a
    # different readout and must not leak into the calibration ablation.
    apply_csv = (Path(sys.argv[3]) if len(sys.argv) > 3
                 else newest(f"deltas_{apply_key}_*e1-orig*.csv")
                 or newest(f"deltas_{apply_key}_*.csv"))
    if apply_csv is None or not apply_csv.exists():
        raise SystemExit(f"no Δ CSV for {apply_key} in {OUT_DIR} — run the E1 controls "
                         f"(or a scored run) first to produce one")

    fit_csv = newest(f"deltas_{fit_key}_*e4-fit*.csv")
    if fit_csv is None:
        fit_csv = fit_deltas(fit_case, MODEL_70B)
    else:
        print(f"reusing existing fit-case Δ dump: {fit_csv}")

    fit_panel = build_m2a_panel(fit_case)
    apply_panel = build_m2a_panel(apply_case)
    fit_df = pd.read_csv(fit_csv).dropna(subset=["delta"])
    apply_df = pd.read_csv(apply_csv).dropna(subset=["delta"])
    fD, fW = fit_df["delta"].to_numpy(), fit_df["weight"].to_numpy()
    aD, aW = apply_df["delta"].to_numpy(), apply_df["weight"].to_numpy()

    print(f"\nfit case {fit_key}: bracket [panel {fit_panel.panel_p:.3f} (n={fit_panel.panel_n}), "
          f"flow {fit_panel.gt:.3f} (n={fit_panel.gt_n})] | Δ median {np.median(fD):+.2f} "
          f"[{fD.min():+.2f}, {fD.max():+.2f}] std {fD.std():.2f} (n={len(fD)})")
    print(f"apply case {apply_key} ({apply_csv.name}): bracket [panel {apply_panel.panel_p:.3f}, "
          f"flow {apply_panel.gt:.3f}] | Δ median {np.median(aD):+.2f} "
          f"[{aD.min():+.2f}, {aD.max():+.2f}] std {aD.std():.2f} (n={len(aD)})")

    tau0 = float(fD.std())
    ends = [("flow", fit_panel.gt, apply_panel.gt),
            ("panel", fit_panel.panel_p, apply_panel.panel_p)]
    for name, fit_target, apply_target in ends:
        if fit_target != fit_target:
            print(f"\n[{name}] fit case has no {name} end — skipped")
            continue
        b = solve_shift(fD, fW, tau0, fit_target)
        aP = sigmoid((aD - b) / tau0)
        p_hat = float((aW * aP).sum() / aW.sum())
        line = f"JSD vs {name} {js_divergence(p_hat, apply_target):.4f}" \
            if apply_target == apply_target else f"(apply case has no {name} end)"
        print(f"\n[M2-cal | link fitted to {fit_key}'s {name} end, frozen]")
        print(f"  τ = std(Δ_{fit_key}) = {tau0:.2f} (mechanical) | b = {b:+.2f} "
              f"(solves fit mean = {fit_target:.3f})")
        print(f"  {apply_key}: p_hat {p_hat:.3f} | spread {aP.std():.3f} | {line}")
        by = apply_df.assign(p_cal=aP).groupby("disposition")["p_cal"].agg(["mean", "count"])
        parts = [f"{band}: {row['mean']:.3f} (n={int(row['count'])})" for band, row in by.iterrows()]
        print(f"  calibrated p by other-game verdicts: {' | '.join(parts)}")

    # The single aggregate target leaves τ free; show how much the applied
    # answer depends on that choice before anyone leans on the number.
    print(f"\nsensitivity of the flow-end link to the τ rule (b re-solved per τ):")
    print(f"  {'τ':>6} {'b':>8} {f'{apply_key} p_hat':>16} {'spread':>8}")
    for tau in (1.0, 2.0, tau0, 6.0, 10.0):
        b = solve_shift(fD, fW, tau, fit_panel.gt)
        aP = sigmoid((aD - b) / tau)
        p_hat = float((aW * aP).sum() / aW.sum())
        tag = " (=std)" if tau == tau0 else ""
        print(f"  {tau:>6.2f} {b:>+8.2f} {p_hat:>16.3f} {aP.std():>8.3f}{tag}")

    print("\nread: M2-cal is reported as a labelled ablation — persona features "
          "set the ordering, the model's normative confidence sets the level, "
          "and a cross-case-frozen link reconciles the two without the scored "
          "path ever seeing the apply case's ground truth.")


if __name__ == "__main__":
    main()

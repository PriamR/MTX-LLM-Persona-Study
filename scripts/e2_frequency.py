"""E2 — frequency elicitation (the prompt-formulation lever), rung M2-freq.

Hypothesis: the binary question reads to the model as a normative verdict
("should one recommend this?"), which it answers with confident unanimity.
Asking directly for the conditional rate — how many players out of 100 with
this history would recommend — elicits the quantity the metric actually
scores. Numbers are multi-token, so this parses a short temperature-0
generation instead of option logprobs; M2a stays in the bake-off as the
binary comparison, nothing is silently replaced.

Guardrails: same bios, one fixed neutral wording for every case
(``prompt.FREQ_QUESTION``), no valence. Success = weighted p̂ inside the
case's [panel, flow] bracket AND the disposition/exposure ordering surviving
in the per-persona rates. Known failure mode to watch: round-number clumping
(0/50/100) and spread collapse (the "out of 100 people like X" literature
finds most mass inside a 2-point window).

usage: e2_frequency.py [case_key] [model_short] [n] [pool]
       (defaults: payday2 70b 100 3000 — matches the scored 70B runs)
"""
from __future__ import annotations

import dataclasses
import re
import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from llmsonas.cases import CASES, _expo_band, _expo_disposition, build_m2a_panel
from llmsonas.scoring.metrics import aggregate, js_divergence
from llmsonas.survey.prompt import FREQ_QUESTION, frequency_messages
from llmsonas.survey.together_client import completion_text

MODELS = {
    "70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "8b": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
}


def parse_count(text: str) -> int | None:
    """First integer in the generation, accepted only when it's a valid 0-100."""
    m = re.search(r"\d+", text)
    if not m:
        return None
    n = int(m.group())
    return n if 0 <= n <= 100 else None


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "payday2"
    model = MODELS[sys.argv[2] if len(sys.argv) > 2 else "70b"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    pool = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    case = dataclasses.replace(CASES[key], model=model, n_personas=n, pool=pool)

    print(f"E2 FREQUENCY ELICITATION (M2-freq) | {case.title} | {model} | n={n} "
          f"pool={pool} seed={case.seed} | run at {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"question (frozen, all cases): {FREQ_QUESTION}")
    panel = build_m2a_panel(case)
    print(f"bracket: panel {panel.panel_p:.3f} (n={panel.panel_n}) | "
          f"flow {panel.gt:.3f} (n={panel.gt_n}) | prior {panel.pre_ratio:.3f}\n")

    counts: list[int | None] = []
    raws: list[str] = []
    for bio in panel.bios:
        text = completion_text(frequency_messages(bio), model, max_tokens=6)
        raws.append(text)
        counts.append(parse_count(text))

    known = [(i, c) for i, c in enumerate(counts) if c is not None]
    unparsed = [(i, raws[i]) for i, c in enumerate(counts) if c is None]
    P = np.array([c / 100.0 for _, c in known], dtype=float)
    w = np.array([panel.weights[i] for i, _ in known], dtype=float)
    p_hat = aggregate(P, w)

    print(f"parsed {len(known)}/{len(counts)} answers"
          + (f" | unparsed raw: {[t for _, t in unparsed]}" if unparsed else ""))
    ends = [("panel", panel.panel_p), ("flow", panel.gt)]
    jsd = " | ".join(f"JSD vs {name} {js_divergence(p_hat, v):.4f}"
                     for name, v in ends if v == v)
    if panel.panel_p == panel.panel_p:  # a panel end exists (not NaN)
        lo, hi = sorted((panel.panel_p, panel.gt))
        in_bracket = lo <= p_hat <= hi
    else:
        in_bracket = None
    print(f"M2-freq p_hat {p_hat:.3f} | spread {P.std():.3f} | {jsd}"
          + (f" | inside bracket: {'YES' if in_bracket else 'no'}" if in_bracket is not None else ""))

    hist = Counter(c for _, c in known)
    clump = sum(hist[v] for v in (0, 50, 100))
    round5 = sum(v for k, v in hist.items() if k % 5 == 0)
    print(f"answer histogram (top 12): "
          + " ".join(f"{k}:{v}" for k, v in hist.most_common(12)))
    print(f"clumping: 0/50/100 -> {clump}/{len(known)} | multiples of 5 -> {round5}/{len(known)}")

    # Does the persona ordering survive in the verbalized rates?
    for label, fn in (("exposure band", _expo_band), ("other-game verdicts", _expo_disposition)):
        by: dict[str, list[float]] = {}
        for (i, c) in known:
            sid = panel.records[panel.idx[i]].steamid
            by.setdefault(fn(panel.expo, sid), []).append(c / 100.0)
        parts = [f"{b}: {np.mean(v):.3f} (n={len(v)})" for b, v in sorted(by.items())]
        print(f"mean rate by {label}: {' | '.join(parts)}")

    # Per-persona dump for joining against the binary arms' Δ CSVs.
    import pandas as pd

    from llmsonas.cases import OUT_DIR

    rows = [{
        "rank": i + 1,
        "steamid": panel.records[panel.idx[i]].steamid,
        "weight": float(panel.weights[i]),
        "count": counts[i],
        "raw": raws[i],
    } for i in range(len(counts))]
    path = OUT_DIR / f"freq_{case.key}_{model.rsplit('/', 1)[-1]}_{time.strftime('%Y-%m-%d')}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"per-persona dump -> {path}")


if __name__ == "__main__":
    main()

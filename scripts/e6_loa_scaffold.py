"""E6 — logic-of-appropriateness scaffold (labelled ablation M2-loa, not scored).

Borrowed with provenance from Concordia / the Persona Generators paper (E5
scan, session-4 notes §10): before answering, the model first answers three
fixed questions — what kind of situation is this / what kind of person is
this / what would a person like this do — and the survey answer then
conditions on that self-description. The hypothesis is that this reframes the
readout from the normative "is this change good?" (which sets our
case-specific level) to the behavioural "what does someone like this do?".

Guardrail position: the scaffold text is frozen here in code and identical
across cases; the intermediate description is model-generated from the
facts-only bio — nothing researcher-authored is injected. That intermediate
step can still hallucinate attitudes, so every rationale is printed for
audit, and this stays a labelled ablation unless the audit is clean AND the
result earns a promotion decision from Priyam.

usage: e6_loa_scaffold.py [case_key] [model_short] [n] [pool]
       (defaults: payday2 70b 100 3000 — matches the scored runs)
"""
from __future__ import annotations

import dataclasses
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from llmsonas.cases import CASES, _expo_band, _expo_disposition, build_m2a_panel
from llmsonas.harness import recommend_key, swap_labels
from llmsonas.scoring.metrics import aggregate, js_divergence
from llmsonas.survey.prompt import _question_block
from llmsonas.survey.together_client import (
    answer_probability,
    completion_text,
    logit_gap,
)

MODELS = {
    "70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "8b": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
}

# Stage 1 — the three Concordia questions, third-person (our SDB fix), frozen.
LOA_SYSTEM = (
    "You describe, briefly and factually, one specific Steam player from their "
    "described history. Answer each question in one or two sentences."
)
LOA_QUESTIONS = (
    "1. What kind of situation is this player in?\n"
    "2. What kind of player are they?\n"
    "3. What would a player like them do in a situation like this?"
)

# Stage 2 — the standard survey system prompt, answer conditioned on stage 1.
SURVEY_SYSTEM = (
    "You simulate how one specific Steam player would answer a survey question "
    "about a game, judging only from the described player history. Reply with a "
    "single letter and nothing else."
)


def stage1(bio: str, model: str) -> str:
    msgs = [
        {"role": "system", "content": LOA_SYSTEM},
        {"role": "user", "content": f"{bio}\n\n{LOA_QUESTIONS}"},
    ]
    return completion_text(msgs, model, max_tokens=180)


def stage2(bio: str, description: str, question: str, labels: dict[str, str],
           model: str, detail: list[dict]) -> float | None:
    opts, rec = tuple(labels), recommend_key(labels)
    user = (
        f"{bio}\n\nAbout this player:\n{description}\n\n"
        f"{_question_block(question, labels)}\n"
        "Base the answer only on this player's history."
    )
    msgs = [{"role": "system", "content": SURVEY_SYSTEM},
            {"role": "user", "content": user}]
    return answer_probability(msgs, model, options=opts, recommend=rec, detail=detail)


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "payday2"
    model = MODELS[sys.argv[2] if len(sys.argv) > 2 else "70b"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    pool = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    case = dataclasses.replace(CASES[key], model=model, n_personas=n, pool=pool)
    labels = case.answer_labels
    labels_s = swap_labels(labels)

    print(f"E6 LOGIC-OF-APPROPRIATENESS SCAFFOLD (ablation M2-loa, not scored) | "
          f"{case.title} | {model} | n={n} pool={pool} seed={case.seed} | "
          f"run at {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    panel = build_m2a_panel(case)
    print(f"bracket: panel {panel.panel_p:.3f} (n={panel.panel_n}) | "
          f"flow {panel.gt:.3f} (n={panel.gt_n}) | prior {panel.pre_ratio:.3f}")
    print("permutation-averaged readout; stage-1 rationales printed in full "
          "for the attitude-hallucination audit\n")

    P_bar, d_bar = [], []
    for rank, i in enumerate(panel.idx, 1):
        bio = panel.bios[rank - 1]
        desc = stage1(bio, model)
        g_o: list[dict] = []
        g_s: list[dict] = []
        p_o = stage2(bio, desc, case.question, labels, model, g_o)
        p_s = stage2(bio, desc, case.question, labels_s, model, g_s)
        p = np.nanmean([np.nan if v is None else v for v in (p_o, p_s)])
        opts = tuple(labels)
        d_o = logit_gap(g_o[0], opts, recommend_key(labels)) if g_o else None
        d_s = logit_gap(g_s[0], opts, recommend_key(labels_s)) if g_s else None
        d = np.nanmean([np.nan if v is None else v for v in (d_o, d_s)])
        P_bar.append(p)
        d_bar.append(d)
        sid = panel.records[i].steamid
        print(f"--- persona {rank} [{_expo_band(panel.expo, sid)}/"
              f"{_expo_disposition(panel.expo, sid)}] p̄={p:.4f} Δ̄={d:+.2f}")
        print(f"    {desc}")

    P, D = np.array(P_bar, dtype=float), np.array(d_bar, dtype=float)
    ok = ~np.isnan(P)
    p_hat = aggregate(P[ok], panel.weights[ok])
    ends = [("panel", panel.panel_p), ("flow", panel.gt)]
    jsd = " | ".join(f"JSD vs {name} {js_divergence(p_hat, v):.4f}"
                     for name, v in ends if v == v)
    print(f"\nM2-loa: p_hat {p_hat:.3f} | spread {P[ok].std():.3f} | "
          f"Δ̄ median {np.nanmedian(D):+.2f} [{np.nanmin(D):+.2f}, "
          f"{np.nanmax(D):+.2f}] | {jsd}")
    print("compare against the scored M2a (permutation-averaged) from the same "
          "day's ladder run before drawing any conclusion; promotion beyond a "
          "labelled ablation needs a clean audit and Priyam's OK.")


if __name__ == "__main__":
    main()

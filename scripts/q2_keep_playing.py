"""Q2 — the second scored question: "would this player keep playing?"

The first question (recommend) scores words; this one scores feet. The dump's
``author_last_played`` gives each pre-event reviewer's final play date, so
"kept playing" is a real per-person behavioural outcome: last_played at or
after event + 30 days. That gives this question two properties the recommend
question cannot have:

* **No estimand bracket.** The people we simulate and the people we score are
  the SAME individuals — prediction vs their own later behaviour.
* **Individual-level validation.** Beyond the aggregate rate, the per-persona
  p̄ can be ranked against each persona's actual outcome (AUC), the project's
  only person-level check.

It also measures the words-vs-feet gap directly: on Payday 2 the review score
collapsed to 0.11-0.31 while ~79% of the pre-event reviewers kept playing —
a persona method that answers both questions correctly has to decouple
sentiment from behaviour.

Guardrails: ``last_played`` is scrape-time (2024) and is used ONLY as ground
truth, never in any bio; the 30-day horizon and the question wording are
frozen here before any run; same personas, template and permutation-averaged
readout as the scored recommend runs.

usage: q2_keep_playing.py [case_key] [model_short] [n] [pool]
       (defaults: payday2 70b 100 3000 — matches the scored runs)
"""
from __future__ import annotations

import dataclasses
import sys
import time
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from llmsonas.cases import CASES, _expo_disposition, build_m2a_panel
from llmsonas.construction.segment import segment_record
from llmsonas.data.dump import HD2_MEMBER, HD2_ZIP
from llmsonas.harness import recommend_key, survey_permuted, swap_labels
from llmsonas.scoring.metrics import aggregate, js_divergence
from llmsonas.survey.together_client import answer_probability

MODELS = {
    "70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "8b": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
}

# Frozen before any run: the horizon and the neutral wording. "A month" in the
# question matches the 30-day GT cut; "still playing" is behaviour, not verdict.
HORIZON_DAYS = 30
QUESTION = "Would this player still be playing the game a month after this change?"
LABELS = {"A": "Still playing", "B": "No longer playing"}


def last_played_map(appid: int) -> dict[str, int]:
    """steamid -> last_played, read directly from the zip so the field never
    enters the persona-construction path (it is scrape-time = post-event)."""
    member = HD2_MEMBER.format(appid=appid)
    with zipfile.ZipFile(HD2_ZIP) as zf, zf.open(member) as fh:
        df = pd.read_csv(fh, usecols=["author_steamid", "author_last_played"])
    lp = df.dropna().astype({"author_last_played": "int64"})
    return dict(zip(lp["author_steamid"].astype(str), lp["author_last_played"]))


def auc(scores: np.ndarray, outcomes: np.ndarray) -> float:
    """Rank AUC: P(random keeper scores above random quitter)."""
    pos, neg = scores[outcomes], scores[~outcomes]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "payday2"
    model = MODELS[sys.argv[2] if len(sys.argv) > 2 else "70b"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    pool = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    case = dataclasses.replace(CASES[key], model=model, n_personas=n, pool=pool)

    print(f"Q2 KEEP-PLAYING (second scored question, behavioural GT) | {case.title} "
          f"| {model} | n={n} pool={pool} seed={case.seed} | horizon {HORIZON_DAYS}d "
          f"| run at {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"question (frozen): {QUESTION}  [A={LABELS['A']} / B={LABELS['B']}, "
          f"permutation-averaged]")

    panel = build_m2a_panel(case)
    lp = last_played_map(case.appid)
    deadline = case.event_cutoff + HORIZON_DAYS * 86400
    kept = np.array([lp.get(panel.records[i].steamid, 0) >= deadline
                     for i in panel.idx])
    kept_rate = float((panel.weights * kept).sum() / panel.weights.sum())
    print(f"behavioural GT: {int(kept.sum())}/{len(kept)} of these exact personas "
          f"actually still played at event+{HORIZON_DAYS}d (weighted rate "
          f"{kept_rate:.3f}) — same individuals, no estimand bracket")
    print(f"contrast on record: this case's REVIEW reaction bracket is "
          f"[{panel.panel_p:.3f}, {panel.gt:.3f}] — words vs feet\n")

    labels_s = swap_labels(LABELS)
    opts = tuple(LABELS)
    rec, rec_s = recommend_key(LABELS), recommend_key(labels_s)

    def b_orig(msgs: list[dict], mdl: str) -> float | None:
        return answer_probability(msgs, mdl, options=opts, recommend=rec)

    def b_swap(msgs: list[dict], mdl: str) -> float | None:
        return answer_probability(msgs, mdl, options=opts, recommend=rec_s)

    P, _, _ = survey_permuted(panel.bios, model, QUESTION, LABELS, grounded=True,
                              backend=b_orig, backend_swap=b_swap)

    p_hat = aggregate(P, panel.weights)
    acc = float(((P >= 0.5) == kept).mean())
    print(f"predicted keep-playing rate: {p_hat:.3f} | actual {kept_rate:.3f} | "
          f"JSD {js_divergence(p_hat, kept_rate):.4f} | spread {P.std():.3f}")
    print(f"person-level: AUC {auc(P, kept):.3f} (0.5 = chance) | "
          f"accuracy at 0.5 cut: {acc:.3f} | base rate {kept.mean():.3f}")

    hist, _ = np.histogram(P, bins=np.linspace(0, 1, 11))
    print("distribution of per-persona p̄ (deciles 0.0-1.0): "
          + " ".join(str(int(c)) for c in hist))

    print("\nmean p̄ (and actual keep rate) by segment:")
    for name, fn in (
        ("investment", lambda i: segment_record(panel.records[i], panel.bands).investment),
        ("other-game verdicts", lambda i: _expo_disposition(panel.expo, panel.records[i].steamid)),
    ):
        by: dict[str, list[tuple[float, bool]]] = {}
        for pos, i in enumerate(panel.idx):
            by.setdefault(fn(i), []).append((float(P[pos]), bool(kept[pos])))
        parts = [f"{b}: {np.mean([p for p, _ in v]):.2f}/{np.mean([k for _, k in v]):.2f} "
                 f"(n={len(v)})" for b, v in sorted(by.items())]
        print(f"  {name}: {' | '.join(parts)}   [predicted/actual]")

    out = pd.DataFrame({
        "rank": range(1, len(P) + 1),
        "steamid": [panel.records[i].steamid for i in panel.idx],
        "weight": panel.weights,
        "p_keep": P,
        "kept_actual": kept,
    })
    from llmsonas.cases import OUT_DIR

    path = OUT_DIR / (f"q2keep_{case.key}_{model.rsplit('/', 1)[-1]}_"
                      f"{time.strftime('%Y-%m-%d')}.csv")
    out.to_csv(path, index=False)
    print(f"\nper-persona dump -> {path}")


if __name__ == "__main__":
    main()

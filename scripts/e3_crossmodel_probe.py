"""E3 — cross-model saturation probe: is the −16-nat level Llama-specific?

Measurement, not model-shopping: the bake-off holds its model constant, but if
another instruct family reads the same grounded bios near the decision
boundary with real crossings, the full-run model choice reopens *before* more
budget goes to Llama — a pre-run decision, not per-case tuning. Prediction on
record: same collapse, different depth (Chameleon's Limit found persona
collapse in all 10 models it measured; DeepSeek is the most convergence-prone
family in the dialog literature).

Runs the same personas, bios, template and readout as the scored M2a run
against an OpenAI-shaped chat endpoint (DeepSeek verified to return
top-20 logprobs; use deepseek-chat, NOT deepseek-reasoner, which errors on
logprobs requests). GPT-4.1-mini runs through the same path if an OpenAI key
is present.

usage: e3_crossmodel_probe.py [provider] [case_key] [k]
       (defaults: deepseek payday2 20 — first k of the n=100 M2a personas)
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx
import numpy as np

import llmsonas.config  # noqa: F401  (loads .env)
from llmsonas.cases import CASES, _expo_band, _expo_disposition, build_m2a_panel
from llmsonas.harness import recommend_key
from llmsonas.scoring.metrics import js_divergence
from llmsonas.survey.prompt import grounded_messages
from llmsonas.survey.together_client import _p_from_dist, _option_logprobs, logit_gap

# provider -> (endpoint, key env var, model id, max top_logprobs)
PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY",
                 "deepseek-chat", 20),
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY",
               "gpt-4.1-mini", 20),
}


def openai_shaped_dist(client: httpx.Client, url: str, model: str,
                       messages: list[dict], top_logprobs: int) -> dict[str, float]:
    """One-token call against an OpenAI-shaped endpoint; returns the first
    position's top-logprob dist as {token: logprob} (Together-shaped) so the
    existing readout functions apply unchanged."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    for attempt in range(5):
        try:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            content = (resp.json()["choices"][0].get("logprobs") or {}).get("content") or []
            if not content:
                return {}
            # DeepSeek pads top_logprobs with a -9999 sentinel for tokens below
            # its reporting floor; a sentinel is "unmeasurably small", not a
            # measured logprob, so it must not enter a Δ.
            return {t["token"]: t["logprob"]
                    for t in content[0].get("top_logprobs", [])
                    if t["logprob"] > -9990.0}
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == 4 or (status is not None and status not in (429, 500, 502, 503, 504)):
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    key = sys.argv[2] if len(sys.argv) > 2 else "payday2"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    url, env, model, top_lp = PROVIDERS[provider]
    api_key = os.getenv(env, "")
    if not api_key:
        raise SystemExit(f"{env} is empty — add it to .env")

    # Pin to the scored 70B sample so the depths are comparable persona-for-persona.
    case = dataclasses.replace(CASES[key], n_personas=100, pool=3000)
    labels = case.answer_labels
    opts, rec = tuple(labels), recommend_key(labels)

    print(f"E3 CROSS-MODEL SATURATION PROBE | {case.title} | {model} | first {k} "
          f"of the n=100 M2a personas (seed {case.seed}) | run at "
          f"{time.strftime('%Y-%m-%dT%H:%M:%S')}")
    panel = build_m2a_panel(case)
    print(f"bracket: panel {panel.panel_p:.3f} (n={panel.panel_n}) | "
          f"flow {panel.gt:.3f} (n={panel.gt_n}) | prior {panel.pre_ratio:.3f}")
    print(f"comparison (same personas, 70B): Δ median −16.12 [−27.00, −5.75], all P<0.004\n")

    client = httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    Ps, deltas, below_floor = [], [], 0
    for rank, i in enumerate(panel.idx[:k], 1):
        dist = openai_shaped_dist(
            client, url, model,
            grounded_messages(panel.bios[rank - 1], case.question, labels), top_lp)
        lp = _option_logprobs(dist, opts)
        p = _p_from_dist(dist, opts, rec) if dist else None
        # When the recommend token never reaches the reported top-k, its depth
        # is unmeasurable — deeper saturation than any finite Δ, not a Δ of
        # (floor − 10). Count it separately instead of faking a number.
        measurable = rec in lp
        d = logit_gap(lp, opts, rec) if measurable else None
        if lp and not measurable:
            below_floor += 1
        Ps.append(np.nan if p is None else p)
        deltas.append(np.nan if d is None else d)
        sid = panel.records[i].steamid
        print(f"persona {rank:>3} [{_expo_band(panel.expo, sid)}/"
              f"{_expo_disposition(panel.expo, sid)}] "
              f"P={'n/a' if p is None else f'{p:.4f}'} "
              f"Δ={f'{d:+.2f}' if d is not None else ('< top-' + str(top_lp) + ' floor' if lp else 'n/a')}")

    P, D = np.array(Ps, dtype=float), np.array(deltas, dtype=float)
    ok = ~np.isnan(P)
    if not ok.any():
        print("\nno parsable answers — cannot read a level from this family")
        return
    p_hat = float(P[ok].mean())
    dm = D[~np.isnan(D)]
    crossings = int((dm > 0).sum())
    depth = (f"Δ median {np.median(dm):+.2f} [{dm.min():+.2f}, {dm.max():+.2f}] "
             f"(over {len(dm)} measurable)" if len(dm) else "Δ unmeasurable")
    print(f"\n{model}: p_hat {p_hat:.3f} (unweighted, k={int(ok.sum())}) | "
          f"spread {P[ok].std():.3f} | {depth} | personas with Δ>0: {crossings} | "
          f"recommend token below the top-{top_lp} reporting floor: {below_floor}/{k}")
    ends = [("panel", panel.panel_p), ("flow", panel.gt)]
    print("  " + " | ".join(f"JSD vs {name} {js_divergence(p_hat, v):.4f}"
                            for name, v in ends if v == v))
    print("\nread: median Δ far below zero with no crossings = the normative-"
          "confidence collapse generalizes (deck evidence); a median near zero "
          "with real crossings = family-specific depth -> the full-run model "
          "choice reopens before more Llama budget is spent.")


if __name__ == "__main__":
    main()

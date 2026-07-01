"""Reasoning probe — which bio facts does the model actually use?

DIAGNOSTIC ONLY, never the scored path: free-text rationales exist to expose
attribute truncation (which persona markers survive into the model's judgment,
which get ignored), so we know what feature work is worth paying for. Nothing
here feeds a p_hat or a JSD. Same personas, bios and question as the scored M2a
run (same seed), so the rationales describe the population we actually survey.

usage: probe_reasoning.py [case_key] [model_short] [k]
       (defaults: payday2 70b 10; n_personas/pool pinned to the 100/3000 runs)
"""
from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx

from llmsonas.cases import CASES
from llmsonas.config import TOGETHER_API_KEY
from llmsonas.construction.exposure import build_footprint, exposures
from llmsonas.construction.profile import situation_bio
from llmsonas.construction.segment import population_bands, segment_record
from llmsonas.construction.select import stratified_sample
from llmsonas.data.dump import load_hd2, to_user_records

MODELS = {
    "70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "8b": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
}
N_PERSONAS, POOL = 100, 3000  # pinned to the scored runs so the sample matches

SYSTEM = (
    "You explain, briefly and factually, how one specific Steam player would "
    "answer a survey question about a game, judging only from the described "
    "player history."
)


def probe_text(client: httpx.Client, model: str, bio: str, question: str) -> str:
    user = (
        f"{bio}\n\nQuestion: {question}\n\n"
        "In two or three sentences, name the specific facts in this player's "
        "history that most influence their answer, then end with the single "
        "letter of the more likely answer: A) Recommend or B) Not recommend."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 160,
        "temperature": 0.0,
    }
    for attempt in range(4):
        try:
            resp = client.post("https://api.together.xyz/v1/chat/completions", json=body)
            resp.raise_for_status()
            return (resp.json()["choices"][0]["message"]["content"] or "").strip()
        except (httpx.HTTPStatusError, httpx.TransportError):
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    return ""


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "payday2"
    model = MODELS[sys.argv[2] if len(sys.argv) > 2 else "70b"]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    case = CASES[key]
    cut = case.event_cutoff

    df = load_hd2(case.appid)
    pre = df[(df["ts"] < cut) & (df["language"] == "english")]
    pool = pre.sample(n=min(POOL, len(pre)), random_state=case.seed)
    records = to_user_records(pool, require_review=False)
    bands = population_bands(records, cut)
    expo = exposures(build_footprint(case.appid, cut))
    idx, _ = stratified_sample(records, N_PERSONAS, seed=case.seed)

    print(f"REASONING PROBE (diagnostic, unscored) | {case.title} | {model} | "
          f"first {k} of the n={N_PERSONAS} M2a personas (seed {case.seed})")
    client = httpx.Client(
        headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    for rank, i in enumerate(idx[:k], 1):
        r = records[i]
        seg = segment_record(r, bands)
        e = expo.get(r.steamid)
        bio = situation_bio(r, case.change, bands, e)
        tags = (f"{seg.investment}/{seg.vocalness}/{seg.channel}/{seg.tenure}"
                f"/expo:{e.band if e else 'none'}"
                f"/verdicts:{e.disposition if e else 'n-a'}")
        print(f"\n--- persona {rank} [{tags}] ---")
        print(f"bio: {bio}")
        print(f"model: {probe_text(client, model, bio, case.question)}")


if __name__ == "__main__":
    main()

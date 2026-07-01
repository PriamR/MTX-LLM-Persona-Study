# Steam-LLMSonas

Modelling a group of Steam players with LLM personas, and testing which
persona-construction methods best reproduce the group's real opinions on a
consequential decision.

Evaluation is grounded in observed behaviour: personas are built from a group's
review history *before* a dated event and scored against the real recommend split
*after* it, so every result is a held-out prediction rather than a fit.

## Method ladder

- **M1** — naive prompting (control).
- **M2a** — replay of real, stratified-sampled users.
- **M2b** — behavioural cluster archetypes weighted to real proportions.
- **M3** — the stronger M2 variant placed on a homophily graph, with
  Friedkin–Johnsen opinion dynamics before answering.

Each method yields an answer distribution for a single-select question; methods
are compared by Jensen–Shannon divergence against the Steam ground truth.

## Layout

```
llmsonas/
  config.py        pinned run settings (model, target app, event cutoff, seed)
  data/            ingestion (reuses the Senti-Minted Steam clients)
  features/        per-user behavioural feature vectors
  construction/    persona selection (M2a / M2b) + third-person profiles
  survey/          survey prompts + option-token logprob extraction
  graph/           M3 homophily graph + Friedkin–Johnsen contagion
  scoring/         JS divergence, bootstrap CIs, swing error
scripts/           entry points (smoke test)
tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (use source .venv/bin/activate on *nix)
pip install -r requirements.txt

copy .env.example .env          # then edit .env and add your Together key
```

## First milestone — CS:GO smoke test

Proves the whole ladder connects end-to-end on a minimal slice (~20 personas, one
question, 3 graph rounds, Llama-3.1-8B). Needs no data download — it pulls a few
hundred recent CS:GO reviews from the public Steam reviews API.

```bash
python scripts/smoke_test.py
```

The full historical run uses the frozen Steam review dump (wired in later).

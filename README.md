# Steam-LLMSonas

LLM persona panels built from real Steam reviewers, used to predict how a game's
players react to a monetization change before it happens.

For each case we take the people who reviewed a game *before* a dated event (a
paid game adding microtransactions, a battle pass, an economy change, a
paid-to-free switch, and so on), turn each of them into a short third-person
persona from their behavioural history alone, and ask a model whether that player
would still recommend the game after the change. The prediction is scored blind
against the real recommend rate that shows up in reviews *after* the event, read
straight from a historical review dump. Nothing about the outcome touches persona
construction, so every score is a held-out prediction, not a fit.

## Result, honestly

Across six cases the grounded persona panels land inside or near the real
post-event recommend range on most of them, including one event that happened
after the model's training cutoff (so it cannot have been memorised). The method
predicts calm when the real reaction is calm and a collapse when it collapses,
rather than always leaning negative. It is not perfect: the failure modes are
measured and reported, not hidden. The main one is a level problem: the model
orders personas correctly, but the overall level follows the model's own reading
of the change, sometimes too harsh and sometimes too forgiving, which we probe
with a set of labelled control runs.

## Layout

```
llmsonas/
  config.py        pinned run settings (model, seed, the smoke-test case)
  cases.py         the scored cases and the run harness that drives them
  data/            load a game's reviews from the dump; live-pull for the smoke test
  features/        per-user behavioural feature vectors
  construction/    persona selection, segmentation, exposure, third-person bios
  survey/          survey prompt + option-token logprob readout
  graph/           the M3 homophily graph and Friedkin-Johnsen opinion dynamics
  scoring/         aggregation, bootstrap CIs, divergence
scripts/           one entry point per case, plus the control/ablation probes
tests/
```

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

copy .env.example .env          # then put your Together key in .env
```

The persona calls go to Together (`TOGETHER_API_KEY` in `.env`). You do not need a
key to check the wiring: set `LLMSONAS_OFFLINE=1` and a deterministic stub stands
in for the model, so the whole ladder runs end to end without a network call.

```bash
python -m pytest                              # 22 tests
set LLMSONAS_OFFLINE=1                         # Windows; use export on *nix
python scripts/smoke_test_payday2.py          # the Payday 2 flagship case, offline
```

The historical review dump is large and is not included here. The code expects
the "Steam Reviews 2024" per-app dump as a zip under `Historical Data/` (see
`llmsonas/data/dump.py` for the exact path and member layout). The offline flag
only swaps in a stub for the model, so the dump-backed cases still need that zip
to read their ground truth; without it they raise a clear file-not-found.

Scored runs print a methods-by-question table to stdout and drop their
transcripts and per-persona CSVs in `out/` (git-ignored).

# Changes & Decisions

A running log of the non-obvious choices made while building the first passes, and
where they deviate from the planning docs (which remain the source of truth). Dated
2026-07-01.

## Environment / packaging
- Flat package `llmsonas/` installed editable (`pip install -e .`) so `import
  llmsonas` resolves regardless of where a script is run from.
- torch installed as the **CPU** build. Fine for the smoke tests (MiniLM on a few
  hundred rows). The full run over the 15M-row dump should switch to a CUDA build
  to use the 3080.

## Model (deviation from plan)
- The plan pinned `Llama-3.1-8B-Instruct-Turbo`, but Together has moved it to
  **dedicated-endpoint only** — it no longer runs pay-per-token. Pinned
  `Meta-Llama-3-8B-Instruct-Lite` instead: serverless, cheap, exposes option-token
  logprobs (max `top_logprobs = 5`). Closest serverless 8B; fine for integrity-only
  smoke tests. **Full-run model choice is reopened** (dedicated 3.1-8B endpoint,
  serverless 3.3-70B, or GPT-4.1-mini fallback).
- Together returns logprobs in its own shape (`logprobs.top_logprobs` = list of
  `{token: logprob}`), which the OpenAI SDK does not map — so the survey client
  calls the endpoint directly and parses that shape.

## Features (simplification for the smoke test)
- Feature vector = z-scored numeric block (log1p playtime / games / reviews +
  recommend flag) ⊕ PCA-reduced MiniLM embedding of the review text.
- **Tag-exposure block omitted.** Live `appreviews` is single-app and the dump has
  no per-user library, so SteamSpy tag exposure needs Steam Web API enrichment,
  which is deferred to the full run.

## Survey
- Third-person prompt, single-token labels A/B, `max_tokens=1`, `temperature=0`,
  softmax over the A/B logprobs → per-persona P(recommend). No free-text answers.

## Graph / M3 (scope choice)
- M3 runs the **numeric** Friedkin–Johnsen update over the M2 stances (anchored,
  S = 0.5, T = 3). This satisfies "the graph runs and changes stances" cheaply and
  deterministically. The full **LLM-in-the-loop** contagion (re-prompt each round
  with neighbours' stances + grievance seeding, per Approach §4.2–4.3) is the
  fidelity upgrade, not yet built.
- M3 is built on **M2a** (individuals) for the smoke test; the "better of M2a/M2b"
  selection is deferred.

## Homophily null-shuffle check (added on request)
- `graph/nullcheck.py`: Moran's I of stance over the graph vs a label-permutation
  null (~200 shuffles, no LLM calls). Lightweight stand-in for the full
  degree-preserving rewiring null (modularity + assortativity), which stays deferred
  to the validation run per Requirements §8.

## Historical dump handling (second smoke test)
- One DuckDB pass filters CS:GO (`appid 730`) out of the ~7 GB CSVs and caches it as
  Parquet (`data/reviews_730.parquet`, git-ignored) — reruns are instant.
- CSV dialect pinned (`delim`, `quote`, `escape`, `strict_mode=false`,
  `ignore_errors=true`, no `null_padding`) to survive quoted newlines in review text.
  Parquet is read back through DuckDB to avoid a pyarrow dependency.
- **Ground truth = a tight post-event window (7 days).** Recommend recovers as F2P
  newcomers dilute the existing-owner backlash (89% → 42% at 7d, → 66% by 90d), so a
  narrow window is the faithful ground truth. The plan's peak "~29% recommend" is a
  single-day (Dec 7) figure; the 7-day aggregate is 0.416.
- Steam Web API key is **not** used here — a dump-based time-split is self-contained.
  The key is kept for profile enrichment (full run).

## Key finding (second smoke test)
- The real swing is unambiguous: CS:GO recommend **0.891 → 0.416** in the backlash
  week (n=618,793 → 10,595).
- **The personas did not predict it.** Built from owners' (mostly positive) pre-F2P
  records, M2a predicted **0.90** — it anchored on the stated prior stance and
  under-weighted the "I paid, now it's free" grievance. M1 was worse (0.999); M3
  nudged marginally toward truth (0.876). None captured the swing.
- Interpretation: isolated grounded personas echo their prior; predicting a
  behavioural *flip* likely needs (a) stronger grievance framing, (b) the
  LLM-in-the-loop M3 with grievance seeding so the narrative can propagate, and/or
  (c) a more capable model. The numeric FJ smoke version cannot manufacture the
  swing — it only averages existing stances.

## Recommended next steps
1. Build the **LLM-in-the-loop M3** with grievance seeding (the mechanism actually
   hypothesised to produce the backlash).
2. Re-run on a **stronger model** (70B or GPT-4.1-mini) to separate capability from
   method.
3. Isolate the owner group in the ground truth (post-event reviewers with pre-event
   tenure) rather than the raw window, which mixes in F2P newcomers.

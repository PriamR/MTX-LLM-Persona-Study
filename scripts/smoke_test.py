"""CS:GO smoke test — end-to-end pipeline integrity check (first pass).

Runs the full ladder (M1 -> M2a -> M2b -> M3 -> score) on a minimal slice:
~20 personas, one question (Q1a), 3 graph rounds, on Llama-3.1-8B via Together.
The goal is to prove every stage connects, not to produce valid numbers.

Flow:
    1. Ingest a few hundred recent CS:GO reviews (live appreviews).
    2. Build per-user records and behavioural feature vectors.
    3. Select 20 personas (M2a stratified sample and M2b tiny clustering).
    4. Synthesise third-person profiles.
    5. Pose Q1a; extract option-token logprobs -> per-persona probability.
    6. Build the graph, run the Friedkin-Johnsen loop (T=3), re-answer (M3).
    7. Aggregate, bootstrap, and print a methods x 1-question table with JSD.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Pipeline not implemented yet — scaffolding only. "
        "Add TOGETHER_API_KEY to .env, then build out the stages under llmsonas/."
    )


if __name__ == "__main__":
    main()

"""Pinned settings shared across the pipeline.

The smoke-test inputs are fixed before any pipeline code is written (per the
first-pass requirements) so runs are reproducible: model id, target app, event
cutoff, answer labels, and the random seed are all locked here rather than passed
around loosely.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
STEAM_API_KEY = os.getenv("STEAM_API_KEY", "")

# The Senti-Minted checkout we reuse (Steam clients, measures, viz).
SENTI_MINTED_PATH = Path(
    os.getenv("SENTI_MINTED_PATH", str(Path.home() / "Senti-Minted"))
)


@dataclass(frozen=True)
class SmokeConfig:
    """Locked inputs for the CS:GO (Q1a) smoke test."""

    # NB: the plan pinned Llama-3.1-8B-Instruct-Turbo, but Together moved it to
    # dedicated-endpoint only. Meta-Llama-3-8B-Instruct-Lite is the closest
    # serverless 8B that still exposes option-token logprobs — fine for the
    # integrity-only smoke test. Full-run model choice is revisited separately.
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
    appid: int = 730                 # CS:GO
    event_cutoff: int = 1544054400   # 2018-12-06, paid -> free-to-play
    n_personas: int = 20
    n_users: int = 300               # live review pull the personas are drawn from
    graph_rounds: int = 3            # Friedkin-Johnsen T
    knn_k: int = 8
    susceptibility: float = 0.5      # FJ anchoring S; lit finds LLM S>0.8, so anchor deliberately
    seed: int = 42

    # STUB ground truth for the smoke test only: ~71% not-recommend after the
    # CS:GO F2P conversion -> recommend ratio ~= 0.29. The real histogram (live /
    # dump) is wired in for the validation run; no accuracy claim is made here.
    gt_recommend_stub: float = 0.29

    # Single-token labels keep option-logprob extraction clean; mapped back after.
    answer_labels: dict[str, str] = field(
        default_factory=lambda: {"A": "Recommend", "B": "Not recommend"}
    )
    question: str = (
        "CS:GO, which this player paid for, is becoming free-to-play for "
        "everyone. Would this player recommend the game after the change?"
    )


SMOKE = SmokeConfig()

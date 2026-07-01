"""Run harness.

One path per method so the deliverable is a single methods x questions table.
``survey`` turns a list of personas into per-persona P(recommend); ``score``
aggregates that into p_hat with a bootstrap CI and the JS divergence vs ground
truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llmsonas.scoring.metrics import aggregate, bootstrap_ci, js_divergence
from llmsonas.survey.prompt import grounded_messages, naive_messages
from llmsonas.survey.together_client import answer_probability


@dataclass
class MethodResult:
    method: str
    p_hat: float
    ci: tuple[float, float]
    spread: float          # std of per-persona P — v1 diversity proxy
    jsd: float
    P: np.ndarray
    weights: np.ndarray


def survey(
    items: list[str],
    model: str,
    question: str,
    labels: dict[str, str],
    *,
    grounded: bool,
    fallback: float = 0.5,
) -> np.ndarray:
    """Query each persona; return per-persona P(recommend)."""
    build = grounded_messages if grounded else naive_messages
    out = []
    for item in items:
        p = answer_probability(build(item, question, labels), model)
        out.append(fallback if p is None else p)
    return np.asarray(out, dtype=float)


def score(
    method: str,
    P: np.ndarray,
    weights: np.ndarray | None,
    gt_recommend: float,
    *,
    seed: int = 42,
) -> MethodResult:
    P = np.asarray(P, dtype=float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, dtype=float)
    p_hat = aggregate(P, w)
    return MethodResult(
        method=method,
        p_hat=p_hat,
        ci=bootstrap_ci(P, w, seed=seed),
        spread=float(P.std()),
        jsd=js_divergence(p_hat, gt_recommend),
        P=P,
        weights=w,
    )

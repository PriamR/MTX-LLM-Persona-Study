"""Run harness.

One path per method so the deliverable is a single methods x questions table.
``survey`` turns a list of personas into per-persona P(recommend); ``score``
aggregates that into p_hat with a bootstrap CI and the JS divergence vs ground
truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from llmsonas.scoring.metrics import aggregate, bootstrap_ci, js_divergence
from llmsonas.survey.prompt import grounded_messages, naive_messages

# A survey backend maps (messages, model) -> P(recommend) or None. Default is the
# Together client; an offline stub can be injected for wiring runs without a key.
Backend = Callable[[list, str], Optional[float]]


@dataclass
class MethodResult:
    method: str
    p_hat: float
    ci: tuple[float, float]
    spread: float          # std of per-persona P — v1 diversity proxy
    jsd: float
    P: np.ndarray
    weights: np.ndarray


def recommend_key(labels: dict[str, str]) -> str:
    """The option key that means "recommend" — the value that is exactly
    "Recommend" (so the A/B label swap needs no extra config), else the first
    key (the historical A=Recommend layout). Variants with other wordings
    (e.g. Yes/No) pass the key explicitly instead."""
    for k, v in labels.items():
        if v.strip().lower() == "recommend":
            return k
    return next(iter(labels))


def survey(
    items: list[str],
    model: str,
    question: str,
    labels: dict[str, str],
    *,
    grounded: bool,
    fallback: float = 0.5,
    backend: Backend | None = None,
    recommend: str | None = None,
) -> np.ndarray:
    """Query each persona; return per-persona P(recommend).

    ``backend`` defaults to the Together client (imported lazily so an offline
    wiring run needn't have the HTTP client installed). Which option token
    counts as "recommend" follows ``labels`` (see ``recommend_key``) unless
    ``recommend`` overrides it.
    """
    if backend is None:
        from llmsonas.survey.together_client import answer_probability

        options = tuple(labels)
        rec = recommend or recommend_key(labels)

        def backend(msgs: list[dict], mdl: str) -> float | None:
            return answer_probability(msgs, mdl, options=options, recommend=rec)
    build = grounded_messages if grounded else naive_messages
    out = []
    for item in items:
        p = backend(build(item, question, labels), model)
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

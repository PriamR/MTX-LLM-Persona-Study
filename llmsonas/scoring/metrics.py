"""Scoring — aggregate the per-persona probabilities and compare them to the
real outcome.

The scored tables report the weighted mean p_hat with a bootstrap CI over
personas (the Desirability-Bias paper's protocol), plus the Jensen-Shannon
divergence of the implied binary answer distribution against the real
recommend split.
"""
from __future__ import annotations

import numpy as np


def aggregate(P: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted mean recommend probability p_hat."""
    P = np.asarray(P, dtype=float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, dtype=float)
    return float((w * P).sum() / w.sum())


def bootstrap_ci(
    P: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI on p_hat by resampling personas with replacement."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)

    estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        estimates[b] = (w[idx] * P[idx]).sum() / w[idx].sum()
    lo, hi = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def js_divergence(p: float, q: float) -> float:
    """JS divergence (in bits) between binary distributions [p, 1-p] and [q, 1-q]."""
    P = np.array([p, 1 - p], dtype=float)
    Q = np.array([q, 1 - q], dtype=float)
    M = 0.5 * (P + Q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)

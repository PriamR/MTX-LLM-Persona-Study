"""The homophily graph (M3 only).

Edges are *modelled* from behavioural similarity, not observed social ties: each
persona connects to its k most cosine-similar others, edge weight = similarity,
plus a few long-range edges so minority views can still travel. Columns are then
scaled by each node's influence (vocal users = hubs) and rows normalised, giving
the row-stochastic matrix W the Friedkin-Johnsen update runs on.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_influence_matrix(
    X: np.ndarray,
    k: int,
    *,
    influence: np.ndarray | None = None,
    long_range_frac: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Return the row-stochastic influence matrix W (W[i, j] = influence of j on i)."""
    n = len(X)
    k = min(k, n - 1)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)

    W = np.zeros((n, n))
    for i in range(n):
        neighbours = np.argsort(sim[i])[::-1][:k]
        for j in neighbours:
            W[i, j] = max(sim[i, j], 0.0)

    # A few long-range edges so clusters aren't fully sealed off.
    rng = np.random.default_rng(seed)
    for _ in range(max(1, round(long_range_frac * n * k))):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i != j and W[i, j] == 0:
            W[i, j] = max(sim[i, j], 0.01)

    if influence is not None:
        W = W * np.asarray(influence, dtype=float)[np.newaxis, :]  # vocal hubs

    row = W.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return W / row

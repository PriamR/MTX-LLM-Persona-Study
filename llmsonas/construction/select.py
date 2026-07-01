"""Selecting the 100 (here 20) — the one step where M2a and M2b differ.

M2a  stratified sample of real users, so sample marginals track the population.
M2b  k-means archetypes, each persona weighted by its cluster's real share.
"""
from __future__ import annotations

import numpy as np

from llmsonas.data.ingest import UserRecord


def _kmeans(X: np.ndarray, k: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """k-means (Lloyd, k-means++ seeding). Returns (labels, centres).

    Prefers scikit-learn when installed; otherwise a compact NumPy fallback, so
    the pipeline still runs in a minimal environment — the 2024-dump path needs
    neither embeddings nor scikit-learn.
    """
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        return km.labels_, km.cluster_centers_
    except ImportError:
        pass

    rng = np.random.default_rng(seed)
    centres = X[rng.integers(len(X))][None, :]
    for _ in range(1, k):  # k-means++ seeding
        d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1).min(1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(len(X), 1 / len(X))
        centres = np.vstack([centres, X[rng.choice(len(X), p=probs)]])

    labels = np.zeros(len(X), dtype=int)
    for _ in range(50):
        labels = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1).argmin(1)
        new = np.array([X[labels == c].mean(0) if np.any(labels == c) else centres[c]
                        for c in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    return labels, centres


def stratified_sample(
    records: list[UserRecord], n: int, *, seed: int = 42
) -> tuple[list[int], np.ndarray]:
    """Pick ``n`` user indices, allocated across playtime x recommend strata.

    Returns the picked indices and their weights (all 1 — each persona is one
    real person).
    """
    rng = np.random.default_rng(seed)
    pt = np.asarray([r.playtime_forever for r in records], dtype=float)
    median = np.median(pt)

    strata: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        strata.setdefault((bool(pt[i] >= median), r.voted_up), []).append(i)

    total = len(records)
    picked: list[int] = []
    for idxs in strata.values():
        take = min(len(idxs), max(1, round(n * len(idxs) / total)))
        picked.extend(rng.choice(idxs, size=take, replace=False).tolist())

    picked = list(dict.fromkeys(picked))
    if len(picked) > n:
        picked = rng.choice(picked, size=n, replace=False).tolist()
    elif len(picked) < n:
        rest = [i for i in range(total) if i not in picked]
        picked += rng.choice(rest, size=n - len(picked), replace=False).tolist()

    return picked, np.ones(len(picked))


def cluster_archetypes(
    X: np.ndarray, records: list[UserRecord], k: int, *, seed: int = 42
) -> tuple[list[int], np.ndarray]:
    """Cluster into ``k`` archetypes; each persona = the medoid real user of its
    cluster, weighted by the cluster's share of the population."""
    k = min(k, len(records))
    labels, centres = _kmeans(X, k, seed=seed)

    reps: list[int] = []
    weights: list[float] = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        dist = np.linalg.norm(X[members] - centres[c], axis=1)
        reps.append(int(members[np.argmin(dist)]))  # medoid = closest real user
        weights.append(len(members) / len(records))

    return reps, np.asarray(weights, dtype=float)

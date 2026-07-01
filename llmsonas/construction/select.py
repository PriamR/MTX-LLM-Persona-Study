"""Selecting the 100 (here 20) — the one step where M2a and M2b differ.

M2a  stratified sample of real users, so sample marginals track the population.
M2b  k-means archetypes, each persona weighted by its cluster's real share.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from llmsonas.data.ingest import UserRecord


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
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)

    reps: list[int] = []
    weights: list[float] = []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        dist = np.linalg.norm(X[members] - km.cluster_centers_[c], axis=1)
        reps.append(int(members[np.argmin(dist)]))  # medoid = closest real user
        weights.append(len(members) / len(records))

    return reps, np.asarray(weights, dtype=float)

"""Lightweight homophily validation for the smoke test.

Does the modelled homophily graph actually make sense — i.e. are personas that
the graph *connects* (behaviourally similar) also more *opinion*-aligned than you
would expect by chance? We answer it with Moran's I of the stance vector over the
graph and a label-permutation null: shuffle the stances across nodes many times
and see where the real graph's Moran's I falls in that null distribution.

If the observed I sits well above the null (small p), behavioural similarity
tracks opinion similarity, so wiring edges from behaviour is justified on this
data. This is the cheap stand-in for the full degree-preserving rewiring null
(modularity + assortativity over ~1000 rewirings), which stays deferred to the
validation run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def morans_i(W: np.ndarray, x: np.ndarray) -> float:
    """Spatial autocorrelation of ``x`` over weights ``W`` (>0 = homophily)."""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    w_total = W.sum()
    denom = (xc**2).sum()
    if w_total == 0 or denom == 0:
        return 0.0
    num = xc @ W @ xc
    return (len(x) / w_total) * (num / denom)


@dataclass
class NullCheck:
    observed: float
    null_mean: float
    null_std: float
    z: float
    p_value: float
    n_perm: int

    @property
    def passes(self) -> bool:
        # real homophily clearly above the shuffled null
        return self.p_value < 0.05 and self.observed > self.null_mean


def homophily_nullcheck(
    W: np.ndarray, stances: np.ndarray, *, n_perm: int = 200, seed: int = 42
) -> NullCheck:
    """Compare the graph's Moran's I against ``n_perm`` stance-shuffled nulls."""
    rng = np.random.default_rng(seed)
    x = np.asarray(stances, dtype=float)
    observed = morans_i(W, x)

    null = np.array([morans_i(W, rng.permutation(x)) for _ in range(n_perm)])
    mean, std = float(null.mean()), float(null.std())
    z = (observed - mean) / std if std > 0 else 0.0
    # one-sided empirical p (add-one smoothing)
    p = (1 + int(np.sum(null >= observed))) / (n_perm + 1)

    return NullCheck(observed, mean, std, z, p, n_perm)

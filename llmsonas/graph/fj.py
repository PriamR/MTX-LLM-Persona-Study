"""Friedkin-Johnsen opinion dynamics.

    x(t+1) = S . W . x(t) + (I - S) . x(1)

Each persona keeps its initial stance x(1) as an anchor with susceptibility S in
[0, 1]; S = I recovers DeGroot (pure averaging, the collapse-prone null). The
anchor is what keeps the network from sliding into fake unanimity — load-bearing
because LLM agents are measured to be highly convergent (S > 0.8).
"""
from __future__ import annotations

import numpy as np


def friedkin_johnsen(
    W: np.ndarray, x0: np.ndarray, susceptibility: float, rounds: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run the FJ update for ``rounds`` steps.

    Returns the final stance vector and the (rounds+1, n) trajectory (row 0 = x0).
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    S = np.full(n, float(susceptibility))

    x = x0.copy()
    trajectory = [x.copy()]
    for _ in range(rounds):
        x = S * (W @ x) + (1.0 - S) * x0
        trajectory.append(x.copy())

    return x, np.asarray(trajectory)

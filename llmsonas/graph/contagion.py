"""LLM-in-the-loop M3 — social propagation of *emergent* opinions.

Each round every persona re-answers while seeing how its most-similar peers
currently lean. The opinions that propagate are the personas' *own* — emergent
from their grounding plus the neutral decision already stated in the question.
Nothing about the reaction is injected, so the graph cannot become an amplifier of
a conclusion we handed it (the failure mode Approach §4.6 warns about). This is the
version whose output may be scored against ground truth.

``shock`` is a separate, explicitly-labelled sensitivity lever: inject a framing at
the vocal hubs and let it spread, to ask "*if* a shock of this kind takes hold, how
far does the network carry it?". It answers a mechanism question, never a
prediction — so it must never be the number compared to the real outcome. With
``shock=None`` (default) M3 measures only what the personas generate themselves.
"""
from __future__ import annotations

import numpy as np

from llmsonas.survey.prompt import contagion_messages
from llmsonas.survey.together_client import answer_probability

# Not used in the scored path. Only for an explicitly-labelled amplification study.
GRIEVANCE_SHOCK = (
    "Important context: this game was one that players had to pay for, and it has "
    "just been made free-to-play for everyone. Many long-time paying players feel "
    "their purchase has been devalued and are angry about the change."
)


def _neighbour_summary(x: np.ndarray, neighbours: list[int]) -> str:
    if not neighbours:
        return "There is no clear signal yet from similar players."
    rec = int(sum(x[j] >= 0.5 for j in neighbours))
    return (
        f"Among the players most similar to this one, {rec} would recommend the game "
        f"after the change and {len(neighbours) - rec} would not."
    )


def run_contagion(
    bios: list[str],
    W: np.ndarray,
    x0: np.ndarray,
    *,
    model: str,
    question: str,
    labels: dict[str, str],
    rounds: int,
    shock: str | None = None,
    seed_influence: np.ndarray | None = None,
    seed_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate emergent opinions over the graph; return final stances and the
    (rounds+1, n) trajectory.

    Default (``shock=None``) is the non-circular scored path. Passing ``shock`` runs
    the labelled amplification study: the framing is seeded at the vocal hubs
    (``seed_influence``) and spreads to anyone with a dissenting neighbour.
    """
    n = len(bios)
    x = np.asarray(x0, dtype=float).copy()
    neighbours = [np.where(W[i] > 0)[0].tolist() for i in range(n)]

    seeds: set[int] = set()
    if shock is not None and seed_influence is not None:
        n_seeds = max(1, round(seed_frac * n))
        seeds = set(np.argsort(seed_influence)[::-1][:n_seeds].tolist())

    trajectory = [x.copy()]
    for _ in range(rounds):
        x_next = x.copy()
        for i in range(n):
            inject = shock if (shock is not None and (i in seeds or any(x[j] < 0.5 for j in neighbours[i]))) else None
            p = answer_probability(
                contagion_messages(bios[i], _neighbour_summary(x, neighbours[i]), inject, question, labels),
                model,
            )
            if p is not None:
                x_next[i] = p
        x = x_next
        trajectory.append(x.copy())

    return x, np.asarray(trajectory)

"""Building the single-select survey prompt (third-person, single-letter answer).

Single-token labels (A / B) keep option-logprob extraction clean; we map them
back to Recommend / Not recommend afterwards.
"""
from __future__ import annotations

SYSTEM = (
    "You simulate how one specific Steam player would answer a survey question "
    "about a game, judging only from the described player history. Reply with a "
    "single letter and nothing else."
)


def _question_block(question: str, labels: dict[str, str]) -> str:
    options = "\n".join(f"{k}) {v}" for k, v in labels.items())
    letters = " or ".join(labels)
    return (
        f"Question: {question}\n"
        f"Answer options:\n{options}\n\n"
        f"Answer with just the single letter {letters}."
    )


def grounded_messages(bio: str, question: str, labels: dict[str, str]) -> list[dict]:
    """M2 / M3 — the persona is a real behavioural profile."""
    user = (
        f"{bio}\n\n{_question_block(question, labels)}\n"
        "Base the answer only on this player's history."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def naive_messages(descriptor: str, question: str, labels: dict[str, str]) -> list[dict]:
    """M1 control — a generic descriptor, no behavioural grounding."""
    user = f"Consider {descriptor}.\n\n{_question_block(question, labels)}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

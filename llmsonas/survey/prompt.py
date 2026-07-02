"""Building the single-select survey prompt (third-person, single-letter answer).

Single-token labels (A / B) keep option-logprob extraction clean; we map them
back to Recommend / Not recommend afterwards.
"""
from __future__ import annotations

SYSTEM = (
    "You simulate how one specific Steam player would answer a survey question "
    "about a game, judging only from the described player history. Reply with a "
    "single {unit} and nothing else."
)


def _unit(labels: dict[str, str]) -> str:
    # "letter" only describes single-character keys; a word-token variant
    # (e.g. Yes/No) keeps the identical wording otherwise.
    return "letter" if all(len(k) == 1 for k in labels) else "word"


def _question_block(question: str, labels: dict[str, str]) -> str:
    options = "\n".join(f"{k}) {v}" for k, v in labels.items())
    letters = " or ".join(labels)
    return (
        f"Question: {question}\n"
        f"Answer options:\n{options}\n\n"
        f"Answer with just the single {_unit(labels)} {letters}."
    )


def grounded_messages(bio: str, question: str, labels: dict[str, str]) -> list[dict]:
    """M2 / M3 — the persona is a real behavioural profile."""
    user = (
        f"{bio}\n\n{_question_block(question, labels)}\n"
        "Base the answer only on this player's history."
    )
    system = SYSTEM.format(unit=_unit(labels))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def naive_messages(descriptor: str, question: str, labels: dict[str, str]) -> list[dict]:
    """M1 control — a generic descriptor, no behavioural grounding."""
    user = f"Consider {descriptor}.\n\n{_question_block(question, labels)}"
    return [{"role": "system", "content": SYSTEM.format(unit=_unit(labels))}, {"role": "user", "content": user}]


# M2-freq rung: instead of a binary verdict, ask for the conditional rate the
# scored metric actually needs. One fixed neutral wording for every case — the
# decision still lives in the persona's situation, never in the question.
FREQ_QUESTION = (
    "Out of 100 players with this player's history, how many would recommend "
    "the game to other players after this change? Answer with a number from 0 "
    "to 100."
)

FREQ_SYSTEM = (
    "You estimate how specific Steam players would answer a survey question "
    "about a game, judging only from the described player history. Reply with "
    "a single whole number and nothing else."
)


def frequency_messages(bio: str, question: str = FREQ_QUESTION) -> list[dict]:
    """M2-freq — same grounded bio, frequency elicitation instead of A/B."""
    user = (
        f"{bio}\n\nQuestion: {question}\n"
        "Base the answer only on this player's history."
    )
    return [{"role": "system", "content": FREQ_SYSTEM}, {"role": "user", "content": user}]


def contagion_messages(
    bio: str,
    neighbour_summary: str,
    grievance: str | None,
    question: str,
    labels: dict[str, str],
) -> list[dict]:
    """M3 LLM-in-the-loop — the persona re-answers seeing the grievance (once it has
    reached them) and how their most-similar peers currently lean."""
    parts = [bio]
    if grievance:
        parts.append(grievance)
    parts.append(neighbour_summary)
    parts.append(_question_block(question, labels))
    parts.append("Weigh this player's own history against what similar players are saying.")
    system = SYSTEM.format(unit=_unit(labels))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]

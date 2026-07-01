"""Record -> third-person biographical profile.

Third person is deliberate: it pre-empts the social-desirability triggers that a
first-person "would you…" framing sets off (the one SDB fix that reliably works).
No demographic invention — only what the behavioural record actually contains.
"""
from __future__ import annotations

from llmsonas.data.ingest import UserRecord

# Generic, ungrounded descriptors for the M1 control — no behavioural data.
M1_PROFILES = [
    "a casual player who owns this game",
    "a dedicated player who has put many hours into this game",
    "a competitive player who takes the game seriously",
    "a player who mostly plays for fun with friends",
    "a long-time player who has followed the game for years",
    "a newer player who picked the game up recently",
]


def _hours(minutes: int) -> int:
    return round(minutes / 60)


def third_person_bio(r: UserRecord) -> str:
    # Frame the prior review as PAST behaviour on the *paid* game, not a present
    # verdict. Stating "currently recommends it" leaks the pre-event answer and the
    # model just parrots it instead of reasoning about the free-to-play change.
    stance = "gave it a positive review" if r.voted_up else "gave it a negative review"
    owned = f"{r.num_games_owned} games" if r.num_games_owned else "an unknown number of games"
    quote = r.review.replace("\n", " ").strip()[:280]
    return (
        f"This Steam user owns {owned} and has written {r.num_reviews} review(s). "
        f"In this game they have about {_hours(r.playtime_forever)} hours "
        f"({_hours(r.playtime_at_review)} at the time of their review). When they "
        f"reviewed the paid version they {stance}. "
        f'In their own words: "{quote}"'
    )


def situation_bio(r: UserRecord, change: str) -> str:
    """Situation-framing profile: neutral behavioural facts + the concrete change
    faced, with the prior verdict WITHHELD so the model must reason forward.

    The smoke test showed a positive-review bio makes the model parrot the prior
    stance (P->1.0); stripping the verdict but keeping the person's real
    investment (playtime, tenure proxy, ownership, vocalness) is the fix. The
    template is mechanical and identical across cases — only ``change`` differs —
    so nothing is tuned to a known answer. No stance, no valence, no quote.
    """
    owned = f"{r.num_games_owned} games" if r.num_games_owned else "an unknown number of games"
    reviews = "no other reviews" if r.num_reviews <= 1 else f"{r.num_reviews - 1} other review(s)"
    return (
        f"This Steam player owns {owned} and has written {reviews}. They have put "
        f"about {_hours(r.playtime_forever)} hours into this game "
        f"({_hours(r.playtime_at_review)} of them by the time they first reviewed it), "
        f"so it is a title they have genuinely invested in. {change}"
    )



def m1_profile(i: int) -> str:
    return M1_PROFILES[i % len(M1_PROFILES)]

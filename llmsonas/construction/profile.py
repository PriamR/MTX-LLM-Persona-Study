"""Record -> third-person biographical profile.

Third person is deliberate: it pre-empts the social-desirability triggers that a
first-person "would you…" framing sets off (the one SDB fix that reliably works).
No demographic invention — only what the behavioural record actually contains.
"""
from __future__ import annotations

from llmsonas.construction.exposure import Exposure, exposure_clause
from llmsonas.construction.segment import Bands, Segment, segment_record
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
    """First-pass bio used only by the live-ingest CS:GO smoke tests — NOT the
    scored path. It embeds the review quote and the prior stance, both of which
    ``situation_bio`` (the scored bio builder) deliberately withholds so the
    model has to reason forward from behaviour alone."""
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


_INVESTMENT = {
    "light": "has barely played the game",
    "casual": "is a casual player of the game",
    "regular": "is a regular player of the game",
    "dedicated": "is a dedicated player of the game",
    "hardcore": "is one of the most invested players in this community",
}
_VOCALNESS = {
    "quiet": "They almost never write reviews",
    "occasional": "They review games occasionally",
    "vocal": "They are a vocal reviewer",
    "prolific": "They review games constantly",
}
_LIBRARY = {
    "focused": "They own only a small library of games",
    "moderate": "They own a moderate library of games",
    "broad": "They own a large, varied library of games",
}
_CHANNEL = {
    "steam": "They paid for the game on Steam.",
    "key": "They did not buy it on Steam — they activated it from a key or bundle.",
    "free": "They received the game for free rather than paying for it.",
}
_TENURE = {
    "recent": "They only picked the game up fairly recently before this.",
    "veteran": "They have owned and played it since long before this — a long-time owner.",
}


def _investment_clause(r: UserRecord, seg: Segment) -> str:
    hours = _hours(r.playtime_forever)
    if seg.investment == "light":
        return f"This Steam player {_INVESTMENT['light']} (about {hours} hours)"
    return f"This Steam player {_INVESTMENT[seg.investment]}, with about {hours} hours in it"


def situation_bio(
    r: UserRecord, change: str, bands: Bands, exposure: Exposure | None = None
) -> str:
    """Situation-framing profile: neutral behavioural facts + the concrete change
    faced, with the prior verdict WITHHELD so the model must reason forward.

    Facts are stated *relative to the population* (``bands``) so that light and
    hardcore players, quiet and prolific reviewers, single-game loyalists and
    broad collectors read as genuinely different people — the segmentation that
    stops 100 personas collapsing into one voice (see ``segment.py``). The
    template is mechanical and identical across cases — only ``change`` differs —
    so nothing is tuned to a known answer. No stance, no valence, no quote.

    ``exposure`` (when the user has a cross-app footprint) adds the one axis a
    monetisation change actually splits on: whether this player already plays
    games financed by in-game purchases (see ``exposure.py``).
    """
    seg = segment_record(r, bands)
    parts = [_investment_clause(r, seg) + "."]
    parts.append(f"{_VOCALNESS[seg.vocalness]} ({r.num_reviews} to date).")

    if seg.loyal_mono:
        parts.append(
            "They own very few other games, so nearly all of their playtime goes to "
            "this one title — a loyal, single-game player."
        )
    elif seg.library != "unknown":
        parts.append(f"{_LIBRARY[seg.library]}.")

    if seg.review_timing == "early":
        parts.append(
            f"They formed their view of it early — after about {_hours(r.playtime_at_review)} "
            "hours — and kept playing well past that point."
        )
    elif seg.review_timing == "veteran":
        parts.append(
            f"They held off judging it until they had put in serious time "
            f"(about {_hours(r.playtime_at_review)} hours)."
        )

    parts.append(_CHANNEL[seg.channel])
    if seg.early_access:
        parts.append("They backed the game during its early access.")
    if seg.tenure in _TENURE:
        parts.append(_TENURE[seg.tenure])

    clause = exposure_clause(exposure)
    if clause:
        parts.append(clause)

    parts.append(change)
    return " ".join(parts)



def m1_profile(i: int) -> str:
    return M1_PROFILES[i % len(M1_PROFILES)]

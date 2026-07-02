"""RuneScape smoke test — umbrella tier (expanding recurrent monetization)

Jagex launched the Hero Pass battle pass in September 2023 - a paid premium
track with gameplay-affecting rewards on top of an existing subscription -
and recommend collapsed 0.861 -> 0.081 in a week, the largest swing in the
case set, before a walkback that the 30-day recovery makes visible in the
data. The 7-day window is the pre-reversal reaction.

Set LLMSONAS_OFFLINE=1 (or leave TOGETHER_API_KEY empty) to run the whole ladder
with a deterministic stub in place of the model, proving the wiring without a
network round-trip.
"""
from __future__ import annotations

from llmsonas.cases import RUNESCAPE, run_dump_smoke


def main() -> None:
    run_dump_smoke(RUNESCAPE)


if __name__ == "__main__":
    main()

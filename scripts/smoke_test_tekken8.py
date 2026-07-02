"""Tekken 8 smoke test — the second in-scope premium-MTX case.

Bandai Namco launched Tekken 8 at full price in January 2024 and added the
real-money "Tekken Shop" one month later (2024-02-28), with a paid battle
pass following in early April; recommend slid 0.835 -> 0.560 over the next 45
days. Two properties make it the ideal complement to Payday 2: the backlash
came heavily from EXISTING owners editing their reviews (the panel our
personas actually simulate — 1,153 edits landing at 0.261), and the event
postdates the 70B's training data, so the prediction is held-out by
construction, no memorisation caveat.

Set LLMSONAS_OFFLINE=1 (or leave TOGETHER_API_KEY empty) to run the whole ladder
with a deterministic stub in place of the model, proving the wiring without a
network round-trip.
"""
from __future__ import annotations

from llmsonas.cases import TEKKEN8, run_dump_smoke


def main() -> None:
    run_dump_smoke(TEKKEN8)


if __name__ == "__main__":
    main()

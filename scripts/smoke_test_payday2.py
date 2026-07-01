"""Payday 2 microtransaction smoke test — the primary graph flagship.

Overkill promised Payday 2 would never have microtransactions, then added paid
safes and drills in October 2015; recommend fell 0.87 -> 0.31 in the backlash
week (and the never-edited reviews are more negative still, so the swing is real,
not a reversion artefact). Personas are built from players who reviewed *before*
that date and asked whether they'd recommend the game after it — a held-out
time-split scored against the recommend ratio read straight from the dump.

Set LLMSONAS_OFFLINE=1 (or leave TOGETHER_API_KEY empty) to run the whole ladder
with a deterministic stub in place of the model, proving the wiring without a
network round-trip.
"""
from __future__ import annotations

from llmsonas.cases import PAYDAY2, run_dump_smoke


def main() -> None:
    run_dump_smoke(PAYDAY2)


if __name__ == "__main__":
    main()

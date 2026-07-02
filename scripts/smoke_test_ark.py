"""ARK: Survival Evolved smoke test — the third same-class scored case.

Studio Wildcard released the paid Scorched Earth expansion in September 2016
while the base game was still in Early Access; recommend fell 0.74 -> 0.31 in
the backlash week. Same decision class as Payday 2 and TW3 (paid content into
a promise-laden game), which is what the reference-class claim needs: N cases
of one decision type, not one anecdote. Screened locally before any model
budget (2026-07-02): swing verified, target bracket [0.249, 0.309] — the
tightest of the scored cases.

Set LLMSONAS_OFFLINE=1 (or leave TOGETHER_API_KEY empty) to run the whole ladder
with a deterministic stub in place of the model, proving the wiring without a
network round-trip.
"""
from __future__ import annotations

from llmsonas.cases import ARK, run_dump_smoke


def main() -> None:
    run_dump_smoke(ARK)


if __name__ == "__main__":
    main()

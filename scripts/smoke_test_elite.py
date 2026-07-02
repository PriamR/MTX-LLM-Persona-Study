"""Elite Dangerous smoke test — the in-scope calm case

Frontier moved Elite Dangerous' cosmetics store in-game in September 2019
with the dual-earn ARX currency, cosmetic items only - the forgiving form of
the add-MTX decision, and the store page barely moved (0.721 -> 0.616 over
the week). The case therefore tests the opposite skill to Payday 2: does the
method predict CALM when calm is the real answer, instead of always
forecasting a backlash.

Set LLMSONAS_OFFLINE=1 (or leave TOGETHER_API_KEY empty) to run the whole ladder
with a deterministic stub in place of the model, proving the wiring without a
network round-trip.
"""
from __future__ import annotations

from llmsonas.cases import ELITE, run_dump_smoke


def main() -> None:
    run_dump_smoke(ELITE)


if __name__ == "__main__":
    main()

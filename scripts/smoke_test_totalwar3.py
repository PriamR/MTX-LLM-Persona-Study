"""Total War: Warhammer III smoke test — the second-backlash validation case.

The "Shadows of Change" DLC (announced 2023-08-08) added fewer new units than
earlier DLC at the same price; recommend fell 0.85 -> 0.24 within days and stayed
down (never-edited reviews are just as negative). A backlash on a different
mechanism from Payday 2 (DLC value, not microtransactions) and a less-memorised
event, so it tests whether the method *replicates* rather than fits one case.

Set LLMSONAS_OFFLINE=1 (or leave TOGETHER_API_KEY empty) to run with a
deterministic stub in place of the model.
"""
from __future__ import annotations

from llmsonas.cases import TOTALWAR3, run_dump_smoke


def main() -> None:
    run_dump_smoke(TOTALWAR3)


if __name__ == "__main__":
    main()

"""War Thunder smoke test — umbrella tier (expanding recurrent monetization)

Gaijin announced economy changes in May 2023 that raised repair costs and
cut battle rewards in the free-to-play War Thunder; recommend collapsed
0.772 -> 0.164 in the week, with 3,088 existing owners editing their reviews
down to 0.058. Not a premium-MTX case: it anchors the broader umbrella of a
live-service game tightening its recurrent monetization. Partial reversal
came weeks later, after the scored window.

Set LLMSONAS_OFFLINE=1 (or leave TOGETHER_API_KEY empty) to run the whole ladder
with a deterministic stub in place of the model, proving the wiring without a
network round-trip.
"""
from __future__ import annotations

from llmsonas.cases import WARTHUNDER, run_dump_smoke


def main() -> None:
    run_dump_smoke(WARTHUNDER)


if __name__ == "__main__":
    main()

"""No Man's Sky smoke test — the redemption (up-swing) validation case.

The free "NEXT" update (2018-07-24) added multiplayer, base building and a visual
overhaul; recommend rose after it. Personas are built from the accumulated (and
launch-scarred) reviewer base and asked whether they'd recommend the game after
the update. This is the direction-generality check: the method must react to the
*content* of a change, not apply a fixed "change = bad" prior — the sharpest test
against the anchoring finding. Caveat: NMS's redemption is famous, so a capable
model may recall it rather than reason from persona attributes.

Set LLMSONAS_OFFLINE=1 (or leave TOGETHER_API_KEY empty) to run with a
deterministic stub in place of the model.
"""
from __future__ import annotations

from llmsonas.cases import NOMANSSKY, run_dump_smoke


def main() -> None:
    run_dump_smoke(NOMANSSKY)


if __name__ == "__main__":
    main()

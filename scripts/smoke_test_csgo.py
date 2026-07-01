"""CS:GO paid -> free-to-play smoke test — the anchoring diagnostic, re-run.

CS:GO went free-to-play on 2018-12-06, devaluing every existing paid owner at
once. It is a *universal shock*, not a divisive split: there is no attribute that
cleanly separates who stays positive, so the earlier finding was that personas
anchor on their prior and the homophily graph has nothing to carry. This re-runs
CS:GO on the same enriched, situation-framed ladder as the other cases to see
whether the segmentation changes that — or whether a genuine shock still collapses,
as the thesis predicts it should.

Note: CS:GO (appid 730) is a very large member of the 2024 dump (~7.6M rows), so
the load is heavier than the other cases.

Set LLMSONAS_OFFLINE=1 (or leave TOGETHER_API_KEY empty) to run the whole ladder
with a deterministic stub in place of the model.
"""
from __future__ import annotations

from llmsonas.cases import CSGO, run_dump_smoke


def main() -> None:
    run_dump_smoke(CSGO)


if __name__ == "__main__":
    main()

"""Helldivers 2 PSN-linking smoke test — the divisive-contrast case.

Kept as a documented negative example: this dump was scraped after Sony reversed
the mandatory-PSN policy, ~58% of the post-event window was edited, and the
measured post-event recommend ratio comes out *positive* (~0.84) — so the live
histogram's backlash is not reconstructable from it. Run it to see the pipeline
connect on the 2024 schema and to make that ground-truth caveat concrete; the
scored flagship is Payday 2 (``smoke_test_payday2.py``).

Set LLMSONAS_OFFLINE=1 (or leave TOGETHER_API_KEY empty) to run with a
deterministic stub in place of the model.
"""
from __future__ import annotations

from llmsonas.cases import HELLDIVERS2, run_dump_smoke


def main() -> None:
    run_dump_smoke(HELLDIVERS2)


if __name__ == "__main__":
    main()

"""Render the deliverable's figures from the scored transcripts in ``out/``.

Every number plotted here is parsed from a run transcript, never typed in, so
the figures stay verifiable against the same artifacts the tables cite. Output
lands in ``deliverable/figs/`` as PNGs referenced by ``submission.md``.

Usage: .venv\\Scripts\\python.exe scripts/make_figures.py
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = ROOT / "deliverable" / "figs"

DATE = "2026-07-02"

# (transcript, display label, role) — the six cases reported in submission.md §5.
LADDERS = [
    (f"payday2_llama70b_permavg_{DATE}.txt", "Payday 2", "pillar"),
    (f"tekken8_llama70b_permavg_{DATE}.txt", "Tekken 8 (held-out)", "pillar"),
    (f"elite_llama70b_permavg_{DATE}.txt", "Elite Dangerous", "pillar"),
    (f"warthunder_llama70b_permavg_{DATE}.txt", "War Thunder", "umbrella"),
    (f"runescape_llama70b_permavg_{DATE}.txt", "RuneScape", "umbrella"),
    (f"nomanssky_llama70b_permavg_{DATE}.txt", "No Man's Sky", "control"),
]

MENUS = [
    (f"menu_payday2_70b_{DATE}.txt", "Payday 2"),
    (f"menu_tekken8_70b_{DATE}.txt", "Tekken 8"),
]

ROLE_COLOR = {"pillar": "#1f77b4", "umbrella": "#2ca02c", "control": "#7f7f7f"}


@dataclass
class Ladder:
    label: str
    role: str
    prior: float
    flow: float
    panel: float
    m1: float
    m2a: float
    m2a_ci: tuple[float, float]


def parse_ladder(path: Path, label: str, role: str) -> Ladder:
    text = path.read_text(encoding="utf-8")
    prior = float(re.search(r"pre-event recommend ([\d.]+)", text).group(1))
    flow = float(re.search(r"ground truth = ([\d.]+)", text).group(1))
    panel = float(re.search(r"edited in-window: ([\d.]+)", text).group(1))
    rows = {
        m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        for m in re.finditer(
            r"^(M\d\w?)\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]", text, re.MULTILINE
        )
    }
    return Ladder(
        label=label, role=role, prior=prior, flow=flow, panel=panel,
        m1=rows["M1"][0], m2a=rows["M2a"][0], m2a_ci=(rows["M2a"][1], rows["M2a"][2]),
    )


def parse_menu(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Variant p̂ values plus the transcript's own printed deltas-vs-shipped,
    so the figure annotations match the text's rounding exactly."""
    text = path.read_text(encoding="utf-8")
    phats = {
        m.group(1): float(m.group(2))
        for m in re.finditer(r"^\[(\w+)\s*\] p_hat ([\d.]+)", text, re.MULTILINE)
    }
    deltas = {
        m.group(1): float(m.group(2))
        for m in re.finditer(r"(\w+): ([+-][\d.]+)", text.split("deltas vs shipped:")[1])
    }
    return phats, deltas


def fig_scorecard(ladders: list[Ladder]) -> None:
    """Per case: the [panel, flow] bracket as a band, M2a with its 95% CI, and
    the naive M1 control — the whole scored story in one panel."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ys = list(range(len(ladders)))[::-1]
    for y, lad in zip(ys, ladders):
        lo, hi = sorted((lad.panel, lad.flow))
        ax.plot([lo, hi], [y, y], lw=9, color=ROLE_COLOR[lad.role], alpha=0.25,
                solid_capstyle="butt",
                label="_" if y != ys[0] else "ground-truth bracket [panel, flow]")
        ax.errorbar(
            lad.m2a, y,
            xerr=[[lad.m2a - lad.m2a_ci[0]], [lad.m2a_ci[1] - lad.m2a]],
            fmt="o", color=ROLE_COLOR[lad.role], ms=7, capsize=3,
            label="_" if y != ys[0] else "M2a (grounded personas, 95% CI)")
        ax.plot(lad.m1, y, "x", color="#d62728", ms=8, mew=2,
                label="_" if y != ys[0] else "M1 (naive control)")
        ax.plot(lad.prior, y, "|", color="black", ms=14, mew=1.5,
                label="_" if y != ys[0] else "pre-event prior")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{l.label}\n({l.role})" for l in ladders], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("recommend rate")
    ax.set_title("Predicted vs real reaction, all reported cases (70B, n=100, seed 42)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8,
              framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_scorecard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pred_vs_real(ladders: list[Ladder]) -> None:
    """Predicted M2a against the real flow ground truth; the diagonal is a
    perfect forecast, the horizontal span is the [panel, flow] bracket."""
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="perfect prediction")
    for lad in ladders:
        lo, hi = sorted((lad.panel, lad.flow))
        ax.plot([lo, hi], [lad.m2a, lad.m2a], lw=2, color=ROLE_COLOR[lad.role], alpha=0.5)
        ax.plot(lad.flow, lad.m2a, "o", color=ROLE_COLOR[lad.role], ms=8)
        ax.annotate(lad.label, (lad.flow, lad.m2a), textcoords="offset points",
                    xytext=(7, -3), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("real post-event recommend rate (dot = flow GT; line = bracket)")
    ax.set_ylabel("predicted recommend rate (M2a)")
    ax.set_title("Severity tracks reality across cases")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pred_vs_real.png", dpi=200)
    plt.close(fig)


def fig_menus(menus: dict[str, dict[str, float]]) -> None:
    """The two counterfactual decision menus: p̂ per variant, shipped row
    highlighted. Labelled extrapolation — only the shipped row touches GT."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, (title, (rows, deltas)) in zip(axes, menus.items()):
        names = list(rows)
        vals = [rows[n] for n in names]
        colors = ["#d62728" if n == "shipped" else "#1f77b4" for n in names]
        ypos = list(range(len(names)))[::-1]
        ax.barh(ypos, vals, color=colors, alpha=0.85, height=0.6)
        for y, n, v in zip(ypos, names, vals):
            note = (" (shipped, anchored on GT)" if n == "shipped"
                    else f"  {deltas[n]:+.3f} vs shipped")
            ax.text(v + 0.015, y, f"{v:.3f}{note}", va="center", fontsize=7.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8.5)
        ax.set_xlim(0, 1.28)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("predicted recommend rate (p̂)")
        ax.set_title(f"{title} — decision menu\n(extrapolation; GT anchors shipped row only)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_menus.png", dpi=200)
    plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    FIGS.mkdir(parents=True, exist_ok=True)
    ladders = [parse_ladder(OUT / f, label, role) for f, label, role in LADDERS]
    menus = {label: parse_menu(OUT / f) for f, label in MENUS}
    fig_scorecard(ladders)
    fig_pred_vs_real(ladders)
    fig_menus(menus)
    for lad in ladders:
        print(f"{lad.label:<22} M2a {lad.m2a:.3f} {lad.m2a_ci} | bracket "
              f"[{min(lad.panel, lad.flow):.3f}, {max(lad.panel, lad.flow):.3f}] | M1 {lad.m1:.3f}")
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()

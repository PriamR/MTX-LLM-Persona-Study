"""Render the stakeholder brief's plain-language figures from the transcripts.

The brief bans internal jargon, so these figures do too: percentage axes,
plain labels, no method names, no seeds. Every plotted number is parsed from
a run transcript in ``out/``, never typed in. Output: deliverable/figs/fig_sh_*.png.

Usage: .venv\\Scripts\\python.exe scripts/make_stakeholder_figures.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = ROOT / "deliverable" / "figs"
DATE = "2026-07-02"

BLUE, RED, GRAY, GREEN = "#1f77b4", "#d62728", "#666666", "#2ca02c"

LADDERS = [
    (f"payday2_llama70b_permavg_{DATE}.txt", "Payday 2"),
    (f"tekken8_llama70b_permavg_{DATE}.txt", "Tekken 8"),
    (f"elite_llama70b_permavg_{DATE}.txt", "Elite Dangerous"),
    (f"warthunder_llama70b_permavg_{DATE}.txt", "War Thunder"),
    (f"runescape_llama70b_permavg_{DATE}.txt", "RuneScape"),
    (f"nomanssky_llama70b_permavg_{DATE}.txt", "No Man's Sky"),
]


def parse_ladder(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m2a = re.search(r"^M2a\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]", text, re.MULTILINE)
    m1 = re.search(r"^M1\s+([\d.]+)", text, re.MULTILINE)
    return {
        "prior": float(re.search(r"pre-event recommend ([\d.]+)", text).group(1)),
        "flow": float(re.search(r"ground truth = ([\d.]+)", text).group(1)),
        "panel": float(re.search(r"edited in-window: ([\d.]+)", text).group(1)),
        "pred": float(m2a.group(1)),
        "ci": (float(m2a.group(2)), float(m2a.group(3))),
        "m1": float(m1.group(1)),
    }


def parse_menu(path: Path) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    text = path.read_text(encoding="utf-8")
    phats, segs = {}, {}
    for m in re.finditer(r"^\[(\w+)\s*\] p_hat ([\d.]+)", text, re.MULTILINE):
        phats[m.group(1)] = float(m.group(2))
    for name, seg_line in re.findall(
        r"^\[(\w+)\s*\] p_hat.*?p̄ by other-game verdicts: ([^\n]+)", text,
        re.MULTILINE | re.DOTALL,
    ):
        segs[name] = {k: float(v) for k, v in re.findall(r"(\w+):([\d.]+)", seg_line)}
    deltas = {
        m.group(1): float(m.group(2))
        for m in re.finditer(r"(\w+): ([+-][\d.]+)", text.split("deltas vs shipped:")[1])
    }
    return phats, deltas, segs


def fig_pipeline() -> None:
    """How it works, in five plain steps."""
    steps = [
        ("Real players", "thousands of reviewers of\nthe game, from before\nthe change"),
        ("Facts-only profiles", "hours, spending, tenure,\nreview habits. Never\ntheir opinions"),
        ("One neutral question", "“Would this player\nrecommend the game\nafter this change?”"),
        ("Panel prediction", "the share who would still\nrecommend, with an\nuncertainty range"),
        ("Checked against reality", "the real approval record\nafter the change\nactually happened"),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 2.7))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 2.7)
    ax.axis("off")
    w, h, y = 2.15, 1.9, 0.4
    xs = [0.15 + i * 2.5 for i in range(5)]
    for i, (x, (title, sub)) in enumerate(zip(xs, steps)):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                             facecolor="#eaf1f8" if i < 4 else "#e8f4e8",
                             edgecolor=BLUE if i < 4 else GREEN, lw=1.4)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.38, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.42, sub, ha="center", va="center", fontsize=8.6)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + w + 0.07, y + h / 2), (x + 2.5 - 0.07, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=16, color="#444444"))
    fig.tight_layout()
    fig.savefig(FIGS / "fig_sh_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_scorecard(cases: dict[str, dict]) -> None:
    """Predicted vs real, plain labels, percent axis."""
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    names = list(cases)
    ys = list(range(len(names)))[::-1]
    for y, name in zip(ys, names):
        c = cases[name]
        lo, hi = sorted((c["panel"] * 100, c["flow"] * 100))
        ax.plot([lo, hi], [y, y], lw=10, color=GRAY, alpha=0.35, solid_capstyle="butt",
                label="_" if y != ys[0] else "what really happened (the honest range)")
        ax.errorbar(c["pred"] * 100, y,
                    xerr=[[c["pred"] * 100 - c["ci"][0] * 100], [c["ci"][1] * 100 - c["pred"] * 100]],
                    fmt="o", color=BLUE, ms=8, capsize=3,
                    label="_" if y != ys[0] else "our panel's blind prediction")
        ax.plot(c["m1"] * 100, y, "x", color=RED, ms=9, mew=2.2,
                label="_" if y != ys[0] else "a generic chatbot with no player data")
        ax.plot(c["prior"] * 100, y, "|", color="black", ms=15, mew=1.6,
                label="_" if y != ys[0] else "approval before the change")
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of players recommending the game (%)", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9.2, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_sh_scorecard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_range(cases: dict[str, dict]) -> None:
    """Approval points lost across the precedents, mild to severe."""
    drops = {n: (c["prior"] - c["flow"]) * 100 for n, c in cases.items() if c["flow"] < c["prior"]}
    fig, ax = plt.subplots(figsize=(8.6, 2.4))
    ax.axhline(0, color="#444444", lw=1.2)
    above = True
    for name, d in sorted(drops.items(), key=lambda kv: kv[1]):
        ax.plot(d, 0, "o", ms=9, color=RED, alpha=0.85)
        ax.annotate(f"{name}\n−{d:.0f} pts", (d, 0), textcoords="offset points",
                    xytext=(0, 14 if above else -34), ha="center", fontsize=9)
        above = not above
    ax.set_xlim(0, 85)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("approval points lost in the immediate reaction window", fontsize=10.5)
    ax.text(2, 0.72, "mild", fontsize=9.5, color="#444444", style="italic")
    ax.text(82, 0.72, "severe", fontsize=9.5, color="#444444", style="italic", ha="right")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_sh_range.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_levers() -> None:
    """The two decision menus, plain labels, percent axis."""
    label_map = {
        "payday2": ("Payday 2: what could have shipped instead", {
            "shipped": "What shipped: pay-to-win items\n+ a broken no-MTX promise",
            "cosmetic_only": "Cosmetic items only",
            "no_promise": "Same items, promise never made",
            "earnable_drills": "Earnable instead of purchasable",
        }),
        "tekken8": ("Tekken 8: what could have shipped instead", {
            "shipped": "What shipped: cash shop\n+ paid battle pass",
            "earnable_currency": "Shop currency earnable in game",
            "announced_upfront": "Announced before launch",
            "no_battle_pass": "Shop only, no battle pass",
        }),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.7))
    for ax, key in zip(axes, ("payday2", "tekken8")):
        phats, deltas, _ = parse_menu(OUT / f"menu_{key}_70b_{DATE}.txt")
        title, labels = label_map[key]
        order = ["shipped"] + sorted(deltas, key=deltas.get, reverse=True)
        ypos = list(range(len(order)))[::-1]
        vals = [phats[n] * 100 for n in order]
        colors = [RED if n == "shipped" else BLUE for n in order]
        ax.barh(ypos, vals, color=colors, alpha=0.85, height=0.62)
        for y, n, v in zip(ypos, order, vals):
            note = "" if n == "shipped" else f"  +{deltas[n] * 100:.0f} pts"
            ax.text(v + 1.5, y, f"{v:.0f}%{note}", va="center", fontsize=9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([labels[n] for n in order], fontsize=9)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("predicted share recommending (%)", fontsize=10)
        ax.set_title(title, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_sh_levers.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_whoflips() -> None:
    """Predicted approval by player segment, Tekken 8 panel, shipped decision."""
    _, _, segs = parse_menu(OUT / f"menu_tekken8_70b_{DATE}.txt")
    seg = segs["shipped"]
    labels = [
        ("all", "Recommends everything\nthey review"),
        ("none", "No other reviews\non record"),
        ("most", "Mostly positive\nreviewer"),
        ("half", "Mixed reviewer"),
        ("few", "Habitual critic"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ypos = list(range(len(labels)))[::-1]
    vals = [seg[k] * 100 for k, _ in labels]
    colors = [BLUE if v >= 50 else RED for v in vals]
    ax.barh(ypos, vals, color=colors, alpha=0.8, height=0.6)
    for y, v in zip(ypos, vals):
        ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=9.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([l for _, l in labels], fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("predicted share still recommending after the change (%)", fontsize=10)
    ax.axvline(50, color="#888888", lw=0.9, ls=":")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_sh_whoflips.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_wordsfeet() -> None:
    """What players said vs what they did, Payday 2."""
    text = (OUT / f"q2keep_payday2_70b_{DATE}.txt").read_text(encoding="utf-8")
    kept = float(re.search(r"weighted rate ([\d.]+)", text).group(1)) * 100
    lo, hi = (float(x) * 100 for x in
              re.search(r"REVIEW reaction bracket is \[([\d.]+), ([\d.]+)\]", text).groups())
    fig, ax = plt.subplots(figsize=(8.2, 2.5))
    ax.barh(1, kept, color=BLUE, alpha=0.85, height=0.52)
    ax.barh(0, lo, color=RED, alpha=0.85, height=0.52)
    ax.barh(0, hi - lo, left=lo, color=RED, alpha=0.4, height=0.52)
    ax.text(kept + 1.5, 1, f"{kept:.0f}%", va="center", fontsize=10.5)
    # round half away from zero so the label matches the text's "11-31%"
    ax.text(hi + 1.5, 0, f"{int(lo + 0.5)}–{int(hi + 0.5)}% (range)", va="center", fontsize=10.5)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Still playing the game\na month later",
                        "Recommending the game\nin the same window"], fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the same players (%)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_sh_wordsfeet.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    FIGS.mkdir(parents=True, exist_ok=True)
    cases = {name: parse_ladder(OUT / f) for f, name in LADDERS}
    fig_pipeline()
    fig_scorecard(cases)
    fig_range(cases)
    fig_levers()
    fig_whoflips()
    fig_wordsfeet()
    for name, c in cases.items():
        print(f"{name:<16} pred {c['pred']:.3f} | real [{min(c['panel'], c['flow']):.3f}, "
              f"{max(c['panel'], c['flow']):.3f}] | prior {c['prior']:.3f}")
    print(f"figures -> {FIGS} (fig_sh_*.png)")


if __name__ == "__main__":
    main()

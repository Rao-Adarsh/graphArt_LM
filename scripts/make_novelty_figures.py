"""Figures for the 2-page novelty-assessment report (report/novelty/).

    .venv/Scripts/python.exe scripts/make_novelty_figures.py

Every number here is copied from docs/experiments.md / docs/research_story.md (the scored
runs); nothing is recomputed. Writes PDFs to report/novelty/figures/.
"""

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parents[1] / "report" / "novelty" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# reference palette, light mode (dataviz skill): slots 1-3 + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8d8c87", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def legend(ax, items):
    ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=l) for l, c in items], loc="upper center", columnspacing=1.0,
              bbox_to_anchor=(0.5, -0.16 if ax.get_xlabel() == "" else -0.3), ncol=len(items), frameon=False, handlelength=1.2)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def hbars(ax, labels, k, n, color, label_fmt="{k}/{n}"):
    """Horizontal bars of k/n in percent, top to bottom, labelled with the count."""
    y = list(range(len(labels)))[::-1]
    pct = [100 * a / b for a, b in zip(k, n)]
    ax.barh(y, pct, height=0.62, color=color, edgecolor="white", linewidth=1)
    for yy, p, a, b in zip(y, pct, k, n):
        ax.text(p + 1.5, yy, label_fmt.format(k=a, n=b), va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 2. workflow gantt
rows = [  # (label, start, end, colour)
    ("Set-up: torch.fx answer key, generator v1 [D1]", date(2026, 10, 14), date(2026, 10, 17), BLUE),
    ("Prompting 1.5B to 31B [M1-M4]", date(2026, 10, 17), date(2026, 10, 22), BLUE),
    ("LoRA fine-tuning, checks on real code [M5]", date(2026, 10, 22), date(2026, 10, 26), BLUE),
    ("Realistic generator, v2 [D2]", date(2026, 10, 28), date(2026, 10, 31), BLUE),
    ("Two-step: list nodes, then wire [M6]", date(2026, 11, 1), date(2026, 11, 10), BLUE),
    ("Composition, edits, freeze, holdout [M7]", date(2026, 11, 10), date(2026, 11, 15), BLUE),
    ("Hierarchical schema, generator v3 + timm [M8]", date(2026, 11, 16), date(2026, 11, 21), ORANGE),
    ("14B, per-instance trees, lighter schemas [M9-M11]", date(2026, 11, 21), date(2026, 11, 23), ORANGE),
]
fig, ax = plt.subplots(figsize=(7.0, 0.86))
for i, (lab, s, e, c) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, (e - s).days, left=mdates.date2num(s), height=0.6, color=c, edgecolor="white", linewidth=1)
    ax.text(mdates.date2num(e) + 0.4, y, lab, va="center", fontsize=5.4, color=INK)
for d, lab, ha in [(date(2026, 10, 26), "intensive assessment ", "right"),
                   (date(2026, 11, 15), "holdout scored: 0/17 ", "right"),
                   (date(2026, 11, 19), " switch to hierarchical", "left")]:
    ax.axvline(mdates.date2num(d), color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)))
    ax.text(mdates.date2num(d), len(rows) - 0.3, lab, ha=ha, va="bottom", fontsize=5.8, color=INK2)
ax.set_xlim(mdates.date2num(date(2026, 10, 13)), mdates.date2num(date(2026, 12, 9)))
ax.set_ylim(-0.6, len(rows) + 0.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=BLUE, edgecolor="none", label="flat fx-graph schema"), Patch(facecolor=ORANGE, edgecolor="none", label="hierarchical schema")],
          loc="upper right", bbox_to_anchor=(1.0, 0.86), frameon=False, ncol=1)
save(fig, "fig2_gantt")

# ---------------------------------------------------------------- 2b. poster gantt (large, tags only)
POSTER = OUT.parents[1] / "poster" / "figures"
POSTER.mkdir(parents=True, exist_ok=True)
prow = [  # (phase, tag, start, end, colour)
    ("Set-up, answer key", "D1", date(2026, 10, 14), date(2026, 10, 17), BLUE),
    ("Prompting", "M1-M4", date(2026, 10, 17), date(2026, 10, 22), BLUE),
    ("Fine-tuning", "M5", date(2026, 10, 22), date(2026, 10, 26), BLUE),
    ("Generator v2", "D2", date(2026, 10, 28), date(2026, 10, 31), BLUE),
    ("Two-step", "M6", date(2026, 11, 1), date(2026, 11, 10), BLUE),
    ("Composition, holdout", "M7, D4", date(2026, 11, 10), date(2026, 11, 15), BLUE),
    ("Hierarchical format", "M8, D6-D8", date(2026, 11, 16), date(2026, 11, 21), ORANGE),
    ("14B, lighter formats", "M9-M11", date(2026, 11, 21), date(2026, 11, 23), ORANGE),
]
with plt.rc_context({"font.size": 15, "xtick.labelsize": 14, "ytick.labelsize": 15, "legend.fontsize": 15}):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for i, (phase, tag, s, e, c) in enumerate(prow):
        y = len(prow) - 1 - i
        ax.barh(y, (e - s).days, left=mdates.date2num(s), height=0.66, color=c, edgecolor="white", linewidth=2)
        ax.text(mdates.date2num(e) + 0.5, y, f"[{tag}]", va="center", fontsize=15, fontweight="bold", color=INK)
    for d, lab, ha in [(date(2026, 10, 26), "intensive assessment ", "right"),
                       (date(2026, 11, 15), "holdout scored ", "right"),
                       (date(2026, 11, 19), " format switch", "left")]:
        ax.axvline(mdates.date2num(d), color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
        ax.text(mdates.date2num(d), len(prow) - 0.35, lab, ha=ha, va="bottom", fontsize=14, color=INK2)
    ax.set_xlim(mdates.date2num(date(2026, 10, 13)), mdates.date2num(date(2026, 12, 3)))
    ax.set_ylim(-0.6, len(prow) + 0.5)
    ax.set_yticks([len(prow) - 1 - i for i in range(len(prow))], [r[0] for r in prow])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=BLUE, edgecolor="none", label="flat fx-graph format"),
                       Patch(facecolor=ORANGE, edgecolor="none", label="hierarchical format")],
              loc="lower left", frameon=False, ncol=1)
    fig.savefig(POSTER / "gantt_poster.pdf")
    fig.savefig(POSTER / "gantt_poster.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- 3. flat schema results
# one column wide: the size-scaling numbers of R1 are in the text
fig, b = plt.subplots(figsize=(3.35, 1.5))
systems = ["Fine-tuned 1.5B [M5]", "14B, prompted [M1]", "+ realistic data [D2]",
           "+ generator v2 [D2]", "Two-step [M6]", "Final: lister + wirer [M7]", "Lister + 14B wirer [M7]"]
dev = [0, 4, 7, 7, 9, 11, 13]
hold = {0: 0, 3: 0, 5: 0}          # systems scored on the 17-sample holdout
y = list(range(len(systems)))[::-1]
b.barh([v + 0.17 for v in y], [100 * d / 21 for d in dev], height=0.34, color=BLUE, edgecolor="white",
       linewidth=0.8, label="dev: 21 real blocks [D3]")
for v, d in zip(y, dev):
    b.text(100 * d / 21 + 1.5, v + 0.17, f"{d}/21", va="center", fontsize=6.2, color=INK2)
for i, h in hold.items():
    b.barh(y[i] - 0.17, max(100 * h / 17, 0.6), height=0.34, color=ORANGE, edgecolor="white", linewidth=0.8,
           label="holdout: 17 unseen [D4]" if i == 0 else None)
    b.text(2, y[i] - 0.17, f"{h}/17", va="center", fontsize=6.2, color=INK2)
b.set_yticks(y, systems)
b.set_xlim(0, 100)
b.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
b.tick_params(axis="y", length=0)
b.xaxis.grid(True, color=GRID, linewidth=0.6)
b.set_axisbelow(True)
legend(b, [("dev: 21 real blocks [D3]", BLUE), ("holdout: 17 unseen [D4]", ORANGE)])
save(fig, "fig3_flat")

# ---------------------------------------------------------------- 4. hierarchical results
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 0.9), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.7})
sets = ["Generated test blocks [D8]", "Generated test networks [D8]", "Real blocks [D7]", "Real networks [D7]"]
own = [(21, 24), (24, 51), (9, 21), (0, 30)]
oracle = {2: (16, 21), 3: (10, 30)}
y = list(range(len(sets)))[::-1]
a.barh([v + 0.17 for v in y], [100 * k / n for k, n in own], height=0.34, color=BLUE, edgecolor="white",
       linewidth=0.8, label="model writes tree and wiring")
for v, (k, n) in zip(y, own):
    a.text(100 * k / n + 1.5, v + 0.17, f"{k}/{n}", va="center", fontsize=6.2, color=INK2)
for i, (k, n) in oracle.items():
    a.barh(y[i] - 0.17, 100 * k / n, height=0.34, color=AQUA, edgecolor="white", linewidth=0.8,
           label="true tree given, model wires" if i == 2 else None)
    a.text(100 * k / n + 1.5, y[i] - 0.17, f"{k}/{n}", va="center", fontsize=6.2, color=INK2)
a.set_yticks(y, sets)
a.set_xlim(0, 112)
a.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
a.tick_params(axis="y", length=0)
a.xaxis.grid(True, color=GRID, linewidth=0.6)
a.set_axisbelow(True)
legend(a, [("model writes tree and wiring", BLUE), ("true tree given, model wires", AQUA)])
a.set_title("(a) Hierarchical drawing completely right [M8]", loc="left")

levels = ["S3: full tree + wiring", "S2: no layer settings", "S1: skeleton + class graphs", "S0: class graphs only"]
dev_net = [0, 3, 5, 9]
gen_net = [24, 28, 30, 35]
y = list(range(len(levels)))[::-1]
b.barh([v + 0.17 for v in y], [100 * k / 30 for k in dev_net], height=0.34, color=BLUE, edgecolor="white",
       linewidth=0.8, label="real networks (30) [D7]")
b.barh([v - 0.17 for v in y], [100 * k / 51 for k in gen_net], height=0.34, color="#9cc3ef", edgecolor="white",
       linewidth=0.8, label="generated networks (51) [D8]")
for v, k in zip(y, dev_net):
    b.text(100 * k / 30 + 1.5, v + 0.17, f"{k}/30", va="center", fontsize=6.2, color=INK2)
for v, k in zip(y, gen_net):
    b.text(100 * k / 51 + 1.5, v - 0.17, f"{k}/51", va="center", fontsize=6.2, color=INK2)
b.set_yticks(y, levels)
b.set_xlim(0, 112)
b.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
b.tick_params(axis="y", length=0)
b.xaxis.grid(True, color=GRID, linewidth=0.6)
b.set_axisbelow(True)
legend(b, [("real networks (30) [D7]", BLUE), ("generated networks (51) [D8]", "#9cc3ef")])
b.set_title("(b) Asking for less: same replies, lighter schemas [M11]", loc="left")
save(fig, "fig4_hier")

# ---------------------------------------------------------------- 4b. poster: the two hierarchical panels apart
def pair_bars(ax, labels, top, bottom, top_c, bot_c, fs):
    """Two bars per row (top/bottom), each (k, n) or None, labelled k/n."""
    y = list(range(len(labels)))[::-1]
    for off, vals, c in [(0.2, top, top_c), (-0.2, bottom, bot_c)]:
        for v, kn in zip(y, vals):
            if kn is None:
                continue
            k, n = kn
            ax.barh(v + off, max(100 * k / n, 0.8), height=0.4, color=c, edgecolor="white", linewidth=1.5)
            ax.text(100 * k / n + 1.5, v + off, f"{k}/{n}", va="center", fontsize=fs - 1, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


with plt.rc_context({"font.size": 14, "xtick.labelsize": 13, "ytick.labelsize": 14, "legend.fontsize": 13}):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    pair_bars(ax, ["Generated blocks [D8]", "Generated networks [D8]", "Real blocks [D7]", "Real networks [D7]"],
              [(21, 24), (24, 51), (9, 21), (0, 30)], [None, None, (16, 21), (10, 30)], BLUE, AQUA, 14)
    ax.legend(handles=[Patch(facecolor=BLUE, edgecolor="none", label="model writes tree and wiring"),
                       Patch(facecolor=AQUA, edgecolor="none", label="true tree given, model wires")],
              loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, frameon=False, handlelength=1.2,
              columnspacing=1.0)
    fig.savefig(POSTER / "fig5a_hier.pdf")
    fig.savefig(POSTER / "fig5a_hier.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    pair_bars(ax, ["S3: full tree + wiring", "S2: no layer settings", "S1: skeleton + class graphs",
                   "S0: class graphs only"],
              [(0, 30), (3, 30), (5, 30), (9, 30)], [(24, 51), (28, 51), (30, 51), (35, 51)], BLUE, "#9cc3ef", 14)
    ax.legend(handles=[Patch(facecolor=BLUE, edgecolor="none", label="real networks [D7]"),
                       Patch(facecolor="#9cc3ef", edgecolor="none", label="generated networks [D8]")],
              loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=2, frameon=False, handlelength=1.2,
              columnspacing=1.0)
    fig.savefig(POSTER / "fig5b_lighter.pdf")
    fig.savefig(POSTER / "fig5b_lighter.png", dpi=150)
    plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))

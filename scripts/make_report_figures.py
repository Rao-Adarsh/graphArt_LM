"""Figures and tables for the final report (report/figures/*.pdf, report/tables/*.tex).

    python scripts/make_report_figures.py

Every number is computed here from results/*.jsonl with the project's own scorer,
so a figure can never drift from the data. Colours follow the validated reference
palette (blue, orange, aqua, yellow in that fixed order); aqua and yellow sit below
3:1 contrast on white, so every chart has a table with the same numbers beside it.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from graphart.render import compare_figure
from pipeline.score import load_truths, score_reply, summarise
from scripts.analyze_rq6 import GROUPS as RQ6_GROUPS, included_dead_layer
from scripts.classify_errors import classify, parse

RIGHT = {"exact", "notation", "spelling", "commutative"}      # the `right graph` categories

FIG, TAB = ROOT / "report" / "figures", ROOT / "report" / "tables"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4, fixed order
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
TIERS = [1, 2, 3, 4]
TIER_NAMES = ["T1 flat", "T2 loops", "T3 branches", "T4 blocks"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})


def _save(fig, name: str) -> None:
    """PDF for the report; a high-resolution PNG of the same figure for the slides."""
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=260)


def scored(run: str, mode: str = "full") -> list[dict]:
    truths = load_truths("test")
    rows = [json.loads(l) for l in (ROOT / "results" / f"{run}.jsonl").open(encoding="utf-8")]
    return [{**r, **score_reply(r["reply"], truths[r["id"]], mode=mode)} for r in rows if r["id"] in truths]


def data(split: str) -> dict:
    return {r["id"]: r for r in map(json.loads, (ROOT / "data" / "generated" / f"{split}.jsonl").open(encoding="utf-8"))}


def categories(run: str, split: str = "test") -> list[tuple[dict, str]]:
    """(data row, error category) per reply, by scripts/classify_errors.py's fixed rules.
    Edit-set replies are also checked against the unedited program's graph."""
    rows, truths = data(split), load_truths(split)
    originals = load_truths("test") if split == "edits" else {}
    out = []
    for rec in map(json.loads, (ROOT / "results" / f"{run}.jsonl").open(encoding="utf-8")):
        row = rows[rec["id"]]
        original = originals[row["base_id"]] if split == "edits" and row["kind"] == "meaningful" else None
        out.append((row, classify(rec["reply"], truths[rec["id"]], row["code"], original)[0]))
    return out


def right_graph(run: str) -> float:
    cats = categories(run)
    return sum(c in RIGHT for _, c in cats) / len(cats)


def by_tier(rows: list[dict], key: str = "exact_topo") -> list[float]:
    return [summarise([r for r in rows if r["tier"] == t])[key] for t in TIERS]


def _mark_zeros(ax, positions, values, width, color, horizontal: bool) -> None:
    """A zero-height bar is invisible, which reads as missing data. Mark each true zero
    with a short line on the baseline and a small 0."""
    for p, v in zip(positions, values):
        if v == 0:
            if horizontal:
                ax.plot([0, 0], [p - width * 0.4, p + width * 0.4], color=color, linewidth=2, zorder=4)
                ax.text(0.012, p, "0", va="center", ha="left", fontsize=7, color=INK2)
            else:
                ax.plot([p - width * 0.4, p + width * 0.4], [0, 0], color=color, linewidth=2, zorder=4)
                ax.text(p, 0.015, "0", ha="center", va="bottom", fontsize=7, color=INK2)


def grouped_bars(ax, groups: list[str], series: dict[str, list[float]], ylabel: str) -> None:
    n = len(series)
    width = 0.8 / n
    for i, (name, values) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        # 2px surface gap between neighbouring bars; thin, flat-topped marks.
        ax.bar(xs, values, width, color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        _mark_zeros(ax, xs, values, width, SERIES[i], horizontal=False)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="both", length=0)
    ax.legend(frameon=False, ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5,
              labelcolor=INK)


def table(name: str, header: list[str], rows: list[list], caption: str) -> None:
    lines = ["\\begin{table}[!h]", "\\centering", "\\small",
             "\\begin{tabular}{" + "l" + "c" * (len(header) - 1) + "}", "\\hline",
             " & ".join(header) + " \\\\", "\\hline"]
    for r in rows:
        lines.append(" & ".join(f"{c:.2f}" if isinstance(c, float) else str(c) for c in r) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", f"\\caption{{{caption}}}\\label{{tab:{name}}}", "\\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fig_scale() -> None:
    runs = {"Qwen 1.5B": "ollama__qwen2.5-coder-1.5b__zero_shot", "Qwen 7B": "ollama__qwen2.5-coder-7b__zero_shot",
            "Qwen 14B": "ollama__qwen2.5-coder-14b__zero_shot",
            "Gemma 4 31B": "openrouter__google-gemma-4-31b-it-free__zero_shot"}
    data = {k: scored(v) for k, v in runs.items()}
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    grouped_bars(ax, TIER_NAMES, {k: by_tier(v) for k, v in data.items()}, "topology correct")
    fig.tight_layout()
    _save(fig, "scale_by_tier")
    plt.close(fig)
    table("scale", ["Model (zero-shot)", "T1", "T2", "T3", "T4", "topo", "exact", "right graph", "reach F1",
                    "parsed"],
          [[k, *by_tier(v), summarise(v)["exact_topo"], summarise(v)["exact"], right_graph(runs[k]),
            summarise(v)["reach_f1"], summarise(v)["parsed"]] for k, v in data.items()],
          "Zero-shot accuracy by model size on the 40 stratified test programs (10 per tier). "
          "Tier columns and \\emph{topo} count graphs with the right topology; \\emph{exact} is strict; "
          "\\emph{right graph} also forgives spelling that does not change the computation; \\emph{reach F1} "
          "measures connectivity preservation (shared ancestor--descendant pairs).")


def fig_prompts() -> None:
    runs = {"zero-shot": "zero_shot", "few-shot": "few_shot", "CoT": "cot", "CoS (ours)": "cos"}
    data = {k: scored(f"ollama__qwen2.5-coder-14b__{v}") for k, v in runs.items()}
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    grouped_bars(ax, TIER_NAMES, {k: by_tier(v) for k, v in data.items()}, "topology correct")
    fig.tight_layout()
    _save(fig, "prompts_by_tier")
    plt.close(fig)
    decomp = {"nodes only": scored("ollama__qwen2.5-coder-14b__nodes_only", "nodes"),
              "edges only": scored("ollama__qwen2.5-coder-14b__edges_only", "edges"),
              "full task": data["zero-shot"]}
    right = {k: right_graph(f"ollama__qwen2.5-coder-14b__{v}") for k, v in runs.items()}
    right["full task"] = right["zero-shot"]
    table("prompts", ["Prompt (Qwen 14B)", "T1", "T2", "T3", "T4", "topo", "exact", "right graph", "parsed"],
          [[k, *by_tier(v), summarise(v)["exact_topo"], summarise(v)["exact"], right.get(k, "--"),
            summarise(v)["parsed"]] for k, v in {**data, **decomp}.items()],
          "Prompting strategies and the read/wire decomposition on Qwen2.5-Coder 14B, same 40 programs. "
          "\\emph{right graph} applies to whole-graph replies only.")


def fig_finetune() -> None:
    runs = {"Qwen 1.5B, untuned": "hf__Qwen-Qwen2.5-Coder-1.5B-Instruct__zero_shot",
            "Qwen 14B, few-shot": "ollama__qwen2.5-coder-14b__few_shot",
            "Gemma 4 31B, zero-shot": "openrouter__google-gemma-4-31b-it-free__zero_shot",
            "Qwen 1.5B + LoRA (ours)": "hf__qwen2.5-coder-1.5b-graphart-lora__zero_shot"}
    data = {k: {**summarise(scored(v)), "right": right_graph(v)} for k, v in runs.items()}
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    names = list(data)
    metrics = [("exact", "exact"), ("exact_topo", "topology"), ("right", "right graph")]
    h = 0.8 / len(metrics)
    for i, (metric, label) in enumerate(metrics):
        ys = [y + ((len(metrics) - 1) / 2 - i) * h for y in range(len(names))]   # first legend item on top
        values = [data[n][metric] for n in names]
        ax.barh(ys, values, h, color=SERIES[i], label=label, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        _mark_zeros(ax, ys, values, h, SERIES[i], horizontal=True)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, 1.05)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="both", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("share of the 40 test programs")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    _save(fig, "finetune_vs_prompting")
    plt.close(fig)
    table("finetune", ["Model", "exact", "topo", "right graph", "reach F1", "parsed"],
          [[k, v["exact"], v["exact_topo"], v["right"], v["reach_f1"], v["parsed"]] for k, v in data.items()],
          "Fine-tuning against prompting, same 40 test programs and the same zero-shot prompt for the "
          "two 1.5B rows. All on our generator's programs; Table~\\ref{tab:closing_rq4} has real code.")


def fig_workflow() -> None:
    steps = [("Problem", "code $\\rightarrow$ graph,\nno execution"),
             ("Domain", "Qiskit circuits\n$\\rightarrow$ PyTorch (TA)"),
             ("Data", "tiered generator,\ntorch.fx truth"),
             ("Harness", "validator, metrics,\nscorer"),
             ("Phase 1", "prompting:\n4 models\n6 prompts"),
             ("Phase 2", "LoRA fine-tune\n1.5B on Kaggle"),
             ("Closing", "fresh, edits,\ntorchvision,\nheld-out")]
    fig, ax = plt.subplots(figsize=(7.2, 1.7))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (title, body) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.12), 0.84, 0.76, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor="#f2f1ee", edgecolor=SERIES[0] if i >= 4 else GRID, linewidth=1.0))
        ax.text(i + 0.48, 0.70, title, ha="center", va="center", fontsize=8.5, color=INK, weight="bold")
        ax.text(i + 0.48, 0.36, body, ha="center", va="center", fontsize=6.4, color=INK2, linespacing=1.2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.06, 0.5), xytext=(i + 0.9, 0.5),
                        arrowprops=dict(arrowstyle="->", color=INK2, linewidth=0.8))
    fig.tight_layout(pad=0.2)
    _save(fig, "workflow")
    plt.close(fig)


def fig_decomposition() -> None:
    """RQ2 at a glance: reading alone, wiring alone, both at once (Qwen 14B, topology)."""
    parts = {"read the code\n(nodes only)": scored("ollama__qwen2.5-coder-14b__nodes_only", "nodes"),
             "wire given nodes\n(edges only)": scored("ollama__qwen2.5-coder-14b__edges_only", "edges"),
             "both at once\n(full task)": scored("ollama__qwen2.5-coder-14b__zero_shot")}
    values = [summarise(v)["exact_topo"] for v in parts.values()]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.bar(range(3), values, 0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for x, v in enumerate(values):
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=INK, weight="bold")
    ax.set_xticks(range(3), list(parts))
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.tick_params(axis="both", length=0)
    fig.tight_layout()
    _save(fig, "decomposition")


def table_data() -> None:
    rows = {s: [json.loads(l) for l in (ROOT / "data" / "generated" / f"{s}.jsonl").open(encoding="utf-8")]
            for s in ("train", "val", "test")}
    names = {1: "1 flat", 2: "2 loops", 3: "3 branches", 4: "4 blocks"}
    out = []
    for t in TIERS:
        tier = [r for s in rows.values() for r in s if r["tier"] == t]
        nodes = [r["num_nodes"] for r in tier]
        merges = sum(r["num_merges"] for r in tier) / len(tier)
        out.append([names[t], len(tier), *(sum(r["tier"] == t for r in rows[s]) for s in ("train", "val", "test")),
                    f"{min(nodes)} / {sum(nodes) / len(nodes):.1f} / {max(nodes)}", merges])
    total = [r for s in rows.values() for r in s]
    out.append(["all", len(total), *(len(rows[s]) for s in ("train", "val", "test")), "", ""])
    table("data", ["Tier", "programs", "train", "val", "test", "nodes min / mean / max", "merges / graph"], out,
          "The synthetic data set: four tiers by how much the code hides the graph. A merge is a node "
          "with more than one incoming value (residuals, concatenations).")


# --------------------------------------------------------------------------
# closing experiments (RQ4-RQ6): the fine-tuned 1.5B against the prompted 14B
# --------------------------------------------------------------------------
LORA, Q14 = "hf__qwen2.5-coder-1.5b-graphart-lora__zero_shot", "ollama__qwen2.5-coder-14b__zero_shot"
HELDOUT = "hf__qwen2.5-coder-1.5b-graphart-lora-no-dead-branch__zero_shot"
CLOSING_MODELS = {"Qwen 14B, prompted": Q14, "Qwen 1.5B + LoRA": LORA}

# The in-distribution rule for torchvision, fixed before the tuned model's results were
# opened: none of the constructs that the 1,280 training programs never contain.
UNSEEN = [("own block class", lambda r: len(re.findall(r"^class \w+\(nn\.Module\)", r["code"], re.M)) > 1),
          ("grouped conv", lambda r: bool(re.search(r",g=\d", r["target"]))),
          ("tuple kernel", lambda r: bool(re.search(r"k=\[", r["target"]))),
          ("un-shortened F. function", lambda r: bool(re.search(r"call_function F\.", r["target"]))),
          ("assert", lambda r: "assert" in r["code"])]


def in_distribution(row: dict) -> bool:
    return not any(test(row) for _, test in UNSEEN)


def is_chain(graph: dict) -> bool:
    """No node with two inputs or two consumers: a straight line."""
    fan_in = [e["dst"] for e in graph["edges"]]
    fan_out = [e["src"] for e in graph["edges"]]
    return all(fan_in.count(n) <= 1 for n in set(fan_in)) and all(fan_out.count(n) <= 1 for n in set(fan_out))


def rq4() -> dict:
    out = {}
    for name, run in CLOSING_MODELS.items():
        cats = categories(f"{run}__torchvision", "torchvision")
        replies = {r["id"]: r["reply"] for r in
                   map(json.loads, (ROOT / "results" / f"{run}__torchvision.jsonl").open(encoding="utf-8"))}
        parsed = [g for g in (parse(replies[row["id"]])[0] for row, _ in cats) if g]
        summary = summarise([{**row, **score_reply(replies[row["id"]], row["graph"])} for row, _ in cats])
        ind = [c for row, c in cats if in_distribution(row)]
        uns = [c for row, c in cats if not in_distribution(row)]
        out[name] = {"parsed": len(parsed), "exact": round(summary["exact"] * len(cats)),
                     "topo": round(summary["exact_topo"] * len(cats)), "right": sum(c in RIGHT for _, c in cats),
                     "node_f1": summary["node_f1"], "edge_f1": summary["edge_f1"], "reach_f1": summary["reach_f1"],
                     "ind": (sum(c in RIGHT for c in ind), len(ind)), "uns": (sum(c in RIGHT for c in uns), len(uns)),
                     "chain": (sum(map(is_chain, parsed)), len(parsed))}
    return out


def rq5() -> dict:
    out = {}
    for name, run in CLOSING_MODELS.items():
        solved = {row["id"] for row, c in categories(run) if c in ("exact", "notation")}
        cats = [(row, c) for row, c in categories(f"{run}__edits", "edits") if row["base_id"] in solved]
        kind = lambda k: [c for row, c in cats if row["kind"] == k]
        edit = lambda e: [c for row, c in cats if row["edit"] == e]
        count = lambda xs, test: (sum(map(test, xs)), len(xs))
        right = lambda c: c in RIGHT
        out[name] = {"solved": len(solved), "neutral": count(kind("neutral"), right),
                     "meaningful": count(kind("meaningful"), right),
                     "original": count(kind("meaningful"), lambda c: c == "gave_original"),
                     "reroute": count(edit("reroute"), right), "swap_cat": count(edit("swap_cat"), right),
                     "drop_call": count(edit("drop_call"), right)}
    return out


def rq6() -> dict:
    rows = data("rq6")
    out = {}
    for name, run in {"full adapter": LORA, "dead branches held out": HELDOUT}.items():
        by = {g: [] for g in RQ6_GROUPS}
        dead = [0, 0]
        for rec in map(json.loads, (ROOT / "results" / f"{run}__rq6.jsonl").open(encoding="utf-8")):
            row = rows[rec["id"]]
            by[row["group"]].append(score_reply(rec["reply"], row["graph"])["exact_topo"])
            if row["group"] in ("dead_branch", "dead_flag"):
                dead[1] += 1
                dead[0] += included_dead_layer(rec["reply"], row["code"], row["graph"])
        out[name] = {**{g: (sum(v), len(v)) for g, v in by.items()}, "dead": tuple(dead)}
    return out


def _frac(pair) -> str:
    return f"{pair[0]} / {pair[1]}"


def fig_closing() -> None:
    a, b, c = rq4(), rq5(), rq6()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [3.3, 3.3, 4]})
    share = lambda p: p[0] / p[1] if p[1] else 0.0

    def panel(ax, groups, series, colors, title):
        n = len(series)
        width = 0.8 / n
        for i, (label, values) in enumerate(series.items()):
            xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
            ax.bar(xs, values, width, color=colors[i], label=label, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            _mark_zeros(ax, xs, values, width, colors[i], horizontal=False)
        ax.set_xticks(range(len(groups)), groups, fontsize=6.3)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.tick_params(axis="both", length=0)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=6.6, loc="upper center", ncol=1, labelcolor=INK)

    # colour follows the entity: the 14B blue, the full adapter orange, the held-out adapter aqua
    panel(axes[0], ["in-distr.:\nright", "unseen:\nright", "drawn as\na chain"],
          {k: [share(v["ind"]), share(v["uns"]), share(v["chain"])] for k, v in a.items()},
          SERIES[:2], "RQ4  real code")
    panel(axes[1], ["neutral:\nright", "meaningful:\nright", "meaningful:\ngave orig."],
          {k: [share(v["neutral"]), share(v["meaningful"]), share(v["original"])] for k, v in b.items()},
          SERIES[:2], "RQ5  controlled edits")
    panel(axes[2], ["control", "dead\nbranch", "dead flag\nadded", "flag set\nto True"],
          {k: [share(v[g]) for g in RQ6_GROUPS] for k, v in c.items()},
          [SERIES[1], SERIES[2]], "RQ6  held-out construct")
    axes[0].set_ylabel("share")
    fig.tight_layout()
    _save(fig, "closing")
    plt.close(fig)

    table("closing_rq4", ["torchvision (21)", "parsed", "exact", "topo", "right", "node F1", "edge F1",
                          "reach F1", "in-distr.", "unseen", "chain"],
          [[k, v["parsed"], v["exact"], v["topo"], v["right"], v["node_f1"], v["edge_f1"], v["reach_f1"],
            _frac(v["ind"]), _frac(v["uns"]), _frac(v["chain"])] for k, v in a.items()],
          "RQ4, real code: counts over 21 torchvision samples. \\emph{In-distr.} (5 samples) have none of the "
          "constructs absent from all 1,280 training programs (own block classes, grouped convolutions, tuple "
          "kernels, un-shortened \\texttt{F.} functions, \\texttt{assert}); the rule was fixed before the tuned "
          "model's results were opened. \\emph{Right} is the right-graph level. \\emph{Chain}: readable replies "
          "with no branch at all; 4 of the 21 true graphs are chains.")
    table("closing_rq5", ["edits", "solved", "neutral", "meaningful", "gave original", "reroute",
                          "swap cat", "drop call"],
          [[k, v["solved"], _frac(v["neutral"]), _frac(v["meaningful"]), _frac(v["original"]),
            _frac(v["reroute"]), _frac(v["swap_cat"]), _frac(v["drop_call"])] for k, v in b.items()],
          "RQ5, controlled edits of the test programs, counting only the programs each model got right "
          "before the edit (\\emph{solved}). Columns count right graphs, except \\emph{gave original}: "
          "meaningful edits answered with the graph of the unedited program.")
    table("closing_rq6", ["rq6 probe (84)", "control", "dead branch", "dead flag added", "flag True",
                          "dead layer in"],
          [[k, *(_frac(v[g]) for g in RQ6_GROUPS), _frac(v["dead"])] for k, v in c.items()],
          "RQ6, a held-out construct: the full adapter against one trained without any dead-branch program "
          "(1,112 examples, one epoch against two). Group columns count topology-correct graphs. All 168 "
          "flagged branches in the training data have the flag \\texttt{False}.")



# --------------------------------------------------------------------------
# qualitative: a reply beside its reference, every mismatch coloured
# --------------------------------------------------------------------------
QUALITATIVE = [
    # (figure name, run, split, sample, title)
    ("qual_14b_dead_branch", "ollama__qwen2.5-coder-14b__zero_shot", "test", "t3-00090",
     "t3-00090: reference (left) vs Qwen 14B zero-shot (right)"),
    ("qual_lora_chain", "hf__qwen2.5-coder-1.5b-graphart-lora__zero_shot__torchvision", "torchvision",
     "tv-squeezenet-fire", "SqueezeNet Fire block (torchvision): reference vs fine-tuned 1.5B"),
    ("qual_lora_reroute", "hf__qwen2.5-coder-1.5b-graphart-lora__zero_shot__edits", "edits",
     "t1-00343~reroute", "Edit: fc1(h) -> fc1(x): reference vs fine-tuned 1.5B"),
]
# the deck's qualitative slide uses the same renderer, at the same names as before
DECK_COPIES = {"qual_14b_dead_branch": "qual_t3-00090.png"}


def fig_qualitative() -> None:
    deck = ROOT / "report" / "deck_figures"
    for name, run, split, sample, title in QUALITATIVE:
        reply = next(r["reply"] for r in map(json.loads, (ROOT / "results" / f"{run}.jsonl").open(encoding="utf-8"))
                     if r["id"] == sample)
        pred, _ = parse(reply)
        fig = compare_figure(load_truths(split)[sample], pred, title=title)
        _save(fig, name)
        if name in DECK_COPIES:
            fig.savefig(deck / DECK_COPIES[name], dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for make in (fig_workflow, fig_scale, fig_prompts, fig_finetune, fig_decomposition, table_data, fig_closing, fig_qualitative):
        make()
        print("made", make.__name__)


if __name__ == "__main__":
    main()

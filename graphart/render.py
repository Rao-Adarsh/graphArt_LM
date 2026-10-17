"""Draw a forward-pass graph, and draw a prediction against the truth.

Two jobs:

* `side_by_side` -- source on the left, the graph it traces to on the right.
  This is the figure that explains the task.
* `compare` -- the reference graph and a model's graph next to each other, with
  everything that does not match coloured. This is deliverable 4, "enabling
  direct comparison between the reference graph and the model-generated graph",
  and it is the figure that explains a *result*.

Layout runs top to bottom rather than left to right. These graphs are mostly
long chains with occasional branches, and that is the shape neural networks are
normally drawn in -- a 25-node chain laid out horizontally is unreadable.
"""

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

from .validate import edge_label, node_label, to_networkx

# Colour says what kind of operation a node is, so the eye can find the layers.
OP_COLOURS = {
    "placeholder": ("#e8f5e9", "#2e7d32"),
    "call_module": ("#eef3fb", "#3b5b92"),
    "call_function": ("#fdf3e3", "#e3760e"),
    "call_method": ("#f3e9f7", "#7b4397"),
    "get_attr": ("#f0f0f0", "#666666"),
    "output": ("#e8f5e9", "#2e7d32"),
}
MISSING = ("#fdeaea", "#c62828")   # in the reference, absent from the prediction
SPURIOUS = ("#fff4e5", "#ef6c00")  # in the prediction, absent from the reference

BOX_W, BOX_H, X_GAP, Y_GAP = 2.4, 0.78, 2.9, 1.5


def _short_params(node: dict) -> str:
    p, cls = node["params"], node["cls"] or ""
    if cls == "Linear":
        return f"({p.get('in')}→{p.get('out')})"
    if cls.startswith(("Conv", "ConvTranspose")):
        return f"({p.get('in')}→{p.get('out')}, k{p.get('k')})"
    if cls.startswith("BatchNorm") or cls == "GroupNorm":
        return f"({p.get('features')})"
    if cls == "LayerNorm":
        shape = p.get("shape")
        return f"({shape[0]})" if isinstance(shape, list) and shape else ""
    if cls in ("MaxPool2d", "AvgPool2d"):
        return f"(k{p.get('k')})"
    return ""


def _lines(node: dict) -> tuple[str, str, str]:
    """Three lines of label: who it is, what it is, what comes out."""
    head = f"{node['id']}: {node['target']}"
    if node["op"] == "call_module":
        body = f"{node['cls']}{_short_params(node)}"
    elif node["op"] in ("placeholder", "output"):
        body = node["op"]
    else:
        body = node["op"].replace("call_", "")
    shape = node["shape"]
    tail = "" if shape is None else str(shape).replace(" ", "")
    return head, body, tail


def layout(graph: dict) -> dict:
    """Top-to-bottom by topological generation, spread within each generation."""
    g = to_networkx(graph)
    pos = {}
    for depth, generation in enumerate(nx.topological_generations(g)):
        row = sorted(generation)
        for i, nid in enumerate(row):
            pos[nid] = ((i - (len(row) - 1) / 2) * X_GAP, -depth * Y_GAP)
    return pos


def draw_graph(graph: dict, ax, *, bad_nodes: set | None = None,
               bad_edges: set | None = None, bad_style: tuple = MISSING,
               title: str = "") -> None:
    """`bad_style` is the colour pair for mismatches: red for something the
    reference has and the prediction lost, orange for something invented."""
    bad_nodes, bad_edges = bad_nodes or set(), bad_edges or set()
    pos = layout(graph)
    by_id = {n["id"]: n for n in graph["nodes"]}
    fallback = OP_COLOURS["get_attr"]

    for e in graph["edges"]:
        if e["src"] not in pos or e["dst"] not in pos:
            continue
        (x1, y1), (x2, y2) = pos[e["src"]], pos[e["dst"]]
        wrong = (e["src"], e["dst"]) in bad_edges
        colour = bad_style[1] if wrong else "#90a4ae"
        ax.annotate("", xy=(x2, y2 + BOX_H / 2), xytext=(x1, y1 - BOX_H / 2),
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                    lw=2.0 if wrong else 1.3,
                                    connectionstyle="arc3,rad=0.06"))
        # Only label the slots worth noticing. Almost every edge is arg0, so
        # labelling them all buries the merges that actually matter.
        if e["args"] != [0]:
            ax.text((x1 + x2) / 2 + 0.25, (y1 + y2) / 2,
                    " ".join(f"arg{a}" for a in e["args"]),
                    fontsize=7, color=colour, ha="left", va="center")

    for nid, (x, y) in pos.items():
        node = by_id[nid]
        fill, edge = bad_style if nid in bad_nodes else OP_COLOURS.get(node["op"], fallback)
        ax.add_patch(FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
                                    boxstyle="round,pad=0.03", fc=fill, ec=edge,
                                    lw=2.0 if nid in bad_nodes else 1.1))
        head, body, tail = _lines(node)
        ax.text(x, y + 0.20, head, ha="center", va="center", fontsize=7.5,
                weight="bold", color=edge)
        ax.text(x, y + 0.01, body, ha="center", va="center", fontsize=7, color="#263238")
        if tail:
            ax.text(x, y - 0.21, tail, ha="center", va="center", fontsize=6, color="#5a626e")

    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    ax.set_xlim(min(xs) - BOX_W, max(xs) + BOX_W)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=11)


def _unmatched(truth: dict, pred: dict) -> tuple[set, set, set, set]:
    """Which nodes and edges on each side have no counterpart on the other.

    Matching is by label multiset, the same basis `metrics` scores on, so the
    picture and the number always agree. Greedy, which is enough: labels are
    either present in the right quantity or they are not.
    """
    def pick(items, key, pool):
        left, ok = dict(pool), set()
        for it in items:
            k = key(it)
            if left.get(k, 0) > 0:
                left[k] -= 1
                ok.add(it["id"] if "id" in it else (it["src"], it["dst"]))
        return ok

    tl = Counter(node_label(n) for n in truth["nodes"])
    pl = Counter(node_label(n) for n in pred["nodes"])
    shared_nodes = tl & pl
    t_ok = pick(truth["nodes"], node_label, shared_nodes)
    p_ok = pick(pred["nodes"], node_label, shared_nodes)

    def ekey(graph):
        labels = {n["id"]: node_label(n) for n in graph["nodes"]}
        return lambda e: (labels.get(e["src"]), labels.get(e["dst"]), edge_label(e))

    te, pe = ekey(truth), ekey(pred)
    shared_edges = Counter(te(e) for e in truth["edges"]) & Counter(pe(e) for e in pred["edges"])
    t_edge_ok = pick(truth["edges"], te, shared_edges)
    p_edge_ok = pick(pred["edges"], pe, shared_edges)

    return ({n["id"] for n in truth["nodes"]} - t_ok,
            {n["id"] for n in pred["nodes"]} - p_ok,
            {(e["src"], e["dst"]) for e in truth["edges"]} - t_edge_ok,
            {(e["src"], e["dst"]) for e in pred["edges"]} - p_edge_ok)


def side_by_side(code: str, graph: dict, title: str = "", path: str | None = None):
    """Source on the left, the graph it traces to on the right."""
    depth = len(list(nx.topological_generations(to_networkx(graph))))
    fig = plt.figure(figsize=(13, max(5.0, 0.62 * depth + 1.2)))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.5])
    ax_code, ax_graph = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])

    body = code.split("import torch.nn.functional", 1)[-1]
    body = code[code.find("class "):] if "class " in code else body
    ax_code.text(0, 1, body.strip(), family="monospace", fontsize=8, va="top",
                 transform=ax_code.transAxes)
    ax_code.axis("off")
    ax_code.set_title("1. source code (what the model sees)", loc="left", fontsize=11)

    draw_graph(graph, ax_graph, title="2. forward-pass graph (what it must produce)")
    if title:
        fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def compare_figure(truth: dict, pred: dict, title: str = "", path: str | None = None):
    """Reference and prediction side by side, with every mismatch coloured.

    Red on the left is something the reference has that the prediction missed;
    orange on the right is something the prediction invented.
    """
    bad_t, bad_p, bad_te, bad_pe = _unmatched(truth, pred)
    depth = max(len(list(nx.topological_generations(to_networkx(g))))
                for g in (truth, pred) if g["nodes"]) if pred["nodes"] else 4
    fig = plt.figure(figsize=(13, max(5.0, 0.62 * depth + 1.4)))
    ax_t, ax_p = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    draw_graph(truth, ax_t, bad_nodes=bad_t, bad_edges=bad_te, bad_style=MISSING,
               title=f"reference   (red: {len(bad_t)} node(s), {len(bad_te)} edge(s) missed)")
    draw_graph(pred, ax_p, bad_nodes=bad_p, bad_edges=bad_pe, bad_style=SPURIOUS,
               title=f"model output   (orange: {len(bad_p)} node(s), "
                     f"{len(bad_pe)} edge(s) invented)")
    for ax in (ax_t, ax_p):
        ax.set_title(ax.get_title(), loc="left", fontsize=11)
    if title:
        fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig

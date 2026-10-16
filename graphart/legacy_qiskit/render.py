"""Draw  code | circuit | DAG  side by side (for sanity checks and slides)."""

from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

from .validate import to_networkx


def dag_layout(graph: dict) -> dict:
    """x = ASAP layer (topological generation), y = mean qubit index."""
    g = to_networkx(graph)
    pos, used = {}, defaultdict(list)
    for x, generation in enumerate(nx.topological_generations(g)):
        for nid in sorted(generation):
            n = g.nodes[nid]
            wires = n["qubits"] or [0]
            y = -sum(wires) / len(wires)
            while any(abs(y - u) < 0.45 for u in used[x]):  # nudge overlapping nodes
                y -= 0.5
            used[x].append(y)
            pos[nid] = (x * 1.6, y)
    return pos


def draw_dag(graph: dict, ax) -> None:
    g, pos = to_networkx(graph), dag_layout(graph)
    cmap = plt.get_cmap("tab10")
    wire_color = lambda w: "0.5" if w[0] == "c" else cmap(int(w[1:]) % 10)

    for s, d, data in g.edges(data=True):
        (x1, y1), (x2, y2) = pos[s], pos[d]
        ax.annotate("", xy=(x2 - 0.45, y2), xytext=(x1 + 0.45, y1),
                    arrowprops=dict(arrowstyle="-|>", color=wire_color(data["wires"][0]), lw=1.4,
                                    connectionstyle="arc3,rad=0.08"))
        ax.text(x1 + 0.62 * (x2 - x1), (y1 + y2) / 2 + 0.16, ",".join(data["wires"]),
                fontsize=8, ha="center", color="0.35")
    for nid, (x, y) in pos.items():
        n = g.nodes[nid]
        ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.2), 0.84, 0.4, boxstyle="round,pad=0.02",
                                    fc="#eef3fb", ec="#3b5b92", lw=1))
        wires = ",".join([f"q{q}" for q in n["qubits"]] + [f"c{c}" for c in n["clbits"]])
        ax.text(x, y + 0.06, f"{nid}: {n['gate']}", ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x, y - 0.11, wires, ha="center", va="center", fontsize=7.5)
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")


def side_by_side(code: str, circuit, graph: dict, title: str = "", path: str | None = None):
    fig = plt.figure(figsize=(18, 4.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.6])
    ax_code, ax_circ, ax_dag = (fig.add_subplot(grid[0, i]) for i in range(3))

    body = code.split("QuantumCircuit\n", 1)[-1].strip()  # drop the import header
    ax_code.text(0, 1, body, family="monospace", fontsize=10.5, va="top", transform=ax_code.transAxes)
    ax_code.axis("off")
    ax_code.set_title("1. source code (model input)", loc="left")

    circuit.draw("mpl", ax=ax_circ, fold=-1)
    ax_circ.set_title("2. the circuit it builds", loc="left")

    draw_dag(graph, ax_dag)
    ax_dag.set_title("3. circuit DAG (model output)", loc="left")

    if title:
        fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig

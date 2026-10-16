"""Score a predicted graph against the ground truth.

- valid      : passes validate.check()
- exact      : labelled-graph isomorphism (same gates on same wires, same edges)
- node_f1    : overlap of the multisets of node labels (gate, qubits, clbits, params)
- edge_f1    : overlap of the multisets of (src label, dst label, wires)

Node/edge F1 give partial credit; `exact` is the headline number. Comparing
graphs (not gate lists) matters: two gates on different qubits can be written
in either order in code and still produce the same DAG.
"""

from collections import Counter

import networkx as nx

from .domain import check, node_label, to_networkx


def _f1(pred: Counter, true: Counter) -> dict:
    if not pred and not true:  # e.g. a circuit with no edges, predicted correctly
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    hit = sum((pred & true).values())
    p = hit / max(sum(pred.values()), 1)
    r = hit / max(sum(true.values()), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _edge_labels(graph: dict) -> Counter:
    labels = {n["id"]: node_label(n) for n in graph["nodes"]}
    return Counter((labels.get(e["src"]), labels.get(e["dst"]), tuple(sorted(e["wires"])))
                   for e in graph["edges"])


def compare(pred: dict, truth: dict) -> dict:
    errors = check(pred)
    nodes = _f1(Counter(node_label(n) for n in pred["nodes"]), Counter(node_label(n) for n in truth["nodes"]))
    edges = _f1(_edge_labels(pred), _edge_labels(truth))
    exact = (not errors and nodes["f1"] == 1.0 and edges["f1"] == 1.0 and nx.is_isomorphic(
        to_networkx(pred), to_networkx(truth),
        node_match=lambda a, b: a["label"] == b["label"],
        edge_match=lambda a, b: a["wires"] == b["wires"]))
    return {"valid": not errors, "errors": errors, "exact": exact,
            "node_f1": nodes["f1"], "edge_f1": edges["f1"]}

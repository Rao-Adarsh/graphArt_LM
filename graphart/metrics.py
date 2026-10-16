"""Score a predicted graph against the ground truth.

- valid      : passes the domain's check() -- well formed, without reference
               to the right answer
- valid_topology : the same, ignoring shape consistency. This is what gates
               `exact`, so one bad dimension cannot sink a correct topology.
- exact      : labelled-graph isomorphism. The headline number.
- node_f1    : overlap of the multisets of node labels
- edge_f1    : overlap of the multisets of (source label, target label, edge label)
- shape_acc  : of the nodes whose label was right, how many also had the right
               shape

Node/edge F1 give partial credit. Comparing graphs rather than sequences is
the point: operations that do not depend on each other can be written in
either order and still describe the same computation.

`shape_acc` is reported separately rather than folded into the node label on
purpose. Shape inference is a different skill from topology, and if a shape
were part of the label then one miscomputed spatial dimension would drop an
otherwise perfect graph to exact=0 and the topology result would be lost.
"""

from collections import Counter

import networkx as nx

from .domain import check, edge_label, node_label, to_networkx


def _f1(pred: Counter, true: Counter) -> dict:
    if not pred and not true:  # e.g. a graph with no edges, predicted correctly
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    hit = sum((pred & true).values())
    p = hit / max(sum(pred.values()), 1)
    r = hit / max(sum(true.values()), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _edge_labels(graph: dict) -> Counter:
    labels = {n["id"]: node_label(n) for n in graph["nodes"]}
    return Counter((labels.get(e["src"]), labels.get(e["dst"]), edge_label(e))
                   for e in graph["edges"])


def _shape_key(node: dict):
    """A hashable form of a node's shape.

    Multi-output nodes (chunk, split) carry a list of shapes, so one level of
    nesting has to be flattened too or the Counter below cannot hold it.
    """
    shape = node.get("shape")
    if isinstance(shape, list):
        return tuple(tuple(s) if isinstance(s, list) else s for s in shape)
    return shape


def _shape_accuracy(pred: dict, truth: dict) -> float | None:
    """Among correctly-labelled nodes, the fraction whose shape also matched.

    None when the truth carries no shapes at all, so an absent-shape dataset
    reports nothing rather than a misleading 1.00.
    """
    if not any(n.get("shape") is not None for n in truth["nodes"]):
        return None
    pred_labels = Counter(node_label(n) for n in pred["nodes"])
    true_labels = Counter(node_label(n) for n in truth["nodes"])
    matched = sum((pred_labels & true_labels).values())
    if not matched:
        return 0.0
    pred_full = Counter((node_label(n), _shape_key(n)) for n in pred["nodes"])
    true_full = Counter((node_label(n), _shape_key(n)) for n in truth["nodes"])
    return sum((pred_full & true_full).values()) / matched


def compare(pred: dict, truth: dict) -> dict:
    errors = check(pred)
    structural = check(pred, shapes=False)
    nodes = _f1(Counter(node_label(n) for n in pred["nodes"]),
                Counter(node_label(n) for n in truth["nodes"]))
    edges = _f1(_edge_labels(pred), _edge_labels(truth))
    exact = (not structural and nodes["f1"] == 1.0 and edges["f1"] == 1.0 and nx.is_isomorphic(
        to_networkx(pred), to_networkx(truth),
        node_match=lambda a, b: a["label"] == b["label"],
        edge_match=lambda a, b: a["label"] == b["label"]))
    return {"valid": not errors, "valid_topology": not structural,
            "errors": errors, "exact": exact,
            "node_f1": nodes["f1"], "edge_f1": edges["f1"],
            "shape_acc": _shape_accuracy(pred, truth)}

"""Structural checks: is this (possibly LLM-written) graph a well-formed circuit DAG?

A circuit DAG must satisfy, besides being acyclic, a "wire" invariant: the
nodes touching a given wire form a single chain along edges labelled with that
wire. That is what separates a circuit DAG from an arbitrary DAG.
"""

from collections import Counter

import networkx as nx

from .schema import node_wires


def to_networkx(graph: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in graph["nodes"]:
        g.add_node(n["id"], label=node_label(n), **n)
    for e in graph["edges"]:
        g.add_edge(e["src"], e["dst"], wires=tuple(sorted(e["wires"])))
    return g


def node_label(node: dict) -> tuple:
    return (node["gate"], tuple(node["qubits"]), tuple(node["clbits"]), tuple(node["params"]))


def check(graph: dict) -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    errors = []
    ids = [n["id"] for n in graph["nodes"]]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        errors.append(f"duplicate node ids: {dupes}")
    by_id = {n["id"]: n for n in graph["nodes"]}

    for n in graph["nodes"]:
        if any(q >= graph["num_qubits"] for q in n["qubits"]) or any(c >= graph["num_clbits"] for c in n["clbits"]):
            errors.append(f"node {n['id']} uses a wire outside the declared register")
        if len(set(node_wires(n))) != len(node_wires(n)):
            errors.append(f"node {n['id']} uses the same wire twice")

    wire_in, wire_out, wire_edges = Counter(), Counter(), Counter()
    for e in graph["edges"]:
        s, d = e["src"], e["dst"]
        if s not in by_id or d not in by_id:
            errors.append(f"edge {s}->{d} references a missing node")
            continue
        if s == d:
            errors.append(f"self-loop on node {s}")
        for w in e["wires"]:
            if w not in node_wires(by_id[s]) or w not in node_wires(by_id[d]):
                errors.append(f"edge {s}->{d} carries {w}, but not both endpoints act on {w}")
            wire_out[(s, w)] += 1
            wire_in[(d, w)] += 1
            wire_edges[w] += 1

    if not errors and not nx.is_directed_acyclic_graph(to_networkx(graph)):
        errors.append("graph has a cycle")

    for key, count in list(wire_in.items()) + list(wire_out.items()):
        if count > 1:
            errors.append(f"node {key[0]} has {count} edges on wire {key[1]} in one direction")
    nodes_on_wire = Counter(w for n in graph["nodes"] for w in node_wires(n))
    for w, k in nodes_on_wire.items():
        if wire_edges[w] != k - 1:
            errors.append(f"wire {w}: {k} nodes should be chained by {k - 1} edges, found {wire_edges[w]}")
    return errors

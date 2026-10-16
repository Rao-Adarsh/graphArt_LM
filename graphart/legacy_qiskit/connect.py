"""Build the edges of a circuit DAG from an ordered list of operations.

This is the "code connects" half of the Phase-1 ablation: the model is asked
only to unroll the code into an ordered gate list, and this module derives the
edges the way `extract.py` does -- by walking each wire and joining consecutive
operations on it.

If the ablation scores much higher than a full-graph prompt, the hard part of
the task is building the graph. If it does not, the hard part is executing the
code. Either answer is worth knowing before Phase 2.
"""

from .schema import node_wires


def _wire_sort_key(wire: str) -> tuple[bool, int]:
    """Qubits before clbits, then numerically -- matching extract.circuit_to_graph."""
    return (wire[0] == "c", int(wire[1:]))


def add_edges(graph: dict) -> dict:
    """Return a copy of `graph` whose edges are derived from the node order.

    Nodes are taken in increasing id order, which is the order the model was
    asked to list them in. Any edges already present are discarded.
    """
    last_on_wire: dict[str, int] = {}
    merged: dict[tuple[int, int], list[str]] = {}

    for node in sorted(graph["nodes"], key=lambda n: n["id"]):
        for wire in node_wires(node):
            previous = last_on_wire.get(wire)
            if previous is not None:
                merged.setdefault((previous, node["id"]), []).append(wire)
            last_on_wire[wire] = node["id"]

    edges = [{"src": src, "dst": dst, "wires": sorted(wires, key=_wire_sort_key)}
             for (src, dst), wires in sorted(merged.items())]
    return {**graph, "edges": edges}

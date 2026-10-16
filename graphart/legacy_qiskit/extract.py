"""Ground truth: run the source code, take the circuit it builds, and convert
Qiskit's DAGCircuit into our schema (see schema.py)."""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from .schema import PARAM_DECIMALS, wire_name


def circuit_from_code(code: str, var: str = "qc") -> QuantumCircuit:
    """Execute trusted (generated) code and return the circuit stored in `var`."""
    namespace: dict = {}
    exec(code, namespace)
    circuit = namespace.get(var)
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError(f"code did not define a QuantumCircuit named {var!r}")
    return circuit


def circuit_to_graph(circuit: QuantumCircuit) -> dict:
    dag = circuit_to_dag(circuit)
    qidx = {q: i for i, q in enumerate(dag.qubits)}
    cidx = {c: i for i, c in enumerate(dag.clbits)}

    # Number op nodes in Qiskit's deterministic topological order.
    ids, nodes = {}, []
    for node in dag.topological_op_nodes():
        ids[node._node_id] = len(nodes)
        nodes.append({
            "id": len(nodes),
            "gate": node.op.name,
            "qubits": [qidx[q] for q in node.qargs],
            "clbits": [cidx[c] for c in node.cargs],
            "params": [round(float(p), PARAM_DECIMALS) for p in node.op.params],
        })

    # Qiskit gives one edge per wire; merge parallel wires into one edge.
    merged: dict[tuple[int, int], list[str]] = {}
    for src, dst, wire in dag.edges():
        if src._node_id not in ids or dst._node_id not in ids:
            continue  # edge touches an input/output node
        name = wire_name("q", qidx[wire]) if wire in qidx else wire_name("c", cidx[wire])
        merged.setdefault((ids[src._node_id], ids[dst._node_id]), []).append(name)
    edges = [{"src": s, "dst": d, "wires": sorted(w, key=lambda x: (x[0] == "c", int(x[1:])))}
             for (s, d), w in sorted(merged.items())]

    return {"num_qubits": circuit.num_qubits, "num_clbits": circuit.num_clbits,
            "nodes": nodes, "edges": edges}


def graph_from_code(code: str) -> tuple[dict, QuantumCircuit]:
    circuit = circuit_from_code(code)
    return circuit_to_graph(circuit), circuit

"""The shared graph format used for ground truth AND for LLM output.

A circuit DAG is stored as a plain dict:

    {
      "num_qubits": 3, "num_clbits": 0,
      "nodes": [{"id": 0, "gate": "h", "qubits": [0], "clbits": [], "params": []}, ...],
      "edges": [{"src": 0, "dst": 1, "wires": ["q0"]}, ...]
    }

Only operation nodes are kept (Qiskit's input/output nodes are dropped), so a
gate with no predecessor on a wire is simply the first thing on that wire.
If two consecutive gates share several wires (e.g. cx(0,1) then cz(0,1)) that
is ONE edge carrying both wires.

The same graph has a compact text form, which is what an LLM reads/writes:

    QUBITS 3
    CLBITS 0
    NODES
    0 h q0
    1 cx q0 q1
    2 rz(0.785) q2
    3 measure q0 c0
    EDGES
    0 -> 1 : q0
    1 -> 3 : q0
"""

import re

PARAM_DECIMALS = 3


def wire_name(kind: str, index: int) -> str:
    return f"{kind}{index}"


def node_wires(node: dict) -> list[str]:
    return [wire_name("q", q) for q in node["qubits"]] + [wire_name("c", c) for c in node["clbits"]]


def to_text(graph: dict) -> str:
    lines = [f"QUBITS {graph['num_qubits']}", f"CLBITS {graph['num_clbits']}", "NODES"]
    for n in graph["nodes"]:
        gate = n["gate"]
        if n["params"]:
            gate += "(" + ",".join(f"{p:g}" for p in n["params"]) + ")"
        lines.append(f"{n['id']} {gate} {' '.join(node_wires(n))}")
    lines.append("EDGES")
    for e in graph["edges"]:
        lines.append(f"{e['src']} -> {e['dst']} : {' '.join(e['wires'])}")
    return "\n".join(lines)


_NODE_RE = re.compile(r"^(\d+)\s+([a-zA-Z_][\w]*)(?:\(([^)]*)\))?((?:\s+[qc]\d+)*)\s*$")
_EDGE_RE = re.compile(r"^(\d+)\s*->\s*(\d+)\s*:\s*((?:[qc]\d+\s*)+)$")


class ParseError(ValueError):
    pass


def from_text(text: str) -> dict:
    """Parse the text form back into a graph dict. Raises ParseError on bad lines."""
    graph = {"num_qubits": 0, "num_clbits": 0, "nodes": [], "edges": []}
    section = None
    for lineno, raw in enumerate(text.strip().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("QUBITS"):
            graph["num_qubits"] = int(line.split()[1])
        elif line.startswith("CLBITS"):
            graph["num_clbits"] = int(line.split()[1])
        elif line in ("NODES", "EDGES"):
            section = line
        elif section == "NODES" and (m := _NODE_RE.match(line)):
            wires = m.group(4).split()
            params = [float(p) for p in m.group(3).split(",")] if m.group(3) else []
            graph["nodes"].append({
                "id": int(m.group(1)),
                "gate": m.group(2).lower(),
                "qubits": [int(w[1:]) for w in wires if w[0] == "q"],
                "clbits": [int(w[1:]) for w in wires if w[0] == "c"],
                "params": [round(p, PARAM_DECIMALS) for p in params],
            })
        elif section == "EDGES" and (m := _EDGE_RE.match(line)):
            graph["edges"].append({"src": int(m.group(1)), "dst": int(m.group(2)),
                                   "wires": m.group(3).split()})
        else:
            raise ParseError(f"line {lineno}: cannot parse {raw!r}")
    return graph

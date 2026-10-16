"""The shared text format for a PyTorch forward-pass graph.

This is what the ground truth is written to and what a model is asked to
produce, so it is the one definition of "the answer". `extract.py` builds the
dict; this module turns it into text and back.

A graph is a plain dict:

    {
      "num_inputs": 1, "input_shapes": [[1, 8]], "shapes_available": True,
      "nodes": [{"id": 0, "op": "placeholder", "target": "x",
                 "cls": None, "params": {}, "shape": [1, 8]}, ...],
      "edges": [{"src": 0, "dst": 1, "args": [0]}, ...]
    }

and its text form looks like this:

    INPUTS 1
    NODES
    0 placeholder x [1,8]
    1 call_module fc1 Linear(in=8,out=16,bias=True) [1,16]
    2 call_function torch.relu [1,16]
    3 call_module fc2 Linear(in=16,out=8,bias=True) [1,8]
    4 call_function add [1,8]
    5 call_module out Linear(in=8,out=2,bias=True) [1,2]
    6 output output [1,2]
    EDGES
    0 -> 1 : arg0
    1 -> 2 : arg0
    2 -> 3 : arg0
    3 -> 4 : arg0
    0 -> 4 : arg1
    4 -> 5 : arg0
    5 -> 6 : arg0

An edge carries every argument slot its source fills, so `attn(x, x, x)` is one
edge reading `: arg0 arg1 arg2` rather than three separate edges -- the same
merge the circuit schema did when two gates shared several wires.

Three deliberate choices, each with a cost:

* **The `op` keyword is redundant** -- it can be inferred from whether a class
  follows. It is kept because it costs ~16% of the output and buys an
  unambiguous anchor: without it a bare `fc1` and a bare `torch.relu` are not
  distinguishable, and a parser has nothing to resync on after a bad line.
  Measured on a 1.5B model in the previous domain, malformed output was the
  dominant failure mode, not wrong reasoning, so redundancy earns its keep.

* **Parameters are named, not positional.** `Conv2d(in=3,out=64,k=7)` costs
  ~8% more than `Conv2d(3,64,7)` but survives being written in a different
  order, where the positional form would be silently wrong.

* **The batch dimension is written as traced**, not as a symbol. Writing `B`
  would read better but would break `from_text(to_text(g)) == g`, which
  `score.py` depends on when it compares a parsed reply against the truth dict.

Edges carry their argument slots because `a - b` and `cat([a, b])` are order
sensitive -- the same role the wire label played in the circuit schema.
"""

import re

# Inside a parameter list, commas separate parameters, so a list-valued
# parameter (an asymmetric kernel, say) is bracketed and joined with `x`:
# k=[3x5]. The brackets matter for the one-element case -- see _fmt_value.
LIST_SEP = "x"
# A node with several outputs (chunk, split) has several shapes.
TUPLE_SEP = "|"
NO_SHAPE = "?"

OPS = ("placeholder", "get_attr", "call_module", "call_function", "call_method", "output")


class ParseError(ValueError):
    pass


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def _fmt_value(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        # Bracketed even when there is one element, so LayerNorm's shape=[96]
        # does not come back as the plain int 96.
        return "[" + LIST_SEP.join(_fmt_value(v) for v in value) + "]"
    return str(value)


def _fmt_params(params: dict) -> str:
    if not params:
        return ""
    return "(" + ",".join(f"{k}={_fmt_value(v)}" for k, v in params.items()) + ")"


def _fmt_dims(dims) -> str:
    return NO_SHAPE if dims is None else ",".join(str(d) for d in dims)


def _fmt_shape(shape) -> str:
    if shape is None:
        return ""
    if shape and isinstance(shape[0], (list, type(None))):   # several outputs
        return " [" + TUPLE_SEP.join(_fmt_dims(s) for s in shape) + "]"
    return " [" + _fmt_dims(shape) + "]"


def to_text(graph: dict) -> str:
    lines = [f"INPUTS {graph['num_inputs']}", "NODES"]
    for node in graph["nodes"]:
        cls = f" {node['cls']}{_fmt_params(node['params'])}" if node["cls"] else ""
        lines.append(f"{node['id']} {node['op']} {node['target']}{cls}"
                     f"{_fmt_shape(node['shape'])}")
    lines.append("EDGES")
    for edge in graph["edges"]:
        slots = " ".join(f"arg{a}" for a in edge["args"])
        lines.append(f"{edge['src']} -> {edge['dst']} : {slots}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
_NODE_RE = re.compile(
    r"^(\d+)\s+(" + "|".join(OPS) + r")\s+(\S+)"        # id, op, target
    r"(?:\s+([A-Za-z_]\w*)(?:\(([^)]*)\))?)?"           # optional Class(params)
    r"(?:\s+\[([^\]]*)\])?\s*$")                        # optional [shape]

# The slot list is optional when reading. A model that omits it on a
# single-input node is unambiguous; one that omits it elsewhere gets arg0,
# which simply fails to match the truth. Being lenient here never hides an
# error, it only avoids discarding an otherwise sound answer.
_EDGE_RE = re.compile(r"^(\d+)\s*->\s*(\d+)(?:\s*:\s*((?:arg\d+[\s,]*)+))?\s*$")


def _parse_value(text: str):
    # Brackets are what make a list a list, so a string value that happens to
    # contain the separator is never split by accident.
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        return [_parse_value(p) for p in inner.split(LIST_SEP)] if inner else []
    if text == "True":
        return True
    if text == "False":
        return False
    if text == "None":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_params(text: str | None) -> dict:
    if not text:
        return {}
    params = {}
    for item in text.split(","):
        key, sep, value = item.partition("=")
        if not sep:
            raise ParseError(f"parameter {item!r} is not key=value")
        params[key.strip()] = _parse_value(value.strip())
    return params


def _parse_dims(text: str):
    if text == NO_SHAPE:
        return None
    return [int(d) for d in text.split(",") if d.strip()]


def _parse_shape(text: str | None):
    if text is None:
        return None
    if TUPLE_SEP in text:
        return [_parse_dims(part) for part in text.split(TUPLE_SEP)]
    return _parse_dims(text)


def from_text(text: str) -> dict:
    """Parse the text form back into a graph dict. Raises ParseError on bad lines."""
    graph = {"num_inputs": 0, "input_shapes": [], "shapes_available": False,
             "nodes": [], "edges": []}
    section = None
    for lineno, raw in enumerate(text.strip().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("INPUTS"):
            try:
                graph["num_inputs"] = int(line.split()[1])
            except (IndexError, ValueError):
                raise ParseError(f"line {lineno}: cannot read {raw!r}") from None
        elif line in ("NODES", "EDGES"):
            section = line
        elif section == "NODES" and (m := _NODE_RE.match(line)):
            try:
                params = _parse_params(m.group(5))
            except ParseError as exc:
                raise ParseError(f"line {lineno}: {exc}") from None
            graph["nodes"].append({
                "id": int(m.group(1)), "op": m.group(2), "target": m.group(3),
                "cls": m.group(4), "params": params,
                "shape": _parse_shape(m.group(6)),
            })
        elif section == "EDGES" and (m := _EDGE_RE.match(line)):
            slots = [int(a) for a in re.findall(r"arg(\d+)", m.group(3) or "arg0")]
            graph["edges"].append({"src": int(m.group(1)), "dst": int(m.group(2)),
                                   "args": slots})
        else:
            raise ParseError(f"line {lineno}: cannot parse {raw!r}")

    graph["input_shapes"] = [n["shape"] for n in graph["nodes"] if n["op"] == "placeholder"]
    graph["shapes_available"] = any(n["shape"] is not None for n in graph["nodes"])
    return graph

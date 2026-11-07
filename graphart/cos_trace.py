"""Training traces for the two-step task, recorded during the fx trace itself.

What the two steps look like is in graphart/two_step.py. Each node's step-1 line
carries its source call stack, e.g. `@56>16` = line 56 called something whose line 16
made this node; each step-2 input carries the variable it was read from, unless no
variable holds it (a layer feeding the next inside an nn.Sequential, a nested call).

Nothing here is written by hand or inferred from the text. While fx traces the
model, `ProvenanceTracer` looks at the Python frames that are running the source:
their line numbers give the call stack, and their local variables, at the moment a
node is made, say which variable holds each input. The graph is built by the same
`graph_from_module` as the ground truth, so node ids match it by construction, and
`check_trace` confirms the trace reads back into exactly that graph.

This is ground-truth generation, like the extractor; the model never runs fx.
"""

import ast
import re
import sys
from typing import Any

import torch
from torch.fx import Node, Proxy

from graphart import two_step
from graphart.extract import GranularTracer, graph_from_module
from graphart.schema import ParseError

SRC = "<src>"          # the filename the source is compiled under, to find its frames


# --------------------------------------------------------------------------
# where each line's statement starts and ends
# --------------------------------------------------------------------------
def statement_spans(code: str) -> dict[int, tuple[int, int]]:
    """Each source line -> (first, last) line of the innermost statement holding it.

    A compound statement (for, if, with, def) counts only its header, so a line in a
    loop body maps to its own statement, not to the whole loop.
    """
    spans: dict[int, tuple[int, int]] = {}
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.stmt):
            continue
        first, last = node.lineno, node.end_lineno
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            # the header: up to the line before the body (decorators sit above lineno)
            last = max(first, body[0].lineno - 1)
        for line in range(first, last + 1):
            if line not in spans or spans[line][1] - spans[line][0] > last - first:
                spans[line] = (first, last)
    return spans


# --------------------------------------------------------------------------
# the tracer
# --------------------------------------------------------------------------
def _holders(value_of: dict[str, Any], node: Node) -> list[str]:
    """The local names whose value is `node`'s proxy, or a list/tuple holding it."""
    names = []
    for name, value in value_of.items():
        # Only isinstance and .node are touched: anything else on a Proxy (bool, ==,
        # iteration) would itself be traced, or raise.
        if isinstance(value, Proxy):
            if value.node is node:
                names.append(name)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, Proxy) and item.node is node:
                    names.append(f"{name}[{i}]")
    return names


class ProvenanceTracer(GranularTracer):
    """GranularTracer that also records, for every node, its source call stack and
    which local variable held each of its inputs."""

    def __init__(self, code: str):
        super().__init__()
        self.spans = statement_spans(code)
        self.lines = code.splitlines()
        self.records: dict[Node, dict] = {}

    def statement(self, line: int) -> tuple[int, str]:
        first, last = self.spans.get(line, (line, line))
        return first, "\n".join(self.lines[first - 1:last])

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        frames, f = [], sys._getframe(1)
        while f is not None:
            if f.f_code.co_filename == SRC:
                frames.append(f)
            f = f.f_back
        frames.reverse()                                    # outermost first
        stack = [self.statement(fr.f_lineno)[0] for fr in frames]
        names: dict[Node, str] = {}
        if frames:
            _, text = self.statement(frames[-1].f_lineno)
            local = frames[-1].f_locals
            for src in node.all_input_nodes:
                # Keep a name only if this statement spells it: an alias the line does
                # not use is true but no help to a reader. Several -- the same value
                # under two names, both on the line -- the first one written.
                found = []
                for cand in _holders(local, src):
                    m = re.search(rf"(?<![\w.]){re.escape(cand.split('[')[0])}(?!\w)", text)
                    if m:
                        found.append((m.start(), cand))
                if found:
                    names[src] = min(found)[1]
        self.records[node] = {"stack": stack, "names": names}
        return node


def _compile(code: str) -> dict:
    namespace: dict = {}
    exec(compile(code, SRC, "exec"), namespace)
    return namespace


def trace(code: str, var: str = "model") -> dict:
    """The ground-truth graph plus, per node id, its source stack and input names.

    Returns {"graph": ..., "stack": {id: [line, ...]}, "names": {(src, dst): name}}.
    """
    namespace = _compile(code)
    model = namespace.get(var)
    if not isinstance(model, torch.nn.Module):
        found = [v for k, v in namespace.items() if isinstance(v, torch.nn.Module) and not k.startswith("_")]
        if len(found) != 1:
            raise ValueError(f"expected one nn.Module in the source, found {len(found)}")
        model = found[0]
    example = namespace.get("example_input")
    tracer = ProvenanceTracer(code)
    graph = graph_from_module(model, example_input=example if isinstance(example, torch.Tensor) else None,
                              tracer=tracer)
    order = {node: i for i, node in enumerate(tracer.graph.nodes)}
    stack, names = {}, {}
    for node, rec in tracer.records.items():
        if node not in order:                   # made and then dropped by fx
            continue
        stack[order[node]] = rec["stack"]
        for src, name in rec["names"].items():
            names[(order[src], order[node])] = name
    return {"graph": graph, "stack": stack, "names": names}


def render_step1(t: dict) -> str:
    return two_step.render_step1(t["graph"], t["stack"])


def render_step2(t: dict) -> str:
    return two_step.render_step2(t["graph"], t["names"])


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
_OPERATORS = {"add": "+", "sub": "-", "mul": "*", "truediv": "/", "floordiv": "//", "matmul": "@",
              "mod": "%", "neg": "-", "eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
              "getitem": "[", "pow": "**", "iadd": "+=", "imul": "*=", "and_": "&", "or_": "|"}


def check_trace(code: str, t: dict) -> list[str]:
    """Everything that would make a trace wrong; [] when it is sound.

    1. Round trip: both steps read back into exactly the ground-truth graph, and the
       stacks and names read back unchanged.
    2. Every cited line exists, and the innermost one plausibly makes the node: it
       spells a part of the module path, or the function / method / operator.
    3. Every named variable is spelled on the statement that reads it.
    """
    problems = []
    s1, s2 = render_step1(t), render_step2(t)
    try:
        graph, stack = two_step.parse_step1(s1)
        graph["edges"], names = two_step.parse_step2(s2)
    except ParseError as exc:
        return [f"does not parse: {exc}"]
    truth = t["graph"]
    norm = lambda g: (g["nodes"], sorted((e["src"], e["dst"], tuple(e["args"])) for e in g["edges"]))
    if norm(graph) != norm(truth):
        problems.append("round trip: graph differs from the ground truth")
    if stack != {k: v for k, v in t["stack"].items() if v}:
        problems.append("round trip: stacks differ")
    if names != t["names"]:
        problems.append("round trip: names differ")

    lines = code.splitlines()
    spans = statement_spans(code)
    stmt = lambda ln: "\n".join(lines[spans.get(ln, (ln, ln))[0] - 1:spans.get(ln, (ln, ln))[1]])
    # `for blk in self.blocks: x = blk(x)` names the modules on the loop header only
    loops = [(n.lineno, n.end_lineno, stmt(n.lineno)) for n in ast.walk(ast.parse(code)) if isinstance(n, ast.For)]
    headers = lambda ln: [h for a, b, h in loops if a < ln <= b]
    for n in truth["nodes"]:
        st = t["stack"].get(n["id"]) or []
        if any(not 1 <= ln <= len(lines) for ln in st):
            problems.append(f"node {n['id']}: cites a line outside the code {st}")
            continue
        if not st:
            if n["op"] not in ("placeholder", "output"):
                problems.append(f"node {n['id']} ({n['op']} {n['target']}): no source line")
            continue
        text = stmt(st[-1])
        target = n["target"]
        if n["op"] == "call_module":
            parts = [p for p in target.split(".") if not p.isdigit()]
            texts = [stmt(ln) for ln in st] + [h for ln in st for h in headers(ln)]
            ok = any(re.search(rf"\b{re.escape(p)}\b", s) for p in parts for s in texts)
        else:
            short = target.split(".")[-1]
            ok = short in text or _OPERATORS.get(short.lstrip("i"), "\0") in text or _OPERATORS.get(short, "\0") in text
            if short == "getitem":          # `a, b = x.chunk(2)` unpacks with getitem
                ok = ok or bool(re.match(r"\s*[\w\s,()\[\]]+,[\w\s,()\[\]]*=[^=]", text))
            if short == "getattr":          # `x.shape`
                ok = ok or bool(re.search(r"\w\.\w", text))
        if not ok:
            problems.append(f"node {n['id']} ({n['op']} {target}): line {st[-1]} does not spell it: {text.strip()!r}")
    for (src, dst), name in t["names"].items():
        ln = (t["stack"].get(dst) or [None])[-1]
        if ln is None or not re.search(rf"(?<![\w.]){re.escape(name.split('[')[0])}(?!\w)", stmt(ln)):
            problems.append(f"edge {src}->{dst}: name {name!r} not on line {ln}")
    return problems

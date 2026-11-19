"""The hierarchical schema: a module tree plus one small wiring graph per block.

The flat torch.fx graph of a real network is 150-450 nodes. This schema describes the same
network the way its code is written:

* the **module tree** -- what `__init__` builds: every module by attribute name, nested,
  built-in layers with their parameters (`Conv2d(in=16,out=32,k=3,s=2,p=1)`), classes
  defined in the source tagged with a variant (`BasicBlock#2`), classes not in the source
  as opaque boxes, runs of identical numbered siblings written once (`0-2: Block#1`);
* one **wiring graph per variant** -- what a class's `forward()` does: its inputs, its
  direct children and the functions it calls, and which value feeds which argument.

`inline()` expands the tree and the graphs back into the flat graph, and must reproduce
the torch.fx trace exactly (`check_lossless`), so fx stays the answer key.

    from graphart.hier import hier_from_code, to_text, from_text, inline
    h, flat = hier_from_code(code)
    text = to_text(h)                     # what a model is asked to write
    assert same_graph(inline(from_text(text), ref=h), flat)

How the graphs are cut out of the trace
---------------------------------------
One fx trace of the whole model, with a tracer that also records every module *call*:
its path, the arguments it was given (bound to the forward() signature) and what it
returned. Each node is stamped with the call it was created in. A block's wiring graph is
then the part of the trace inside that call, with every child call contracted to a single
node. Cutting graphs out of one trace -- rather than tracing each class on its own -- keeps
the context: a block traced alone sees `mask=None` as a tensor and takes the wrong branch.

Kinds of tree node
------------------
    leaf     a built-in torch.nn layer (fx does not trace inside them): Class(params)
    block    a class defined in the source, with a forward(): Class#k + a wiring graph
    chain    an nn.Sequential (or a source subclass with no forward of its own): runs its
             children in order, so it needs no graph
    list     nn.ModuleList / nn.ModuleDict: never called; the parent's graph uses
             `layers.0` directly
    opaque   a class whose source is not shown (e.g. torchvision's Conv2dNormActivation):
             one box, class name only. Its insides are kept in the truth (hidden) so the
             flat graph can still be rebuilt, but they are never written or asked for.

Wiring graph text
-----------------
    GRAPH BasicBlock#1
    0 input x
    1 conv1 <- 0              a child module, by attribute path relative to the block
    2 bn1 <- 1
    3 relu <- 2
    4 conv2 <- 3
    5 bn2 <- 4
    6 add() <- 5, 0           a function (normalised as in extract.ALIASES)
    7 relu <- 6               a child used twice appears twice
    8 output <- 7

`.view()` is a tensor method, `@pos_embedding` a parameter read (get_attr). Sources are
listed in argument order. `input features*` is an input that receives a *list* of tensors
(DenseNet's layers); `0*` passes all of it, `0[2]` one element; `5[1]` is the second
value returned by a child that returns several.
"""

from __future__ import annotations

import inspect
import re
from collections import Counter, defaultdict
from typing import Any

import torch
import torch.fx
from torch import nn

from .extract import GranularTracer, _input_slots, _layer_params, _target_name, graph_from_module
from .schema import _fmt_params, _parse_params

SRC_MODULE = "__graphart_src__"     # __module__ of classes defined in exec'd source


class HierError(ValueError):
    pass


# --------------------------------------------------------------------------
# tracing with call records
# --------------------------------------------------------------------------
class _Call:
    __slots__ = ("id", "parent", "path", "module", "leaf", "start", "inputs", "outputs", "hidden_in")

    def __init__(self, id, parent, path, module, leaf, start):
        self.id, self.parent, self.path, self.module, self.leaf, self.start = id, parent, path, module, leaf, start
        self.inputs: list[tuple[str, list, bool]] = []   # (name, [fx nodes], is_pack)
        self.outputs: list = []                          # fx nodes returned, flattened


def _proxies(value) -> list:
    """fx nodes inside an argument or return value, in order."""
    if isinstance(value, torch.fx.Proxy):
        return [value.node]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _proxies(v)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _proxies(v)]
    return []


class HierTracer(GranularTracer):
    """GranularTracer that records each module call and stamps nodes with their call."""

    def __init__(self):
        super().__init__()
        self.calls: list[_Call] = [_Call(0, None, "", None, False, 0)]
        self.stack = [0]
        self.n_created = 0

    def create_node(self, *args, **kwargs):
        node = super().create_node(*args, **kwargs)
        node.meta["hier_call"] = self.stack[-1]
        self.n_created += 1
        return node

    def call_module(self, m, forward, args, kwargs):
        path = self.path_of_module(m)
        rec = _Call(len(self.calls), self.stack[-1], path, m, self.is_leaf_module(m, path), self.n_created)
        try:
            bound = inspect.signature(m.forward).bind(*args, **kwargs)
            for name, value in bound.arguments.items():
                kind = inspect.signature(m.forward).parameters[name].kind
                items = value if kind == inspect.Parameter.VAR_POSITIONAL else [value]
                for i, item in enumerate(items):
                    nodes = _proxies(item)
                    if nodes:
                        label = name if kind != inspect.Parameter.VAR_POSITIONAL else f"{name}{i}"
                        rec.inputs.append((label, nodes, not isinstance(item, torch.fx.Proxy)))
        except TypeError:                                   # unbindable: positional order
            for i, item in enumerate(list(args) + list(kwargs.values())):
                nodes = _proxies(item)
                if nodes:
                    rec.inputs.append((f"in{i}", nodes, not isinstance(item, torch.fx.Proxy)))
        self.calls.append(rec)
        self.stack.append(rec.id)
        try:
            ret = super().call_module(m, forward, args, kwargs)
        finally:
            self.stack.pop()
        rec.outputs = _proxies(ret)
        return ret


# --------------------------------------------------------------------------
# kinds
# --------------------------------------------------------------------------
def _is_torch_nn(cls) -> bool:
    return cls.__module__.startswith("torch.nn")


def _own_forward(cls) -> bool:
    return any("forward" in vars(k) for k in cls.__mro__
               if k not in (nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.Module, object))


def _kind(m: nn.Module, in_source) -> str:
    cls = type(m)
    if isinstance(m, (nn.ModuleList, nn.ModuleDict)) and not _own_forward(cls):
        return "list" if (_is_torch_nn(cls) or in_source(cls)) else "opaque"
    if isinstance(m, nn.Sequential) and not _own_forward(cls):
        return "chain" if (_is_torch_nn(cls) or in_source(cls)) else "opaque"
    if _is_torch_nn(cls):
        return "leaf"
    return "block" if in_source(cls) else "opaque"


# --------------------------------------------------------------------------
# wiring graphs cut out of the trace
# --------------------------------------------------------------------------
def _ancestors(calls, cid) -> list[int]:
    out = []
    while cid is not None:
        out.append(cid)
        cid = calls[cid].parent
    return out


def _wiring(calls, cid, fx_nodes, order, anc) -> tuple[list[dict], list[str]]:
    """The wiring graph of call `cid`: its inputs, its units in execution order, its output.

    Returns (nodes, problems). Each node: {"kind", "target", "pack", "src": [token, ...]}.
    """
    problems: list[str] = []
    rec = calls[cid]
    children = [c for c in calls if c.parent == cid and c.id != cid]
    child_of = {}                      # call id (any depth below cid) -> direct child call id
    for c in calls:
        a = anc[c.id]
        if cid in a and c.id != cid:
            child_of[c.id] = a[a.index(cid) - 1]
    direct = [n for n in fx_nodes if n.meta["hier_call"] == cid and n.op not in ("placeholder", "output")]
    if cid == 0:
        inputs = [(n.target, [n], False) for n in fx_nodes if n.op == "placeholder"]
    else:
        inputs = rec.inputs

    nodes: list[dict] = []
    for name, ns, pack in inputs:
        nodes.append({"kind": "input", "target": str(name), "pack": pack, "src": []})
    # units in execution order: a child call sorts before a node created at its start index
    units = [((order[n], 1), "node", n) for n in direct] + [((c.start, 0), "call", c) for c in children]
    units.sort(key=lambda u: u[0])
    unit_id = {}
    for i, (_, typ, obj) in enumerate(units):
        unit_id[(typ, id(obj))] = len(nodes) + i
    base = len(nodes)

    def token(v) -> str | None:
        """How the flat value v is referred to inside this block."""
        owner = v.meta["hier_call"]
        if v.op == "call_module" and calls[owner].leaf:     # a leaf call's node
            if calls[owner].parent == cid:
                return str(unit_id[("call", id(calls[owner]))])
        if owner == cid and v.op not in ("placeholder", "output") and ("node", id(v)) in unit_id:
            return str(unit_id[("node", id(v))])
        if owner in child_of:
            child = calls[child_of[owner]]
            if v in child.outputs:
                u = unit_id[("call", id(child))]
                return str(u) if len(child.outputs) == 1 else f"{u}[{child.outputs.index(v)}]"
            problems.append(f"value {v.name} escapes {child.path or 'root'} without being returned")
            return None
        for p, (name, ns, pack) in enumerate(inputs):
            if v in ns:
                return f"{p}[{ns.index(v)}]" if pack else str(p)
        problems.append(f"{rec.path or 'root'} reads {v.name}, which it was not given")
        return None

    def tokens(vs) -> list[str]:
        toks = [token(v) for v in vs]
        toks = [t for t in toks if t is not None]
        # p[0], p[1], ... p[k-1] in a row, covering all of a pack -> p*
        out, i = [], 0
        while i < len(toks):
            m = re.fullmatch(r"(\d+)\[0\]", toks[i])
            if m:
                p = int(m.group(1))
                if p < len(inputs) and inputs[p][2]:
                    k = len(inputs[p][1])
                    if toks[i:i + k] == [f"{p}[{j}]" for j in range(k)]:
                        out.append(f"{p}*")
                        i += k
                        continue
            out.append(toks[i])
            i += 1
        return out

    def slot_sources(n) -> list:
        slots = _input_slots(n, order)
        by_slot = {s: src for src, ss in slots.items() for s in ss}
        return [fx_nodes[by_slot[s]] for s in sorted(by_slot)]

    rel = (lambda t: t[len(rec.path) + 1:]) if rec.path else (lambda t: t)
    for _, typ, obj in units:
        if typ == "call":
            kind = "module"
            target = rel(obj.path)
            src = tokens([v for _, ns, _ in obj.inputs for v in ns])
        else:
            n = obj
            kind = {"call_function": "function", "call_method": "method", "get_attr": "get_attr"}[n.op]
            target = rel(str(n.target)) if n.op == "get_attr" else _target_name(n)
            src = tokens(slot_sources(n))
        nodes.append({"kind": kind, "target": target, "pack": False, "src": src})
    if cid == 0:
        out_node = next(n for n in fx_nodes if n.op == "output")
        outs = slot_sources(out_node)
    else:
        outs = rec.outputs
    nodes.append({"kind": "output", "target": "output", "pack": False, "src": tokens(outs)})
    return nodes, problems


def graph_text(nodes: list[dict]) -> str:
    """Body lines of a wiring graph (no GRAPH header)."""
    lines = []
    for i, n in enumerate(nodes):
        k, t = n["kind"], n["target"]
        if k == "input":
            head = f"input {t}{'*' if n['pack'] else ''}"
        elif k == "output":
            head = "output"
        elif k == "module":
            head = t
        elif k == "function":
            head = f"{t}()"
        elif k == "method":
            head = f".{t}()"
        else:
            head = f"@{t}"
        lines.append(f"{i} {head}" + (f" <- {', '.join(n['src'])}" if n["src"] else ""))
    return "\n".join(lines)


def _is_chain(nodes: list[dict], child_names: list[str]) -> bool:
    """input -> each child in order -> output: what nn.Sequential.forward does."""
    if len(nodes) != len(child_names) + 2 or nodes[0]["kind"] != "input" or nodes[0]["pack"]:
        return False
    for i, name in enumerate(child_names, 1):
        n = nodes[i]
        if n["kind"] != "module" or n["target"] != name or n["src"] != [str(i - 1)]:
            return False
    return nodes[-1]["src"] == [str(len(nodes) - 2)]


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def hier_from_module(model: nn.Module, in_source, example_input=None) -> tuple[dict, dict]:
    """(hier, flat) for a module. `in_source(cls)` says whether a class's code is shown."""
    model = model.eval()
    tracer = HierTracer()
    flat = graph_from_module(model, example_input=example_input, tracer=tracer)
    calls = tracer.calls
    calls[0].module = model
    fx_nodes = list(tracer.graph.nodes)
    order = {n: i for i, n in enumerate(fx_nodes)}
    anc = {c.id: _ancestors(calls, c.id) for c in calls}
    problems: list[str] = []

    graphs_by_path: dict[str, list[str]] = defaultdict(list)     # path -> graph texts, one per call
    nodes_by_text: dict[str, list[dict]] = {}
    for c in calls:
        if c.leaf:
            continue
        nodes, probs = _wiring(calls, c.id, fx_nodes, order, anc)
        problems += probs
        text = graph_text(nodes)
        graphs_by_path[c.path].append(text)
        nodes_by_text[text] = nodes

    # the tree, pre-order; variants numbered per class by first appearance
    variant_no: dict[tuple, int] = {}
    per_class = Counter()
    variants: dict[str, dict] = {}

    def build(m: nn.Module, name: str, path: str, hidden: bool) -> dict:
        kind = _kind(m, in_source)
        if path == "" and kind in ("leaf", "opaque"):
            kind = "block"                  # the root is always expanded
        node: dict[str, Any] = {"name": name, "kind": kind, "cls": type(m).__name__, "params": {},
                                "tag": None, "hidden": hidden, "children": []}
        if kind == "leaf":
            node["params"] = _layer_params(m)
            return node
        child_hidden = hidden or kind == "opaque"
        node["children"] = [build(c, n, f"{path}.{n}" if path else n, child_hidden)
                            for n, c in m.named_children()]
        texts = graphs_by_path.get(path, [])
        if len(set(texts)) > 1:
            problems.append(f"{path or 'root'} is called {len(texts)} times with different wiring")
        if kind == "chain" and texts and not _is_chain(nodes_by_text[texts[0]], [c["name"] for c in node["children"]]):
            node["kind"] = kind = "block"          # a Sequential whose calls do not chain
            problems.append(f"{path or 'root'}: Sequential that does not run as a chain")
        if kind in ("block", "opaque") and texts:
            key = (type(m).__qualname__, texts[0])
            if key not in variant_no:
                per_class[type(m).__qualname__] += 1
                variant_no[key] = per_class[type(m).__qualname__]
                tag = f"{type(m).__name__}#{variant_no[key]}"
                variants[tag] = {"nodes": nodes_by_text[texts[0]], "hidden": child_hidden or kind == "opaque"}
            tag = f"{type(m).__name__}#{variant_no[key]}"
            node["tag"] = tag
            if not node["hidden"] and kind == "block":
                variants[tag]["hidden"] = False
        elif kind == "block":
            node["tag"] = f"{type(m).__name__}#0"          # defined but never run
        return node

    tree = build(model, "", "", False)
    # two classes may share a __name__; keep tags unique
    return {"tree": tree, "variants": variants, "problems": sorted(set(problems))}, flat


def load_source(code: str, var: str = "model"):
    """Execute generated / assembled source; return (model, example_input, in_source)."""
    namespace: dict = {"__name__": SRC_MODULE}
    exec(code, namespace)
    model = namespace.get(var)
    if not isinstance(model, nn.Module):
        found = [v for k, v in namespace.items() if isinstance(v, nn.Module) and not k.startswith("_")]
        if len(found) != 1:
            raise ValueError(f"expected one nn.Module in the source, found {len(found)}")
        model = found[0]
    ex = namespace.get("example_input")
    return model, ex if isinstance(ex, torch.Tensor) else None, (lambda cls: cls.__module__ == SRC_MODULE)


def hier_from_code(code: str) -> tuple[dict, dict]:
    model, ex, in_source = load_source(code)
    return hier_from_module(model, in_source, ex)


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
def _tree_desc(node: dict) -> str:
    if node["kind"] == "leaf":
        return node["cls"] + _fmt_params(node["params"])
    if node["kind"] in ("block",) and node["tag"]:
        return node["tag"]
    return node["cls"]


def _subtree_lines(node: dict, depth: int) -> list[str]:
    """Lines of a node's children (the node itself is written by the caller)."""
    if node["kind"] in ("leaf", "opaque"):
        return []
    lines, kids, i = [], node["children"], 0
    while i < len(kids):
        c = kids[i]
        body = _subtree_lines(c, depth + 1)
        j = i + 1
        if c["name"].isdigit():
            while (j < len(kids) and kids[j]["name"].isdigit() and int(kids[j]["name"]) == int(c["name"]) + (j - i)
                   and _tree_desc(kids[j]) == _tree_desc(c) and _subtree_lines(kids[j], depth + 1) == body):
                j += 1
        name = c["name"] if j == i + 1 else f"{c['name']}-{kids[j - 1]['name']}"
        lines.append("  " * depth + f"{name}: {_tree_desc(c)}")
        lines += body
        i = j
    return lines


def tree_text(h: dict) -> str:
    return "\n".join([_tree_desc(h["tree"])] + _subtree_lines(h["tree"], 1))


def to_text(h: dict) -> str:
    """The answer a model is asked to write: TREE, then one GRAPH per visible variant."""
    parts = ["TREE", tree_text(h)]
    seen = []

    def walk(node):
        if node["kind"] == "opaque":
            return
        if node["kind"] == "block" and node["tag"] in h["variants"] and node["tag"] not in seen:
            seen.append(node["tag"])
        for c in node["children"]:
            walk(c)
    walk(h["tree"])
    for tag in seen:
        parts += ["", f"GRAPH {tag}", graph_text(h["variants"][tag]["nodes"])]
    return "\n".join(parts)


_TREE_RE = re.compile(r"^(\s*)(?:([\w.]+(?:-\d+)?):\s*)?([A-Za-z_]\w*)(?:#(\d+)|\(([^)]*)\))?\s*$")
_GNODE_RE = re.compile(r"^(\d+)\s+(.+?)(?:\s*<-\s*(.*))?$")


def _parse_tree(lines: list[str]) -> dict:
    """Indentation gives nesting; `0-2: X` is expanded into three copies (children included)."""
    root, stack = None, []          # stack of (indent, node)
    for raw in lines:
        m = _TREE_RE.match(raw.rstrip())
        if not m:
            raise HierError(f"cannot parse tree line {raw!r}")
        indent, name, cls, tag, params = len(m.group(1).expandtabs(2)), m.group(2), m.group(3), m.group(4), m.group(5)
        node = {"name": name or "", "kind": "leaf" if params is not None else ("block" if tag else None),
                "cls": cls, "params": _parse_params(params) if params else {},
                "tag": f"{cls}#{tag}" if tag else None, "hidden": False, "children": []}
        if root is None:
            root, stack = node, [(indent, node)]
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            raise HierError(f"tree line {raw!r} is not inside the root")
        stack[-1][1]["children"].append(node)
        stack.append((indent, node))
    if root is None:
        raise HierError("empty tree")
    _expand_ranges(root)
    _finish_kinds(root)
    return root


def _expand_ranges(node: dict) -> None:
    kids = []
    for c in node["children"]:
        _expand_ranges(c)
        m = re.fullmatch(r"(\d+)-(\d+)", c["name"])
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a or b - a > 500:
                raise HierError(f"bad range {c['name']}")
            kids += [dict(_deepcopy_tree(c), name=str(k)) for k in range(a, b + 1)]
        else:
            kids.append(c)
    node["children"] = kids


def _deepcopy_tree(node: dict) -> dict:
    return {**{k: v for k, v in node.items() if k != "children"}, "children": [_deepcopy_tree(c) for c in node["children"]]}


def _finish_kinds(node: dict) -> None:
    for c in node["children"]:
        _finish_kinds(c)
    if node["kind"] is None:
        if node["cls"] in ("ModuleList", "ModuleDict"):
            node["kind"] = "list"
        elif node["children"]:
            node["kind"] = "chain"
        elif node["cls"] in ("Sequential",):
            node["kind"] = "chain"
        else:
            node["kind"] = "opaque"          # a bare class name with no children (or a param-less leaf)


def _parse_graph(lines: list[str]) -> list[dict]:
    nodes = []
    for raw in lines:
        m = _GNODE_RE.match(raw.strip())
        if not m:
            raise HierError(f"cannot parse graph line {raw!r}")
        if int(m.group(1)) != len(nodes):
            raise HierError(f"graph node ids must count up from 0: {raw!r}")
        head, src = m.group(2).strip(), [t.strip() for t in (m.group(3) or "").split(",") if t.strip()]
        for t in src:
            if not re.fullmatch(r"\d+(\*|\[\d+\])?", t):
                raise HierError(f"bad source {t!r} in {raw!r}")
        if head.startswith("input"):
            name = head[5:].strip()
            nodes.append({"kind": "input", "target": name.rstrip("*"), "pack": name.endswith("*"), "src": []})
        elif head == "output":
            nodes.append({"kind": "output", "target": "output", "pack": False, "src": src})
        elif head.startswith("@"):
            nodes.append({"kind": "get_attr", "target": head[1:], "pack": False, "src": src})
        elif head.startswith(".") and head.endswith("()"):
            nodes.append({"kind": "method", "target": head[1:-2], "pack": False, "src": src})
        elif head.endswith("()"):
            nodes.append({"kind": "function", "target": head[:-2], "pack": False, "src": src})
        elif re.fullmatch(r"[\w.]+", head):
            nodes.append({"kind": "module", "target": head, "pack": False, "src": src})
        else:
            raise HierError(f"cannot read node {head!r}")
    return nodes


def from_text(text: str) -> dict:
    """Parse a reply. Raises HierError on lines it cannot read."""
    lines = [l for l in text.strip().splitlines() if l.strip() and not l.strip().startswith("```")]
    if not lines or lines[0].strip() != "TREE":
        raise HierError("reply must start with TREE")
    i = 1
    tree_lines = []
    while i < len(lines) and not lines[i].startswith("GRAPH"):
        tree_lines.append(lines[i])
        i += 1
    variants = {}
    while i < len(lines):
        m = re.fullmatch(r"GRAPH\s+(\S+)", lines[i].strip())
        if not m:
            raise HierError(f"expected GRAPH, got {lines[i]!r}")
        j = i + 1
        while j < len(lines) and not lines[j].startswith("GRAPH"):
            j += 1
        variants[m.group(1)] = {"nodes": _parse_graph(lines[i + 1:j]), "hidden": False}
        i = j
    return {"tree": _parse_tree(tree_lines), "variants": variants, "problems": []}


# --------------------------------------------------------------------------
# inline: tree + graphs -> the flat graph
# --------------------------------------------------------------------------
def _child(node: dict, rel_path: str) -> dict | None:
    cur = node
    for part in rel_path.split("."):
        nxt = next((c for c in cur["children"] if c["name"] == part), None)
        if nxt is None:
            return None
        cur = nxt
    return cur


def inline(h: dict, ref: dict | None = None) -> dict:
    """Expand a hierarchical answer into the flat graph schema (shapes left empty).

    `ref` is the truth: its hidden insides of opaque modules are used for any opaque box
    in `h` at the same path, since the code never showed them. Raises HierError when the
    answer does not hold together (missing child, unknown variant, bad reference)."""
    flat_nodes, flat_edges = [], []
    budget = [20000]

    def emit(op, target, cls, params, refs) -> int:
        budget[0] -= 1
        if budget[0] < 0:
            raise HierError("expansion too large (recursive answer?)")
        nid = len(flat_nodes)
        flat_nodes.append({"id": nid, "op": op, "target": target, "cls": cls, "params": params, "shape": None})
        slots: dict[int, list[int]] = {}
        for s, r in enumerate(refs):
            slots.setdefault(r, []).append(s)
        for src, ss in sorted(slots.items()):
            flat_edges.append({"src": src, "dst": nid, "args": ss})
        return nid

    def module(node: dict, path: str, refs: list[int], variants: dict, depth: int) -> list[int]:
        if depth > 60:
            raise HierError("nesting too deep")
        kind = node["kind"]
        if kind == "opaque" and node.get("tag") in variants:          # truth's own hidden insides
            return graph(variants[node["tag"]]["nodes"], node, path, refs, variants, depth + 1)
        if kind == "opaque" and ref is not None:
            rnode = _child(ref["tree"], path) if path else None
            if rnode is not None and rnode["kind"] == "opaque" and rnode["tag"]:
                return graph(ref["variants"][rnode["tag"]]["nodes"], rnode, path, refs, ref["variants"], depth + 1)
        if kind in ("leaf", "opaque"):
            return [emit("call_module", path, node["cls"], node["params"], refs)]
        if kind == "chain":
            vals = refs
            for c in node["children"]:
                vals = module(c, f"{path}.{c['name']}" if path else c["name"], vals, variants, depth + 1)
            return vals
        if kind == "block":
            if node["tag"] not in variants:
                raise HierError(f"no GRAPH for {node['tag']} ({path or 'root'})")
            return graph(variants[node["tag"]]["nodes"], node, path, refs, variants, depth + 1)
        raise HierError(f"{path}: a {kind} cannot be called")

    def graph(nodes, tnode, path, refs, variants, depth) -> list[int]:
        inputs = [n for n in nodes if n["kind"] == "input"]
        n_fixed = sum(1 for n in inputs if not n["pack"])
        if sum(1 for n in inputs if n["pack"]) > 1:
            raise HierError(f"{path}: more than one list input")
        spare = len(refs) - n_fixed
        if spare < 0 or (spare > 0 and n_fixed == len(inputs)):
            raise HierError(f"{path or 'root'}: given {len(refs)} inputs, graph takes {len(inputs)}")
        values: list[list[int]] = []
        k = 0
        for n in nodes:
            if n["kind"] != "input":
                break
            take = spare if n["pack"] else 1
            values.append(refs[k:k + take])
            k += take
        prefix = (path + ".") if path else ""

        def resolve(tokens) -> list[int]:
            out = []
            for t in tokens:
                m = re.fullmatch(r"(\d+)(\*|\[(\d+)\])?", t)
                u = int(m.group(1))
                if u >= len(values):
                    raise HierError(f"{path or 'root'}: source {t} is not defined before use")
                if m.group(3) is not None:
                    idx = int(m.group(3))
                    if idx >= len(values[u]):
                        raise HierError(f"{path or 'root'}: {t} has no element {idx}")
                    out.append(values[u][idx])
                else:
                    out.extend(values[u])
            return out

        for n in nodes[len(values):]:
            src = resolve(n["src"])
            if n["kind"] == "output":
                return src
            if n["kind"] == "input":
                raise HierError(f"{path or 'root'}: input after other nodes")
            if n["kind"] == "module":
                child = _child(tnode, n["target"])
                if child is None:
                    raise HierError(f"{path or 'root'}: no child {n['target']} in the tree")
                values.append(module(child, prefix + n["target"], src, variants, depth))
            elif n["kind"] == "get_attr":
                values.append([emit("get_attr", prefix + n["target"], None, {}, src)])
            else:
                op = "call_function" if n["kind"] == "function" else "call_method"
                values.append([emit(op, n["target"], None, {}, src)])
        raise HierError(f"{path or 'root'}: graph has no output")

    root = h["tree"]
    if root["kind"] == "block":
        if root["tag"] not in h["variants"]:
            raise HierError(f"no GRAPH for the root {root['tag']}")
        rnodes = h["variants"][root["tag"]]["nodes"]
        ins = [n for n in rnodes if n["kind"] == "input"]
    else:
        ins = [{"target": "input"}]
    refs = [emit("placeholder", n["target"], None, {}, []) for n in ins]
    if root["kind"] == "block":
        outs = graph(rnodes, root, "", refs, h["variants"], 0)
    else:
        outs = module(root, "", refs, h["variants"], 0)
    emit("output", "output", None, {}, outs)
    return {"num_inputs": len(refs), "input_shapes": [None] * len(refs), "shapes_available": False,
            "nodes": flat_nodes, "edges": flat_edges}


def same_graph(a: dict, b: dict) -> bool:
    from .metrics import _isomorphic
    from .validate import node_label
    return _isomorphic(a, b, node_label)


def check_lossless(h: dict, flat: dict) -> list[str]:
    """Problems with h as a description of flat: extraction notes, inline, text round trip."""
    issues = list(h["problems"])
    try:
        if not same_graph(inline(h), flat):
            issues.append("inline(truth) differs from the fx graph")
    except HierError as e:
        issues.append(f"inline(truth) failed: {e}")
    try:
        back = from_text(to_text(h))
        if not same_graph(inline(back, ref=h), flat):
            issues.append("inline(from_text(to_text)) differs from the fx graph")
    except HierError as e:
        issues.append(f"text round trip failed: {e}")
    return issues

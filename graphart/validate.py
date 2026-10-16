"""Structural checks: is this (possibly model-written) graph a well-formed forward pass?

Everything here is decided without looking at the ground truth, which is what
makes `valid` a different measurement from `exact`. A graph can be perfectly
well-formed and still be the wrong network; it can also be a tidy DAG and still
be impossible as a neural network, which the shape checks catch.

The circuit domain's defining invariant -- the nodes on a wire form one chain --
has no analogue here. Its replacements are arity (a Linear takes one input, an
add takes two) and shape consistency, which together play the same role: laws
an answer must obey regardless of what the right answer is.
"""

from collections import Counter

import networkx as nx

# Operations whose input count we know. Anything absent is left unchecked
# rather than guessed at -- a wrong arity rule would fail correct answers.
# view/reshape/getitem/permute/transpose are absent on purpose: their shape
# arguments can themselves be traced nodes, so arity is genuinely variable.
# ShuffleNet's view takes one tensor plus four size-derived scalars.
UNARY = {
    "flatten", "mean", "size",
    "contiguous", "torch.relu", "torch.sigmoid", "torch.tanh",
    "torch.flatten", "F.relu", "F.gelu", "F.sigmoid", "F.softmax", "F.dropout",
    "F.adaptive_avg_pool2d", "F.max_pool2d", "F.avg_pool2d", "F.hardswish",
    "F.hardsigmoid", "F.silu",
}
BINARY = {"add", "sub", "mul", "div", "matmul"}
# DenseNet concatenates a single-element list, so the floor is one, not two.
VARIADIC = {"cat", "stack"}

# Layers that take exactly one tensor. Attention and the like are deliberately
# absent, because they do not.
SINGLE_INPUT_LAYERS = (
    "Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "GroupNorm", "LayerNorm",
    "ReLU", "ReLU6", "GELU", "SiLU", "Sigmoid", "Tanh", "Hardswish", "Hardsigmoid",
    "Dropout", "Flatten", "Embedding", "Identity",
    "MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveMaxPool2d",
)


def _hashable(value):
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


def node_label(node: dict) -> tuple:
    """What `exact` compares.

    The dotted module path is included: it is deterministic from the source and
    carries the hierarchy the project is asked to recover, so renaming a layer
    is a real difference. The shape is NOT included -- shape inference is a
    separate skill from topology, and folding it in here would let one
    miscomputed spatial dimension zero out an otherwise perfect graph. Shapes
    are scored on their own in `metrics`.
    """
    return (node["op"], node["target"], node["cls"],
            tuple(sorted((k, _hashable(v)) for k, v in node["params"].items())))


def edge_label(edge: dict) -> tuple:
    return tuple(sorted(edge["args"]))


def to_networkx(graph: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in graph["nodes"]:
        g.add_node(n["id"], label=node_label(n), **n)
    for e in graph["edges"]:
        g.add_edge(e["src"], e["dst"], label=edge_label(e))
    return g


def _expected_arity(node: dict) -> tuple[int, int] | None:
    """(minimum, maximum) tensor inputs, or None when we do not know."""
    op, target, cls = node["op"], node["target"], node["cls"]
    if op in ("placeholder", "get_attr"):
        return (0, 0)
    if op == "output":
        return (1, 10 ** 6)
    if op == "call_module":
        return (1, 1) if cls in SINGLE_INPUT_LAYERS else None
    if target in UNARY:
        return (1, 1)
    if target in BINARY:
        return (2, 2)
    if target in VARIADIC:
        return (1, 10 ** 6)
    return None


def _broadcastable(a: list[int], b: list[int]) -> bool:
    for x, y in zip(reversed(a), reversed(b)):
        if x != y and x != 1 and y != 1:
            return False
    return True


def _plain_shape(shape):
    """A flat list of ints, or None for absent and multi-output shapes."""
    if shape is None or not shape or isinstance(shape[0], (list, type(None))):
        return None
    return shape


def _check_shapes(graph: dict, by_id: dict, inputs: dict) -> list[str]:
    """Is what the graph declares about its own shapes internally consistent?

    Deliberately partial: channel and feature dimensions, plus elementwise and
    concat agreement. Spatial arithmetic (what a stride-2 7x7 convolution does
    to 224) is not checked yet -- it is a lot of code for errors we have not
    yet seen a model make. Everything here is skipped when a shape is absent.
    """
    errors = []
    for node in graph["nodes"]:
        own = _plain_shape(node["shape"])
        srcs = [_plain_shape(by_id[s]["shape"]) for s in inputs.get(node["id"], [])
                if s in by_id]
        params, cls = node["params"], node["cls"]

        if cls == "Linear" and srcs and srcs[0] and "in" in params:
            if srcs[0][-1] != params["in"]:
                errors.append(f"node {node['id']} Linear expects in={params['in']} but "
                              f"its input ends in {srcs[0][-1]}")
            if own and own[-1] != params.get("out"):
                errors.append(f"node {node['id']} Linear declares out={params.get('out')} "
                              f"but its own shape ends in {own[-1]}")
        elif cls and cls.startswith("Conv") and srcs and srcs[0] and len(srcs[0]) > 1:
            if "in" in params and srcs[0][1] != params["in"]:
                errors.append(f"node {node['id']} {cls} expects in={params['in']} channels "
                              f"but its input has {srcs[0][1]}")
            if own and len(own) > 1 and own[1] != params.get("out"):
                errors.append(f"node {node['id']} {cls} declares out={params.get('out')} "
                              f"but its own shape has {own[1]} channels")
        elif node["target"] in BINARY and len(srcs) >= 2 and all(srcs):
            if not _broadcastable(srcs[0], srcs[1]):
                errors.append(f"node {node['id']} {node['target']} combines incompatible "
                              f"shapes {srcs[0]} and {srcs[1]}")
        elif node["target"] == "cat" and len(srcs) >= 2 and all(srcs):
            widths = {len(s) for s in srcs}
            if len(widths) > 1:
                errors.append(f"node {node['id']} cat mixes shapes of different rank: {srcs}")
            else:
                differing = [i for i in range(len(srcs[0]))
                             if len({s[i] for s in srcs}) > 1]
                if len(differing) > 1:
                    errors.append(f"node {node['id']} cat inputs differ on more than one "
                                  f"axis {differing}: {srcs}")
    return errors


def check(graph: dict, *, shapes: bool = True) -> list[str]:
    """Return a list of human-readable problems (empty list == valid).

    `shapes=False` reports structural problems only. `metrics` uses it to gate
    `exact`, so that a graph with perfect topology and one bad dimension is
    still scored as topologically exact -- shape inference is measured on its
    own and must not silently sink the result the project is actually about.
    """
    errors = []
    nodes = graph["nodes"]

    ids = [n["id"] for n in nodes]
    dupes = sorted(i for i, c in Counter(ids).items() if c > 1)
    if dupes:
        errors.append(f"duplicate node ids: {dupes}")
    by_id = {n["id"]: n for n in nodes}
    if not nodes:
        return errors + ["graph has no nodes"]

    # --- edges refer to real nodes, once each, and fill distinct slots -------
    inputs: dict[int, list[int]] = {}
    outgoing: Counter = Counter()
    seen_pairs: Counter = Counter()
    for e in graph["edges"]:
        s, d = e["src"], e["dst"]
        if s not in by_id or d not in by_id:
            errors.append(f"edge {s}->{d} references a missing node")
            continue
        if s == d:
            errors.append(f"self-loop on node {s}")
        seen_pairs[(s, d)] += 1
        inputs.setdefault(d, []).extend([s] * len(e["args"]))
        outgoing[s] += 1
    for (s, d), count in seen_pairs.items():
        if count > 1:
            errors.append(f"edge {s}->{d} is listed {count} times; one edge should carry "
                          f"every slot it fills")
    for d, _ in inputs.items():
        slots = [a for e in graph["edges"] if e["dst"] == d for a in e["args"]]
        repeated = sorted(a for a, c in Counter(slots).items() if c > 1)
        if repeated:
            errors.append(f"node {d} has more than one input in slot(s) {repeated}")

    # --- the endpoints of a forward pass ------------------------------------
    placeholders = [n for n in nodes if n["op"] == "placeholder"]
    outs = [n for n in nodes if n["op"] == "output"]
    if not placeholders:
        errors.append("graph has no placeholder (input) node")
    if len(outs) != 1:
        errors.append(f"a forward pass has exactly one output node, found {len(outs)}")
    for n in outs:
        if outgoing[n["id"]]:
            errors.append(f"output node {n['id']} has outgoing edges")
    for n in placeholders:
        if inputs.get(n["id"]):
            errors.append(f"placeholder {n['id']} has incoming edges")

    # --- how many inputs each operation may take ----------------------------
    for n in nodes:
        bounds = _expected_arity(n)
        if bounds is None:
            continue
        got = len(inputs.get(n["id"], []))
        low, high = bounds
        if not low <= got <= high:
            want = str(low) if low == high else f"{low}..{'many' if high > 99 else high}"
            errors.append(f"node {n['id']} ({n['cls'] or n['target']}) takes {want} "
                          f"input(s) but has {got}")

    # --- module hierarchy ---------------------------------------------------
    # A traced leaf cannot contain another traced leaf, so no module path may
    # be a strict prefix of another.
    paths = sorted({n["target"] for n in nodes if n["op"] == "call_module"})
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            if b.startswith(a + "."):
                errors.append(f"module {b!r} is nested inside {a!r}, but both are nodes")

    # --- acyclicity and reachability ----------------------------------------
    g = to_networkx(graph)
    if not nx.is_directed_acyclic_graph(g):
        errors.append("graph has a cycle")
    elif placeholders:
        # Every node must come from somewhere: a placeholder or a get_attr
        # (a parameter or buffer, such as ViT's class_token).
        #
        # The converse -- that every node must also reach the output -- is NOT
        # checked, because it is not true. ViT traces torch._assert chains that
        # exist for their side effect and feed nothing. Requiring it would fail
        # correct answers, which is the one thing a validator must never do.
        roots = [n["id"] for n in nodes if n["op"] in ("placeholder", "get_attr")]
        reachable = set(roots)
        for r in roots:
            reachable |= nx.descendants(g, r)
        stranded = sorted(set(by_id) - reachable)
        if stranded:
            errors.append(f"node(s) {stranded} are not reachable from any input or "
                          f"parameter")

    if shapes:
        errors.extend(_check_shapes(graph, by_id, inputs))
    return errors

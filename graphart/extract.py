"""Ground truth: trace an nn.Module's forward pass and convert it to our schema.

`torch.fx.symbolic_trace` plays the role `circuit_to_dag` played in the Qiskit
domain -- it is PyTorch's own representation of the forward pass, not something
we invented, so the labels are exact and free.

Two things need care, and both exist because fx records *how the code was
written* rather than what it computes:

1. Node names. fx auto-generates `conv1`, `relu_2`, `add_3`. Those vary with
   source ordering, so they never enter a node's label -- exactly as circuit
   node ids never did. What does enter the label is the module's dotted path
   (`layer1.0.conv1`), which is deterministic from the code and carries the
   hierarchy the project is asked to recover.

2. Operator spelling. `x + y`, `torch.add(x, y)` and `x.add(y)` trace to three
   different targets for identical semantics. Left alone, the task would be
   partly a test of whether the model memorised fx trivia. ALIASES normalises
   them.
"""

import operator
from typing import Any

import torch
import torch.fx

PARAM_DECIMALS = 3

# Semantically identical operations that fx spells differently depending on
# whether the source used an operator, a torch function or a tensor method.
ALIASES = {
    "operator.add": "add", "torch.add": "add", "method.add": "add",
    "operator.sub": "sub", "torch.sub": "sub", "method.sub": "sub",
    "operator.mul": "mul", "torch.mul": "mul", "method.mul": "mul",
    "operator.truediv": "div", "torch.div": "div", "method.div": "div",
    "operator.matmul": "matmul", "torch.matmul": "matmul",
    "operator.getitem": "getitem",
    "torch.cat": "cat", "torch.concat": "cat",
    # Functional activations spelled three ways. The module forms (nn.ReLU) stay
    # distinct on purpose -- those are layers, with a path and a place in the
    # hierarchy, which is part of what the project asks the model to recover.
    "torch.relu": "relu", "F.relu": "relu", "method.relu": "relu",
    "torch.sigmoid": "sigmoid", "F.sigmoid": "sigmoid", "method.sigmoid": "sigmoid",
    "torch.tanh": "tanh", "F.tanh": "tanh", "method.tanh": "tanh",
    "F.gelu": "gelu", "F.silu": "silu", "F.softmax": "softmax", "torch.softmax": "softmax",
    "torch.flatten": "flatten", "method.flatten": "flatten",
    "method.view": "view", "method.reshape": "reshape",
    "method.permute": "permute", "method.transpose": "transpose",
    "method.mean": "mean", "method.size": "size", "method.contiguous": "contiguous",
}

# Compact, canonical hyperparameters per layer type. Anything not listed falls
# back to a trimmed repr, so an unknown layer still gets a stable label.
def _layer_params(m: torch.nn.Module) -> dict[str, Any]:
    name = type(m).__name__
    if name == "Linear":
        return {"in": m.in_features, "out": m.out_features, "bias": m.bias is not None}
    if name.startswith(("Conv", "ConvTranspose")):
        p = {"in": m.in_channels, "out": m.out_channels, "k": _one(m.kernel_size),
             "s": _one(m.stride), "p": _one(m.padding)}
        if getattr(m, "groups", 1) != 1:
            p["g"] = m.groups
        return p
    if name.startswith("BatchNorm") or name == "GroupNorm":
        return {"features": getattr(m, "num_features", None) or m.num_channels}
    if name == "LayerNorm":
        return {"shape": list(m.normalized_shape)}
    if name.startswith(("MaxPool", "AvgPool")):
        return {"k": _one(m.kernel_size), "s": _one(m.stride)}
    if name.startswith("AdaptiveAvgPool") or name.startswith("AdaptiveMaxPool"):
        return {"out": _one(m.output_size)}
    if name == "Dropout":
        return {"p": round(float(m.p), PARAM_DECIMALS)}
    if name == "Embedding":
        return {"num": m.num_embeddings, "dim": m.embedding_dim}
    if name == "Flatten":
        return {"start": m.start_dim, "end": m.end_dim}
    return {}


def _shape_of(meta):
    """A node's output shape, or None when it has no single tensor shape."""
    if meta is None:
        return None
    if hasattr(meta, "shape"):
        return list(meta.shape)
    if isinstance(meta, (tuple, list)):          # multi-output node, e.g. chunk
        return [_shape_of(m) for m in meta]
    return None


def _one(value):
    """kernel_size=(3, 3) -> 3, but keep genuinely asymmetric values as a list."""
    if isinstance(value, (tuple, list)):
        return value[0] if len(set(value)) == 1 else list(value)
    return value


def _target_name(node: torch.fx.Node) -> str:
    """A stable, normalised name for what this node does."""
    if node.op == "call_function":
        target = node.target
        module = getattr(target, "__module__", "") or ""
        short = module.split(".")[-1]
        # Builtin operators report __module__ as "_operator", not "operator".
        if short == "_operator":
            short = "operator"
        raw = f"{short}.{getattr(target, '__name__', str(target))}"
        # Some functional ops are implemented in C and report torch._C._nn
        # rather than torch.nn.functional -- gelu and linear among them. Both
        # are F.<name> as far as the source is concerned, so normalise on the
        # module rather than listing the exceptions one at a time.
        if module.startswith(("torch.nn.functional", "torch._C._nn")):
            raw = f"F.{getattr(target, '__name__', target)}"
        return ALIASES.get(raw, raw)
    if node.op == "call_method":
        return ALIASES.get(f"method.{node.target}", str(node.target))
    return str(node.target)


class GranularTracer(torch.fx.Tracer):
    """Trace with a caller-supplied notion of what counts as a leaf.

    Measured on torchvision: coarsening by module depth shrinks a graph a lot
    but flattens it into a chain -- ResNet-18 drops from 71 nodes with 8
    residual merges to 17 nodes with none, because the skip connection lives
    inside the block. So depth coarsening is NOT a way to keep topology while
    reducing size; for that, contract chains after tracing instead.
    """

    def __init__(self, leaf_depth: int | None = None):
        super().__init__()
        self.leaf_depth = leaf_depth

    def is_leaf_module(self, m: torch.nn.Module, qualname: str) -> bool:
        if super().is_leaf_module(m, qualname):
            return True
        if self.leaf_depth is None:
            return False
        has_children = any(True for _ in m.children())
        depth = 0 if not qualname else qualname.count(".") + 1
        return has_children and depth >= self.leaf_depth


def _input_slots(node: torch.fx.Node, order: dict) -> dict[int, list[int]]:
    """Which argument slots each input node fills, as {src id: [slot, ...]}.

    `node.all_input_nodes` cannot be used for this: it deduplicates, so
    `attn(x, x, x)` reports one input rather than three and `y + y` reports one
    operand rather than two. Recording the real slots matters because the truth
    would otherwise say an attention node has a single input, and a model that
    correctly produced three would be scored wrong.

    Slots count tensor arguments only -- `cat([a, b], dim=1)` has two, not
    three -- which keeps the index meaningful and matches what a reader of the
    source would say.
    """
    slots: dict[int, list[int]] = {}
    position = 0

    def walk(arg) -> None:
        nonlocal position
        if isinstance(arg, torch.fx.Node):
            slots.setdefault(order[arg], []).append(position)
            position += 1
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                walk(item)
        elif isinstance(arg, dict):
            for item in arg.values():
                walk(item)

    for arg in node.args:
        walk(arg)
    for arg in node.kwargs.values():
        walk(arg)
    return slots


def graph_from_module(model: torch.nn.Module, *, example_input: torch.Tensor | None = None,
                      leaf_depth: int | None = None) -> dict:
    """Trace `model` and return the forward pass in our graph schema."""
    model = model.eval()
    graph = GranularTracer(leaf_depth).trace(model)
    modules = dict(model.named_modules())

    # Shapes are a bonus, not a requirement. ShapeProp itself succeeded on all
    # 15 torchvision architectures measured, including the ones with nodes that
    # return tuples (chunk/split in ShuffleNet, attention in ViT) -- reading
    # those metas is what needs care, which `_shape_of` handles. It can still
    # fail in general on data-dependent shapes, and losing shapes must not cost
    # us the graph, so this stays best-effort and reports via
    # `shapes_available`.
    shapes_available = False
    if example_input is not None:
        from torch.fx.passes.shape_prop import ShapeProp
        try:
            ShapeProp(torch.fx.GraphModule(model, graph)).propagate(example_input)
            shapes_available = True
        except Exception:
            shapes_available = False

    order = {node: i for i, node in enumerate(graph.nodes)}
    nodes, edges = [], []
    for node, idx in order.items():
        entry: dict[str, Any] = {"id": idx, "op": node.op, "target": _target_name(node)}
        if node.op == "call_module":
            m = modules[node.target]
            entry["cls"] = type(m).__name__
            entry["params"] = _layer_params(m)
        else:
            entry["cls"] = None
            entry["params"] = {}
        entry["shape"] = _shape_of(node.meta.get("tensor_meta"))
        nodes.append(entry)

        # Argument position matters: `a - b` and `cat([a, b])` are order
        # sensitive, so the edge carries its slots the way circuit edges
        # carried their wires -- including the merge of parallel edges into
        # one, for the same reason.
        for src, slots in sorted(_input_slots(node, order).items()):
            edges.append({"src": src, "dst": idx, "args": slots})

    return {"num_inputs": sum(1 for n in nodes if n["op"] == "placeholder"),
            "input_shapes": [n["shape"] for n in nodes if n["op"] == "placeholder"],
            "shapes_available": shapes_available,
            "nodes": nodes, "edges": edges}


def graph_from_code(code: str, var: str = "model",
                    example_input: torch.Tensor | None = None,
                    leaf_depth: int | None = None) -> tuple[dict, torch.nn.Module]:
    """Execute generated source and trace the nn.Module it defines in `var`.

    If the source also binds `example_input`, that is used for shape
    propagation unless a tensor is passed explicitly. Generated samples carry
    their own input that way, which they must: a model asked to predict shapes
    cannot do it without being told what goes in.
    """
    namespace: dict = {}
    exec(code, namespace)
    model = namespace.get(var)
    if not isinstance(model, torch.nn.Module):
        # The variable is not always called `model`: generated samples vary it
        # on purpose, and real code calls it whatever it likes. Fall back to
        # the module instance the source left behind, when there is just one.
        found = [v for k, v in namespace.items()
                 if isinstance(v, torch.nn.Module) and not k.startswith("_")]
        if len(found) != 1:
            raise ValueError(
                f"expected one nn.Module in the source (named {var!r} or otherwise), "
                f"found {len(found)}")
        model = found[0]
    if example_input is None:
        candidate = namespace.get("example_input")
        if isinstance(candidate, torch.Tensor):
            example_input = candidate
    return graph_from_module(model, example_input=example_input,
                             leaf_depth=leaf_depth), model

"""Per-instance module trees (22 Nov): the tree task split into one call per class instance.

The whole-tree task asks a model to run every constructor in its head at once: loops over
config tables, width arithmetic, conditions, nested builders. Here each call covers ONE
instance of a class defined in the code, with the arguments its constructor receives:

    INSTANCE layer1.0: Bottleneck(inplanes=64, planes=64, downsample=Sequential(0=Conv2d(...), ...))
 -> conv1: Conv2d(in=64,out=64,k=1,s=1,p=0)
    ...
    downsample: Sequential
      0: Conv2d(in=64,out=256,k=1,s=1,p=0)
      1: BatchNorm2d(features=256)

Children that are themselves instances of classes in the code are written as
`name: Class(arg=value, ...)` and get their own call; torch containers (Sequential,
ModuleList, ModuleDict) are opened in place; a class whose code is not shown is written by
name. The root call (the module the program builds) gets the whole reduced program and
writes its own children the same way. Code joins the answers into the hierarchical tree;
the variant tags then come from the wiring graphs (two instances of a class share a tag when
their graphs are the same), as graphart/hier.py numbers them.

The arguments are recorded while the program is built (`capture`), for the training
targets only: a wrapper is put on each class's own __init__ as the class is defined, and the
first (outermost) call for an object records what it was passed. Arguments equal to their
default are left out. The model itself never runs anything.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import functools
import inspect
import re

import torch
from torch import nn

from .extract import _layer_params
from .hier import SRC_MODULE, HierError, _expand_ranges, _finish_kinds, _fmt_params
from .schema import _parse_params
from .hier_tasks import class_source

INST_PROMPT = "hier_inst"
ROOT_LABEL = "model (the module the code builds)"
EMPTY = "(no child modules)"
MAX_DEPTH = 3


# --------------------------------------------------------------------------
# recording constructor arguments
# --------------------------------------------------------------------------
def capture(code: str, var: str = "model") -> tuple[nn.Module, dict[int, tuple]]:
    """Execute the program with every class's own __init__ wrapped; returns (model,
    {id(module): (module, init function, args, kwargs)})."""
    records: dict[int, tuple] = {}
    keep = []                                  # objects stay alive, so ids stay unique

    def wrap(cls):
        orig = cls.__dict__.get("__init__")
        if orig is None or getattr(orig, "_graphart_wrapped", False):
            return

        @functools.wraps(orig)
        def init(self, *a, **k):
            if id(self) not in records:
                records[id(self)] = (self, orig, a, k)
                keep.append(self)
            return orig(self, *a, **k)
        init._graphart_wrapped = True
        cls.__init__ = init

    old = nn.Module.__dict__.get("__init_subclass__")

    def hook(cls, **kw):
        if old is not None:
            old.__func__(cls, **kw) if hasattr(old, "__func__") else None
        if cls.__module__ == SRC_MODULE:
            wrap(cls)
    nn.Module.__init_subclass__ = classmethod(hook)
    try:
        namespace: dict = {"__name__": SRC_MODULE}
        torch.manual_seed(0)
        exec(code, namespace)
    finally:
        if old is None:
            del nn.Module.__init_subclass__
        else:
            nn.Module.__init_subclass__ = old
    model = namespace.get(var)
    if not isinstance(model, nn.Module):
        found = [v for k, v in namespace.items() if isinstance(v, nn.Module) and not k.startswith("_")]
        if len(found) != 1:
            raise ValueError(f"expected one nn.Module in the source, found {len(found)}")
        model = found[0]
    return model, records


def capture_timm(name: str) -> tuple[nn.Module, dict[int, tuple]]:
    """The same for a timm model (its assembled code has relative imports, so it is built
    from timm itself): every timm.models class's own __init__ is wrapped for the build."""
    import timm.models
    from timm.models import model_entrypoint
    records: dict[int, tuple] = {}
    keep, undo = [], []

    def subclasses(c):
        for s in c.__subclasses__():
            yield s
            yield from subclasses(s)
    for cls in set(subclasses(nn.Module)):
        if not cls.__module__.startswith("timm.models") or "__init__" not in cls.__dict__:
            continue
        orig = cls.__dict__["__init__"]

        def make(orig):
            @functools.wraps(orig)
            def init(self, *a, **k):
                if id(self) not in records:
                    records[id(self)] = (self, orig, a, k)
                    keep.append(self)
                return orig(self, *a, **k)
            return init
        cls.__init__ = make(orig)
        undo.append((cls, orig))
    try:
        torch.manual_seed(0)
        model = model_entrypoint(name)()
    finally:
        for cls, orig in undo:
            cls.__init__ = orig
    return model, records


def _bound(rec) -> dict:
    """The arguments a constructor was passed, by parameter name; defaults left out."""
    self, fn, a, k = rec
    sig = inspect.signature(fn)
    try:
        b = sig.bind(self, *a, **k)
    except TypeError:
        return {f"arg{i}": v for i, v in enumerate(a)} | k
    out = {}
    params = list(sig.parameters.values())
    for name, value in list(b.arguments.items())[1:]:
        p = sig.parameters[name]
        if p.kind is p.VAR_POSITIONAL:
            for i, v in enumerate(value):
                out[f"*{i}"] = v
        elif p.kind is p.VAR_KEYWORD:
            out.update(value)
        else:
            if p.default is not p.empty:
                try:
                    if value is p.default or (type(value) is type(p.default) and value == p.default):
                        continue
                except Exception:
                    pass
            out[name] = value
    del params
    return out


def render(v, records: dict, depth: int = 0) -> str:
    """A constructor argument as text: literals as written, classes and functions by name,
    modules by their contents, other objects by their fields."""
    if depth > MAX_DEPTH:
        return "..."
    r = lambda x: render(x, records, depth + 1)
    if v is None or isinstance(v, (bool, int, str)):
        return repr(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, enum.Enum):
        return f"{type(v).__name__}.{v.name}"
    if isinstance(v, list):
        return "[" + ", ".join(r(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(r(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {r(x)}" for k, x in v.items()) + "}"
    if isinstance(v, nn.Module):
        return module_desc(v, records, depth)
    if isinstance(v, type):
        return v.__name__
    if isinstance(v, functools.partial):
        parts = [r(v.func)] + [r(x) for x in v.args] + [f"{k}={r(x)}" for k, x in v.keywords.items()]
        return "partial(" + ", ".join(parts) + ")"
    if isinstance(v, torch.Tensor):
        return f"tensor{list(v.shape)}"
    if callable(v):
        return getattr(v, "__name__", type(v).__name__)
    if dataclasses.is_dataclass(v):
        fields = {f.name: getattr(v, f.name) for f in dataclasses.fields(v)}
    elif hasattr(v, "__dict__"):
        fields = {k: x for k, x in vars(v).items() if not k.startswith("_")}
    else:
        return type(v).__name__
    return type(v).__name__ + "(" + ", ".join(f"{k}={r(x)}" for k, x in fields.items()) + ")"


def args_text(m: nn.Module, records: dict, depth: int = 0) -> str:
    rec = records.get(id(m))
    if rec is None:
        return ""
    return ", ".join(f"{k}={render(v, records, depth + 1)}" if not k.startswith("*")
                     else render(v, records, depth + 1) for k, v in _bound(rec).items())


def module_desc(m: nn.Module, records: dict, depth: int = 0) -> str:
    """A module passed as an argument, written inline."""
    cls = type(m)
    if id(m) in records:                                  # a class in the code
        return f"{cls.__name__}({args_text(m, records, depth)})"
    kids = list(m.named_children())
    if not kids:
        return cls.__name__ + _fmt_params(_layer_params(m))
    return cls.__name__ + "(" + ", ".join(f"{n}={module_desc(c, records, depth + 1)}" for n, c in kids) + ")"


# --------------------------------------------------------------------------
# one instance's answer
# --------------------------------------------------------------------------
def _child_desc(m: nn.Module, records: dict) -> tuple[str, bool]:
    """(how a child is written, whether its own children follow in place)."""
    if id(m) in records:                                  # its own call
        return f"{type(m).__name__}({args_text(m, records)})", False
    if type(m).__module__.startswith("torch.nn"):
        if isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            return type(m).__name__, True
        if not list(m.children()):
            return type(m).__name__ + _fmt_params(_layer_params(m)), False
    return type(m).__name__, False                         # code not shown: by name


def instance_lines(m: nn.Module, records: dict, depth: int = 0) -> list[str]:
    kids = list(m.named_children())
    lines, i = [], 0
    while i < len(kids):
        name, c = kids[i]
        desc, opens = _child_desc(c, records)
        body = instance_lines(c, records, depth + 1) if opens else []
        j = i + 1
        if name.isdigit():
            while (j < len(kids) and kids[j][0].isdigit() and int(kids[j][0]) == int(name) + (j - i)):
                d2, o2 = _child_desc(kids[j][1], records)
                if d2 != desc or (instance_lines(kids[j][1], records, depth + 1) if o2 else []) != body:
                    break
                j += 1
        label = name if j == i + 1 else f"{name}-{kids[j - 1][0]}"
        lines.append("  " * depth + f"{label}: {desc}")
        lines += body
        i = j
    return lines


def inst_source(reduced: str, cls: str) -> str:
    """The code a non-root call sees: the class with its bases and the functions it calls
    (hier_tasks.class_source), after the top-level constants and imports those use."""
    src = class_source(reduced, cls)
    if not src:
        return ""
    tree = ast.parse(reduced)
    used = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {t.id for tgt in targets for t in ast.walk(tgt) if isinstance(t, ast.Name)}
            if names & used:
                keep.append(ast.unparse(node))
    return ("\n".join(keep) + "\n\n\n" if keep else "") + src


def instances(model: nn.Module, records: dict) -> list[tuple[str, nn.Module]]:
    """(path, module) of the root and of every instance of a class in the code, pre-order,
    one per distinct (class, arguments): the first instance stands for the rest."""
    seen, out = set(), []

    def walk(m: nn.Module, path: str, is_call: bool):
        if is_call:
            key = (type(m).__qualname__, args_text(m, records)) if path else ("", "")
            if key not in seen:
                seen.add(key)
                out.append((path, m))
        for name, c in m.named_children():
            p = f"{path}.{name}" if path else name
            if id(c) in records:
                walk(c, p, True)
            elif _child_desc(c, records)[1]:             # a torch container, opened in place
                walk(c, p, False)
    walk(model, "", True)
    return out


def instance_examples(code: str, reduced: str, template: str, timm_name: str | None = None) -> list[dict]:
    """Every distinct instance of one program as {"path", "cls", "user", "answer"}."""
    from .hier_tasks import _fill
    model, records = capture_timm(timm_name) if timm_name else capture(code)
    out = []
    for path, m in instances(model, records):
        answer = "\n".join(instance_lines(m, records)) or EMPTY
        if not path:                                       # the root names its class first
            answer = type(m).__name__ + "\n" + answer
        if path:
            src = inst_source(reduced, type(m).__name__)
            if not src:
                continue
            head = f"{path}: {type(m).__name__}({args_text(m, records)})"
        else:
            src, head = reduced, ROOT_LABEL
        out.append({"path": path, "cls": type(m).__name__, "head": head.split(": ", 1)[1] if path else "",
                    "user": _fill(template, code=src.strip(), instance=head), "answer": answer})
    return out


# --------------------------------------------------------------------------
# joining the answers into one tree
# --------------------------------------------------------------------------
_LINE = re.compile(r"^(\s*)([\w.]+(?:-\d+)?):\s*(.+?)\s*$")
_CALL = re.compile(r"([A-Za-z_]\w*)\((.*)\)", re.S)
CONTAINERS = {"Sequential": "chain", "ModuleList": "list", "ModuleDict": "list"}


def answer_lines(text: str) -> list[tuple[int, str, str]]:
    """(indent, name, description) per line of one instance answer."""
    out = []
    for raw in text.strip().splitlines():
        if not raw.strip() or raw.strip().startswith("```") or raw.strip() == EMPTY:
            continue
        m = _LINE.match(raw.rstrip())
        if not m:
            raise HierError(f"cannot read instance line {raw!r}")
        out.append((len(m.group(1).expandtabs(2)), m.group(2), m.group(3)))
    return out


def assemble(root_answer: str, ask, max_calls: int = 400, max_depth: int = 30, defined=None) -> dict:
    """The module tree (hier.py's node format, no tags yet) from the root's answer;
    `ask(path, description)` returns the answer for an instance of a class in the code,
    called once per distinct description. `defined(cls)`, if given, says whether a class's
    code is shown: a `Class(args)` line for a class that is not becomes an opaque box."""
    answers: dict[str, str] = {}

    def node_for(name: str, desc: str, path: str, depth: int) -> dict:
        m = _CALL.fullmatch(desc)
        cls, inner = (m.group(1), m.group(2)) if m else (desc, None)
        node = {"name": name, "kind": None, "cls": cls, "params": {}, "tag": None, "hidden": False,
                "children": []}
        if cls in CONTAINERS:
            node["kind"] = CONTAINERS[cls]
        elif inner is not None and defined is not None and not hasattr(nn, cls) and not defined(cls):
            node["kind"] = "opaque"                                    # written with arguments, code not shown
        elif hasattr(nn, cls) or inner is None:
            if inner:
                try:
                    node["params"] = _parse_params(inner)
                except Exception as e:
                    raise HierError(f"{path}: {e}")
            node["kind"] = "leaf" if inner is not None else None      # bare names: _finish_kinds
        else:                                                          # a class in the code: its own call
            node["kind"] = "block"
            node["desc"] = desc                                        # equal descriptions, one call
            if desc not in answers:
                if len(answers) >= max_calls:
                    raise HierError(f"more than {max_calls} instance calls")
                answers[desc] = ask(path, desc)
            if depth > max_depth:
                raise HierError(f"{path}: instances nested deeper than {max_depth}")
            node["children"] = children(answers[desc], path, depth + 1)
        return node

    def children(text: str, path: str, depth: int) -> list[dict]:
        top, stack = [], []
        for indent, name, desc in answer_lines(text):
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else None
            if parent is not None and parent["kind"] not in ("chain", "list"):
                raise HierError(f"{path}: {name} is indented under a non-container")
            base = ".".join(x for x in (path, *(s[1]["name"] for s in stack), name.split("-")[0]) if x)
            node = node_for(name, desc, base, depth)
            (parent["children"] if parent is not None else top).append(node)
            stack.append((indent, node))
        return top

    lines = [l for l in root_answer.strip().splitlines() if l.strip() and not l.strip().startswith("```")]
    cls = ""
    if lines and re.fullmatch(r"[A-Za-z_]\w*", lines[0].strip()):
        cls, lines = lines[0].strip(), lines[1:]
    root = {"name": "", "kind": "block", "cls": cls, "params": {}, "tag": None, "hidden": False,
            "children": children("\n".join(lines), "", 0)}
    _expand_ranges(root)
    _finish_kinds(root)
    root["kind"] = "block"
    return root

"""Shrink model source before a language model reads it, without changing what it builds.

    from graphart.reduce import reduce_source
    short = reduce_source(code)

Used the same way in training, evaluation and general use: it is part of the pipeline, a
static rewrite of the text. It never runs the code, never decides which branch of an `if`
runs, and never looks at a trace -- that reasoning is the model's task, and fx is the
answer key. Every removal is one that cannot change the module tree or the forward pass:

  * docstrings and comments
  * type annotations (`x: Tensor`, `-> Tensor`, `norm_layer: Optional[Callable[..., nn.Module]]`);
    an annotated assignment keeps its value (`expansion: int = 4` -> `expansion = 4`)
  * methods nothing refers to: a class keeps __init__, forward, every dunder, and any method
    whose name is used anywhere in the remaining code (`self._forward_impl`, `cls.build`) --
    timm's group_matcher / reset_classifier / forward_intermediates and the like go
  * top-level functions and classes nothing refers to
  * weight initialisation and API logging, which set values, not structure: calls into
    `nn.init` / `init.` / `trunc_normal_` / `named_apply` / `self.apply`, and
    `_log_api_usage_once`; a loop or `if` left empty by that goes too

The result is re-printed by `ast.unparse`, so formatting is uniform (one statement per
line, no wrapped argument lists).
"""

import ast

KEEP_TOP = {"model", "example_input"}
INIT_CALLS = ("trunc_normal_", "named_apply", "_log_api_usage_once", "init_weights",
              "_init_weights", "lecun_normal_", "variance_scaling_")


def _call_name(node: ast.AST) -> str:
    """Dotted name of a call's function: nn.init.kaiming_normal_ -> 'nn.init.kaiming_normal_'."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_init_stmt(stmt: ast.stmt) -> bool:
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    name = _call_name(stmt.value.func)
    last = name.split(".")[-1]
    return (".init." in f".{name}." or name.startswith("init.") or last in INIT_CALLS
            or name in ("self.apply", "model.apply", "self.init_weights"))


class _Strip(ast.NodeTransformer):
    def _body(self, body: list) -> list:
        out = []
        for i, s in enumerate(body):
            if i == 0 and isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                continue                                   # docstring
            if _is_init_stmt(s):
                continue
            s = self.visit(s)
            if s is None:
                continue
            if isinstance(s, (ast.For, ast.If, ast.While, ast.With)) and not s.body and not getattr(s, "orelse", None):
                continue                                   # emptied by the removals
            out.extend(s if isinstance(s, list) else [s])
        return out

    def generic_visit(self, node):
        for field in ("body", "orelse", "finalbody"):
            if isinstance(getattr(node, field, None), list):
                setattr(node, field, self._body(getattr(node, field)))
        for field, value in ast.iter_fields(node):
            if field in ("body", "orelse", "finalbody"):
                continue
            if isinstance(value, list):
                setattr(node, field, [self.visit(v) if isinstance(v, ast.AST) else v for v in value])
            elif isinstance(value, ast.AST):
                setattr(node, field, self.visit(value))
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if not node.body and node.orelse:
            node.body = [ast.Pass()]
        return node

    def visit_For(self, node):
        return self.generic_visit(node)

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        node = self.generic_visit(node)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    in_dataclass = False

    def visit_ClassDef(self, node):
        # a dataclass's fields ARE its annotations: keep them there
        outer = self.in_dataclass
        self.in_dataclass = any("dataclass" in _call_name(d.func if isinstance(d, ast.Call) else d)
                                for d in node.decorator_list)
        node = self.generic_visit(node)
        self.in_dataclass = outer
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_AnnAssign(self, node):
        if self.in_dataclass:
            return node
        if node.value is None:
            return None                                    # a bare declaration
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def _used_names(tree: ast.AST) -> set[str]:
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier():
            names.add(n.value)                             # getattr(self, "name"), registries
    return names


def _prune(tree: ast.Module) -> bool:
    """One pass: drop unreferenced methods and top-level defs. True if anything went."""
    changed = False
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        others = ast.Module(body=[s for s in tree.body], type_ignores=[])
        keep = []
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = item.name
                if not (name.startswith("__") or name == "forward"):
                    # referenced anywhere except inside its own body?
                    saved, item.body = item.body, [ast.Pass()]
                    used = name in _used_names(others)
                    item.body = saved
                    if not used:
                        changed = True
                        continue
            keep.append(item)
        cls.body = keep or [ast.Pass()]
    used = _used_names(ast.Module(body=[s for s in tree.body if not isinstance(s, (ast.FunctionDef, ast.ClassDef))]
                                  + [b for s in tree.body if isinstance(s, (ast.FunctionDef, ast.ClassDef))
                                     for b in ([*s.bases, *s.decorator_list, *s.body] if isinstance(s, ast.ClassDef)
                                               else [*s.decorator_list, *s.body, s.args])], type_ignores=[]))
    body = []
    for s in tree.body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and s.name not in used \
                and s.name not in KEEP_TOP:
            changed = True
            continue
        body.append(s)
    tree.body = body
    return changed


def reduce_source(code: str, prune: bool = True) -> str:
    tree = ast.parse(code)
    tree.body = _Strip()._body(tree.body)
    if prune:
        for _ in range(10):
            if not _prune(tree):
                break
    return ast.unparse(tree) + "\n"

"""Build the controlled-edit evaluation set from the test programs (RQ5).

    python scripts/build_edit_set.py

Writes data/generated/edits.jsonl. Each sample is one test program with one small,
precise edit to its forward pass. The answer is always recomputed by tracing the
edited code with torch.fx, and each edit is classed by what it actually did to the
graph -- not by what it was meant to do:

  neutral      the edited program's graph is the original graph (labelled isomorphism)
  meaningful   it is a different graph

Edits meant to be neutral:  rename every local variable; add an alias and use it;
swap two independent lines; add a branch behind a new flag that is False.
Edits meant to be meaningful: swap the inputs of a cat; switch an off branch on;
reroute one input to an earlier variable; delete one layer call (the layer stays
defined in __init__).

Edits are applied to exact source spans found with `ast`, so the rest of the program --
formatting, comments, everything -- is untouched. A model that tracks dataflow should
keep its answer under neutral edits and change it correctly under meaningful ones. A
model recalling templates tends to answer a meaningful edit with the *original* graph;
`scripts/analyze_edits.py` measures exactly that.
"""

import ast
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx

from graphart.extract import graph_from_code
from graphart.schema import to_text
from graphart.validate import check, to_networkx
from pipeline.run_prompting import load_split

OUT = ROOT / "data" / "generated" / "edits.jsonl"
FRESH_NAMES = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa",
               "lam", "rho", "tau", "phi", "chi", "psi", "zeta", "eta"]
FLAG_NAMES = ["use_refine", "debug_path", "extra_pass", "aux_branch"]


# --- locating things -------------------------------------------------------------

def _forward(tree: ast.Module) -> tuple[ast.ClassDef, ast.FunctionDef, ast.FunctionDef]:
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
        if "forward" in methods and "__init__" in methods:
            return cls, methods["__init__"], methods["forward"]
    raise ValueError("no class with __init__ and forward")


def _offsets(code: str) -> list[int]:
    """Character offset of the start of each line, for (lineno, col) -> index."""
    starts, total = [0], 0
    for line in code.splitlines(keepends=True):
        total += len(line)
        starts.append(total)
    return starts


def _span(code: str, node: ast.AST) -> tuple[int, int]:
    starts = _offsets(code)
    return (starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset)


def _replace(code: str, edits: list[tuple[int, int, str]]) -> str:
    for start, end, text in sorted(edits, reverse=True):
        code = code[:start] + text + code[end:]
    return code


def _line_indent(code: str, node: ast.AST) -> str:
    line = code.splitlines()[node.lineno - 1]
    return line[:len(line) - len(line.lstrip())]


def _assigned(stmt: ast.stmt) -> set[str]:
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _read(stmt: ast.stmt) -> set[str]:
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _loads(stmt: ast.stmt, name: str) -> list[ast.Name]:
    return [n for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name]


def _simple(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Assign)


# --- the edits: each returns edited source, or None when it does not apply ----------

def rename_locals(code, tree, cls, init, fwd, rng):
    params = {a.arg for a in fwd.args.args}
    local = sorted({n.id for s in fwd.body for n in ast.walk(s)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} - params)
    taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    fresh = [f for f in FRESH_NAMES if f not in taken]
    if not local or len(fresh) < len(local):
        return None
    mapping = dict(zip(local, fresh))
    edits = [(*_span(code, n), mapping[n.id]) for s in fwd.body for n in ast.walk(s)
             if isinstance(n, ast.Name) and n.id in mapping]
    return _replace(code, edits)


def add_alias(code, tree, cls, init, fwd, rng):
    body = fwd.body
    for i, stmt in enumerate(body[:-1]):
        if not (_simple(stmt) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            continue
        var = stmt.targets[0].id
        later = next((s for s in body[i + 1:] if var in _read(s)), None)
        if later is None or later.lineno == stmt.end_lineno:
            continue
        alias = f"{var}_ref"
        if alias in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}:
            continue
        uses = [(*_span(code, n), alias) for n in _loads(later, var)]
        end = _offsets(code)[stmt.end_lineno]                       # start of the next line
        insert = (end, end, f"{_line_indent(code, stmt)}{alias} = {var}\n")
        return _replace(code, uses + [insert])
    return None


def swap_independent(code, tree, cls, init, fwd, rng):
    body = fwd.body
    pairs = [(a, b) for a, b in zip(body, body[1:])
             if _simple(a) and _simple(b) and a.end_lineno < b.lineno
             and not (_assigned(a) & (_read(b) | _assigned(b))) and not (_assigned(b) & _read(a))]
    if not pairs:
        return None
    a, b = pairs[0]
    lines = code.splitlines(keepends=True)
    seg_a = "".join(lines[a.lineno - 1:a.end_lineno])
    seg_b = "".join(lines[b.lineno - 1:b.end_lineno])
    starts = _offsets(code)
    return _replace(code, [(starts[a.lineno - 1], starts[a.end_lineno], seg_b),
                           (starts[b.lineno - 1], starts[b.end_lineno], seg_a)])


def add_dead_branch(code, tree, cls, init, fwd, rng):
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    flag = next((f for f in FLAG_NAMES if f not in attrs), None)
    target = next((s for s in fwd.body if _simple(s) and len(s.targets) == 1
                   and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Call)
                   and isinstance(s.value.func, ast.Attribute)
                   and isinstance(s.value.func.value, ast.Name) and s.value.func.value.id == "self"), None)
    if flag is None or target is None:
        return None
    var, layer = target.targets[0].id, target.value.func.attr
    ind = _line_indent(code, target)
    starts = _offsets(code)
    after_target = starts[target.end_lineno]
    branch = f"{ind}if self.{flag}:\n{ind}    {var} = self.{layer}({var})\n"
    last_init = init.body[-1]
    after_init = starts[last_init.end_lineno]
    flag_line = f"{_line_indent(code, last_init)}self.{flag} = False\n"
    return _replace(code, [(after_target, after_target, branch), (after_init, after_init, flag_line)])


def swap_cat(code, tree, cls, init, fwd, rng):
    for call in (n for s in fwd.body for n in ast.walk(s) if isinstance(n, ast.Call)):
        name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
        if name in ("cat", "concat") and call.args and isinstance(call.args[0], (ast.List, ast.Tuple)) \
                and len(call.args[0].elts) >= 2:
            a, b = call.args[0].elts[:2]
            if ast.unparse(a) == ast.unparse(b):
                continue
            (sa, ea), (sb, eb) = _span(code, a), _span(code, b)
            return _replace(code, [(sa, ea, code[sb:eb]), (sb, eb, code[sa:ea])])
    return None


def activate_branch(code, tree, cls, init, fwd, rng):
    flags = {n.test.attr for n in ast.walk(fwd)
             if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
             and isinstance(n.test.value, ast.Name) and n.test.value.id == "self"}
    for stmt in ast.walk(init):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Attribute) \
                and stmt.targets[0].attr in flags and isinstance(stmt.value, ast.Constant) and stmt.value.value is False:
            return _replace(code, [(*_span(code, stmt.value), "True")])
    return None


def reroute_input(code, tree, cls, init, fwd, rng, original=None):
    body = fwd.body
    params = [a.arg for a in fwd.args.args if a.arg != "self"]
    for i in range(len(body) - 1, 0, -1):
        stmt = body[i]
        if not _simple(stmt):
            continue
        earlier = params + [t for s in body[:i] if _simple(s) for t in sorted(_assigned(s))]
        for var in sorted(_read(stmt) - {"self", "torch", "F", "func", "fn", "nn"}):
            for alt in dict.fromkeys(earlier):                         # oldest first, no repeats
                if alt == var or alt in _read(stmt):
                    continue
                edited = _replace(code, [(*_span(code, n), alt) for n in _loads(stmt, var)])
                if _traces(edited) and not _same(edited, original):
                    return edited
    return None


def drop_call(code, tree, cls, init, fwd, rng, original=None):
    for stmt in fwd.body[1:]:
        if _simple(stmt) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) \
                and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute) \
                and getattr(stmt.value.func.value, "id", "") == "self" and len(stmt.value.args) == 1 \
                and isinstance(stmt.value.args[0], ast.Name) and stmt.value.args[0].id == stmt.targets[0].id:
            starts = _offsets(code)
            edited = _replace(code, [(starts[stmt.lineno - 1], starts[stmt.end_lineno], "")])
            if _traces(edited) and not _same(edited, original):
                return edited
    return None


NEUTRAL = {"rename": rename_locals, "alias": add_alias, "reorder": swap_independent,
           "dead_flag": add_dead_branch}
MEANINGFUL = {"swap_cat": swap_cat, "branch_on": activate_branch, "reroute": reroute_input,
              "drop_call": drop_call}


# --- checking ------------------------------------------------------------------------

def _trace(code: str) -> dict | None:
    try:
        graph, _ = graph_from_code(code)
    except Exception:
        return None
    return graph if graph["shapes_available"] and not check(graph) else None


def _traces(code: str) -> bool:
    return _trace(code) is not None


def _iso(a: dict, b: dict) -> bool:
    match = lambda x, y: x["label"] == y["label"]
    return nx.is_isomorphic(to_networkx(a), to_networkx(b), node_match=match, edge_match=match)


def _same(code: str, original: dict) -> bool:
    graph = _trace(code)
    return graph is not None and _iso(graph, original)


def main() -> None:
    bases = load_split("test", None, None, 10)          # the same 40 every model was scored on
    rows, tally = [], {}
    for base in bases:
        tree = ast.parse(base["code"])
        try:
            cls, init, fwd = _forward(tree)
        except ValueError:
            continue
        rng = random.Random(base["id"])
        for name, fn in {**NEUTRAL, **MEANINGFUL}.items():
            extra = {"original": base["graph"]} if name in ("reroute", "drop_call") else {}
            edited = fn(base["code"], tree, cls, init, fwd, rng, **extra)
            if edited is None or edited == base["code"]:
                tally[(name, "not applicable")] = tally.get((name, "not applicable"), 0) + 1
                continue
            graph = _trace(edited)
            if graph is None:
                tally[(name, "does not trace")] = tally.get((name, "does not trace"), 0) + 1
                continue
            kind = "neutral" if _iso(graph, base["graph"]) else "meaningful"
            intended = "neutral" if name in NEUTRAL else "meaningful"
            if kind != intended:
                tally[(name, f"came out {kind}")] = tally.get((name, f"came out {kind}"), 0) + 1
            rows.append({"id": f"{base['id']}~{name}", "base_id": base["id"], "tier": base["tier"],
                         "edit": name, "kind": kind, "code": edited, "target": to_text(graph),
                         "graph": graph, "num_nodes": len(graph["nodes"]),
                         "base_target": base["target"]})
            tally[(name, kind)] = tally.get((name, kind), 0) + 1

    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} edited programs from {len(bases)} test programs -> {OUT.relative_to(ROOT)}")
    for (name, what), n in sorted(tally.items()):
        print(f"  {name:10s} {what:18s} {n:3d}")


if __name__ == "__main__":
    main()

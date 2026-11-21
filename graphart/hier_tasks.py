"""The two tasks a model is trained on and asked, for the hierarchical schema.

    from graphart.hier_tasks import tree_task, wiring_tasks
    t = tree_task(code)                         # {"user", "answer"} for the module tree
    for w in wiring_tasks(code, hier):          # one per block class variant
        ...

1. Tree: the program's code, reduced (graphart/reduce.py) -> the module tree (TREE ...).
2. Wiring, once per variant of a class defined in the code: that class's code (with its
   base classes and the top-level functions it uses), the instance's subtree as the tree
   shows it (which layers it has, their configuration) -> its forward() graph.

In training the subtree comes from the truth; at inference from the model's own tree
answer, so step 2 sees exactly what step 1 produced. The prompts are prompts/hier_tree.txt
and prompts/hier_wire.txt; this module only fills them.
"""

import ast

from .hier import _subtree_lines, _tree_desc, graph_text, tree_text
from .reduce import reduce_source

TREE_PROMPT = "hier_tree"
WIRE_PROMPT = "hier_wire"


def _fill(template: str, **fields) -> str:
    for k, v in fields.items():
        template = template.replace("{" + k + "}", v)
    return template


def class_source(code: str, cls: str) -> str:
    """A class's code as it appears in the (reduced) program, plus the source classes it
    derives from and the top-level functions they call, in program order."""
    tree = ast.parse(code)
    top = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    if cls not in top:
        return ""
    want, todo = {cls}, [cls]
    while todo:
        node = top[todo.pop()]
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        if isinstance(node, ast.ClassDef):
            names |= {b.id for b in node.bases if isinstance(b, ast.Name)}
        for name in names:
            if name in top and name not in want and name != cls:
                # functions it calls, and classes it derives from -- not every class it builds
                is_base = isinstance(node, ast.ClassDef) and any(isinstance(b, ast.Name) and b.id == name
                                                                 for b in node.bases)
                if isinstance(top[name], ast.FunctionDef) or is_base:
                    want.add(name)
                    todo.append(name)
    return "\n\n".join(ast.unparse(n) for n in tree.body if getattr(n, "name", None) in want)


def instance_subtree(h: dict, tag: str) -> str:
    """The first instance with this variant tag, as the module tree writes it."""
    def find(node):
        if node["tag"] == tag and node["kind"] == "block":
            return node
        if node["kind"] == "opaque":
            return None
        for c in node["children"]:
            hit = find(c)
            if hit:
                return hit
        return None
    node = find(h["tree"])
    if node is None:
        return tag
    head = f"{node['name']}: {_tree_desc(node)}" if node["name"] else _tree_desc(node)
    return "\n".join([head] + _subtree_lines(node, 1))


def visible_tags(h: dict) -> list[str]:
    """Variant tags in the order the answer lists their graphs."""
    seen = []

    def walk(node):
        if node["kind"] == "opaque":
            return
        if node["kind"] == "block" and node["tag"] in h["variants"] and node["tag"] not in seen:
            seen.append(node["tag"])
        for c in node["children"]:
            walk(c)
    walk(h["tree"])
    return seen


def tree_task(code: str, template: str, h: dict | None = None) -> dict:
    out = {"user": _fill(template, code=reduce_source(code).strip())}
    if h is not None:
        out["answer"] = "TREE\n" + tree_text(h)
    return out


def wiring_tasks(code: str, h: dict, template: str, with_answers: bool = True) -> list[dict]:
    reduced = reduce_source(code)
    out = []
    for tag in visible_tags(h):
        cls = tag.split("#")[0]
        src = class_source(reduced, cls)
        if not src:
            continue
        item = {"tag": tag, "user": _fill(template, cls=cls, code=src.strip(),
                                          subtree=instance_subtree(h, tag), tag=tag)}
        if with_answers:
            item["answer"] = graph_text(h["variants"][tag]["nodes"])
        out.append(item)
    return out

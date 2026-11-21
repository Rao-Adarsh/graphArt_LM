"""Scoring a hierarchical answer (graphart/hier.py) against the truth.

    from graphart.hier_score import score_answer
    s = score_answer(reply_text, truth_hier)

What is scored, part by part, so the error can be placed (docs/hierarchical_plan.md):

  parse        the reply reads as TREE + GRAPHs (hier.from_text)
  tree         the module tree, each module as (path, description): exact, and F1 over the
               modules; `tree_struct` ignores layer parameters (class and nesting only).
               Numbered runs are expanded first, so `0-2: Block#1` equals three lines.
               Variant tags are left out here and scored as `variants`.
  variants     the block instances grouped into the same variants as the truth, up to
               renumbering (which BasicBlocks run the same forward is config reasoning --
               the `branch_on` skill)
  wiring       per truth variant, the model's graph for the same instances is isomorphic
               to the truth's (node kind + target, edges with their argument slot and
               element selector); `wiring_all` = every variant right
  flat         the answer expanded by hier.inline() -- with the truth's hidden insides for
               opaque boxes, which the code never shows -- compared with inline(truth),
               i.e. the fx graph: `flat_exact` with layer parameters (graphart.validate.
               node_label, the old `exact`), `flat_topo` without (the old `exact_topo`)

`right` is `flat_exact`: the headline, comparable with the flat-schema numbers.
`category` names the first part that failed, in the order above.
"""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx

from .hier import HierError, _fmt_params, _parse_tree, from_text, inline
from .metrics import _isomorphic
from .validate import node_label, topology_label


def modules(tree: dict, with_params: bool = True) -> list[tuple[str, str]]:
    """(path, description) of every module, opaque boxes as one entry, tags left out."""
    out = []

    def walk(node, path):
        if node["kind"] == "leaf":
            desc = node["cls"] + (_fmt_params(node["params"]) if with_params else "")
        else:
            desc = node["cls"]
        out.append((path, desc))
        if node["kind"] in ("leaf", "opaque"):
            return
        for c in node["children"]:
            walk(c, f"{path}.{c['name']}" if path else c["name"])
    walk(tree, "")
    return out


def _f1(pred: list, truth: list) -> float:
    p, t = Counter(pred), Counter(truth)
    hit = sum((p & t).values())
    if not p and not t:
        return 1.0
    prec, rec = hit / max(sum(p.values()), 1), hit / max(sum(t.values()), 1)
    return 0.0 if hit == 0 else 2 * prec * rec / (prec + rec)


def instances(tree: dict) -> dict[str, str]:
    """path -> variant tag of every block instance outside opaque boxes."""
    out = {}

    def walk(node, path):
        if node["kind"] == "opaque":
            return
        if node["kind"] == "block" and node.get("tag"):
            out[path] = node["tag"]
        for c in node["children"]:
            walk(c, f"{path}.{c['name']}" if path else c["name"])
    walk(tree, "")
    return out


def _wiring_graph(nodes: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for i, n in enumerate(nodes):
        g.add_node(i, label=(n["kind"], n["target"], n.get("pack", False)))
    slots: dict[tuple[int, int], list] = {}
    for i, n in enumerate(nodes):
        for slot, tok in enumerate(n.get("src", [])):
            head = tok.split("[")[0].rstrip("*")
            sel = tok[len(head):]
            slots.setdefault((int(head), i), []).append((slot, sel))
    for (a, b), s in slots.items():
        g.add_edge(a, b, label=tuple(s))
    return g


def same_wiring(a: list[dict], b: list[dict]) -> bool:
    """Isomorphic wiring graphs: node kind + target (+ list input), edges with slot + selector."""
    if len(a) != len(b):
        return False
    ga, gb = _wiring_graph(a), _wiring_graph(b)
    if Counter(d["label"] for _, d in ga.nodes(data=True)) != Counter(d["label"] for _, d in gb.nodes(data=True)):
        return False
    return nx.is_isomorphic(ga, gb, node_match=lambda x, y: x["label"] == y["label"],
                            edge_match=lambda x, y: x["label"] == y["label"])


def score_answer(reply: str, truth: dict, truth_flat: dict | None = None) -> dict:
    """Every measure above for one reply; never raises on a bad reply."""
    s = {"parse": False, "tree_exact": False, "tree_struct": False, "tree_f1": 0.0,
         "variants": False, "wiring_right": 0, "wiring_total": 0, "wiring_all": False,
         "flat_ok": False, "flat_exact": False, "flat_topo": False, "error": None}
    visible = [t for t in dict.fromkeys(instances(truth["tree"]).values())
               if t in truth["variants"] and not truth["variants"][t]["hidden"]]
    s["wiring_total"] = len(visible)
    try:
        pred = from_text(reply)
        s["parse"] = True
    except HierError as e:
        s["error"] = f"parse: {e}"
        pred = None
        # a readable tree is still scored when a graph is not
        try:
            lines = [l for l in reply.strip().splitlines() if l.strip() and not l.strip().startswith("```")]
            if lines and lines[0].strip() == "TREE":
                end = next((i for i, l in enumerate(lines) if l.startswith("GRAPH")), len(lines))
                pred = {"tree": _parse_tree(lines[1:end]), "variants": {}, "problems": []}
        except HierError:
            pred = None
    if pred is not None:
        pm, tm = modules(pred["tree"]), modules(truth["tree"])
        s["tree_exact"] = Counter(pm) == Counter(tm)
        s["tree_f1"] = round(_f1(pm, tm), 4)
        s["tree_struct"] = Counter(modules(pred["tree"], False)) == Counter(modules(truth["tree"], False))
        # variants: the same partition of the block instances, up to renaming
        pi, ti = instances(pred["tree"]), instances(truth["tree"])
        fwd, back = {}, {}
        ok = set(pi) == set(ti)
        for path in ti:
            if path not in pi:
                continue
            t, p = ti[path], pi[path]
            ok &= fwd.setdefault(t, p) == p and back.setdefault(p, t) == t
        s["variants"] = ok
        # wiring: each truth variant against the model's graph for its first instance
        first = {}
        for path, t in ti.items():
            first.setdefault(t, path)
        for t in visible:
            p = pi.get(first.get(t, ""), None)
            if p is not None and p in pred["variants"] and same_wiring(pred["variants"][p]["nodes"],
                                                                       truth["variants"][t]["nodes"]):
                s["wiring_right"] += 1
        s["wiring_all"] = s["wiring_right"] == s["wiring_total"]
        if s["parse"]:
            try:
                flat = inline(pred, ref=truth)
                s["flat_ok"] = True
                ref_flat = truth_flat or inline(truth)
                s["flat_exact"] = _isomorphic(flat, ref_flat, node_label)
                s["flat_topo"] = s["flat_exact"] or _isomorphic(flat, ref_flat, topology_label)
            except HierError as e:
                s["error"] = f"inline: {e}"
            except Exception as e:                       # a pathological answer must not stop scoring
                s["error"] = f"inline: {type(e).__name__}: {e}"
    s["right"] = s["flat_exact"]
    s["category"] = ("right" if s["right"] else
                     "parse" if not s["parse"] else
                     "tree" if not s["tree_struct"] else
                     "params" if not s["tree_exact"] else
                     "variants" if not s["variants"] else
                     "wiring" if not s["wiring_all"] else
                     "assembly")
    return s


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for k of n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

"""Sort every reply of a run into one error category, by fixed rules.

    python scripts/classify_errors.py results/<run>.jsonl [--original-of-edits] [--solved results/<test run>.jsonl]

`exact` and `topo` say whether an answer is right; they do not say what is wrong with
the ones that are not. This script does, mechanically: each category below is a
graph-isomorphism test, tried in this order, and a reply gets the first one it passes.
Nothing is judged by reading replies, so anyone can rerun it and get the same table.
Every reply's category, and the first parse error or the labels that differ, go to
results/errors/<run>.csv for auditing.

  RIGHT GRAPH
    exact          labelled isomorphism -- the headline metric
    notation       isomorphic once module parameters are dropped and function/method
                   spellings merged (the `topo` metric)
    spelling       isomorphic once operations are also compared by name alone: an
                   activation written as a function where the code has an nn.ReLU
                   layer (or the reverse), or F.max_pool2d written as max_pool2d
    commutative    isomorphic once the operand order of add and mul is ignored --
                   a + b and b + a are the same computation, but the edge labels
                   (arg0/arg1) differ, so `exact` counts it wrong

  RIGHT WIRING, NARROW ERROR
    arg_slots     same operations, same connections, but some input sits in the
                   wrong argument slot of an order-sensitive operation (cat, sub, ...)
    coarse_*       the reply draws some submodule as a single node where torch.fx
                   traces through it, and is otherwise right: re-tracing the real code
                   with exactly those submodules kept whole gives the reply's graph.
                   coarse_container -- the block is an nn.Sequential/ModuleList, which
                   the prompt explicitly says to expand. coarse_custom -- the block is
                   a user-defined nn.Module class, which the prompt does not mention.

  WRONG
    gave_original  (edit sets, meaningful edits only) the reply is the graph of the
                   program BEFORE the edit -- it ignored what the code now says.
                   Tested before the narrow errors, so it takes precedence over them
    miswired      the right multiset of operations, connected differently
    wrong_ops      operations missing or extra (the CSV lists which)
    unparseable    the reply does not parse, even after dropping module parameters

The relaxations are cumulative: `spelling` includes what `notation` forgives, and
`commutative`, `arg_slots` and `coarse_*` are tested under the `spelling` labels.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx
import torch

from graphart.extract import graph_from_code
from graphart.metrics import compare
from graphart.schema import ParseError, from_text
from pipeline.score import _EMPTY_PARAMS, _MODULE_PARAMS, extract_graph_block, load_truths, run_split

ACTIVATIONS = {"ReLU", "ReLU6", "GELU", "SiLU", "Sigmoid", "Tanh", "Hardswish", "Hardsigmoid"}
COMMUTATIVE = {"add", "mul"}
CONTAINERS = {"Sequential", "ModuleList", "ModuleDict"}

GROUPS = [
    ("right graph", ["exact", "notation", "spelling", "commutative"]),
    ("right wiring, narrow error", ["arg_slots", "coarse_container", "coarse_custom"]),
    ("wrong", ["gave_original", "miswired", "wrong_ops", "unparseable"]),
]


def spelling_label(node: dict) -> tuple:
    op, target, cls = node["op"], node["target"], node["cls"]
    if op == "call_module" and cls in ACTIVATIONS:
        return ("call", cls.lower())
    if op in ("call_function", "call_method"):
        for prefix in ("F.", "torch.", "method."):
            target = target.removeprefix(prefix)
        return ("call", target)
    return (op, target, cls)


def _graph(graph: dict, node_label, commutative: bool, slots: bool) -> nx.DiGraph:
    unordered = {n["id"] for n in graph["nodes"]
                 if n["op"] in ("call_function", "call_method") and spelling_label(n)[1] in COMMUTATIVE}
    g = nx.DiGraph()
    for n in graph["nodes"]:
        g.add_node(n["id"], label=node_label(n))
    for e in graph["edges"]:
        if not slots:
            lab = None
        elif commutative and e["dst"] in unordered:
            lab = ("any", len(e["args"]))  # x + x still differs from x + y
        else:
            lab = tuple(sorted(e["args"]))
        g.add_edge(e["src"], e["dst"], label=lab)
    return g


def iso(pred: dict, truth: dict, node_label=spelling_label, *, commutative=True, slots=True) -> bool:
    if Counter(map(node_label, pred["nodes"])) != Counter(map(node_label, truth["nodes"])):
        return False
    a, b = (_graph(g, node_label, commutative, slots) for g in (pred, truth))
    same = lambda x, y: x["label"] == y["label"]
    return nx.is_isomorphic(a, b, node_match=same, edge_match=same)


def parse(reply: str) -> tuple[dict | None, str | None]:
    """The reply's graph, falling back to dropping module parameters, as `topo` does."""
    try:
        text = extract_graph_block(reply)
    except (ParseError, ValueError, IndexError) as exc:
        return None, str(exc)
    try:
        return from_text(text), None
    except (ParseError, ValueError, IndexError) as exc:
        first_error = str(exc)
    try:
        g = from_text(_EMPTY_PARAMS.sub(r"\1 ", _MODULE_PARAMS.sub(r"\1", text)))
        return (g, None) if g["nodes"] else (None, first_error)
    except (ParseError, ValueError, IndexError):
        return None, first_error


def coarse_category(pred: dict, code: str) -> str | None:
    """coarse_container / coarse_custom if re-tracing with the reply's whole-block
    nodes kept as leaves reproduces the reply; None otherwise."""
    with torch.no_grad():
        _, model = graph_from_code(code)
    modules = dict(model.named_modules())
    tracer = torch.fx.Tracer()
    whole = {n["target"] for n in pred["nodes"]
             if n["op"] == "call_module" and n["target"] in modules
             and not tracer.is_leaf_module(modules[n["target"]], n["target"])}
    if not whole:
        return None
    try:
        coarse, _ = graph_from_code(code, leaf_paths=frozenset(whole))
    except Exception:
        return None
    if not iso(pred, coarse):
        return None
    kinds = {type(modules[p]).__name__ in CONTAINERS for p in whole}
    return "coarse_container" if True in kinds else "coarse_custom"


def _diff(pred: dict, truth: dict) -> str:
    p, t = Counter(map(spelling_label, pred["nodes"])), Counter(map(spelling_label, truth["nodes"]))
    fmt = lambda c: "; ".join(f"{'/'.join(str(x) for x in k if x)}" + (f" x{v}" if v > 1 else "")
                              for k, v in list(c.items())[:6])
    return f"missing: {fmt(t - p) or '-'} | extra: {fmt(p - t) or '-'}"


def classify(reply: str, truth: dict, code: str, original: dict | None = None) -> tuple[str, str]:
    pred, error = parse(reply)
    if pred is None:
        return "unparseable", error
    scores = compare(pred, truth)
    if scores["exact"]:
        return "exact", ""
    if scores["exact_topo"]:
        return "notation", ""
    if iso(pred, truth, commutative=False):
        return "spelling", ""
    if iso(pred, truth):
        return "commutative", ""
    if original is not None and iso(pred, original):
        return "gave_original", ""
    if iso(pred, truth, slots=False):
        return "arg_slots", ""
    try:
        coarse = coarse_category(pred, code)
    except Exception:
        coarse = None
    if coarse:
        return coarse, ""
    if Counter(map(spelling_label, pred["nodes"])) == Counter(map(spelling_label, truth["nodes"])):
        return "miswired", ""
    return "wrong_ops", _diff(pred, truth)


def table(rows: list[tuple[str, str]], title: str) -> None:
    n = len(rows)
    counts = Counter(cat for _, cat in rows)
    print(f"\n{title}  (n={n})")
    for group, cats in GROUPS:
        total = sum(counts[c] for c in cats)
        if not total and group != "right graph":
            continue
        print(f"  {group:30s}{total:5d}  {total / n:5.0%}")
        for c in cats:
            if counts[c]:
                print(f"      {c:26s}{counts[c]:5d}  {counts[c] / n:5.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", help="a results/*.jsonl file written by run_prompting.py")
    ap.add_argument("--solved", help="for an edit set: the same model's test-split run; also "
                                     "report only the programs it got right (topo) before editing")
    ap.add_argument("--by", default=None, help="a row field to break the table down by (tier, edit, kind)")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    path = Path(args.results)
    split = run_split(path)
    data = {r["id"]: r for r in map(json.loads, (ROOT / f"data/generated/{split}.jsonl").open(encoding="utf-8"))}
    truths = load_truths(split)
    edits = split == "edits"
    originals = load_truths("test") if edits else {}

    solved = None
    if args.solved:
        solved = {r["id"] for r in map(json.loads, Path(args.solved).open(encoding="utf-8"))
                  if classify(r["reply"], originals[r["id"]], data.get(r["id"], {}).get("code", ""))[0]
                  in ("exact", "notation")} if edits else None

    out = ROOT / "results" / "errors" / f"{path.stem}.csv"
    out.parent.mkdir(exist_ok=True)
    results = []
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "category", "detail"] + ([args.by] if args.by else []))
        for rec in map(json.loads, path.open(encoding="utf-8")):
            row = data[rec["id"]]
            original = originals[row["base_id"]] if edits and row["kind"] == "meaningful" else None
            cat, detail = classify(rec["reply"], truths[rec["id"]], row["code"], original)
            results.append((row, cat))
            writer.writerow([rec["id"], cat, detail] + ([row.get(args.by)] if args.by else []))

    table([(r["id"], c) for r, c in results], f"{path.name}")
    if args.by:
        groups = defaultdict(list)
        for r, c in results:
            groups[r.get(args.by)].append((r["id"], c))
        for key in sorted(groups, key=str):
            table(groups[key], f"  {args.by} = {key}")
    if solved is not None:
        kept = [(r, c) for r, c in results if r["base_id"] in solved]
        print(f"\n--- only the {len(solved)} programs the model got right (topo) before editing ---")
        for kind in ("neutral", "meaningful"):
            table([(r["id"], c) for r, c in kept if r["kind"] == kind], f"{kind} edits")
        if args.by:
            groups = defaultdict(list)
            for r, c in kept:
                groups[(r["kind"], r.get(args.by))].append((r["id"], c))
            for key in sorted(groups, key=str):
                table(groups[key], f"  {key[0]} / {args.by} = {key[1]}")
    print(f"\nper-reply categories -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

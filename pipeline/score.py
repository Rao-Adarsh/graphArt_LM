"""Score a run produced by run_prompting.py, broken down by tier and circuit size.

    python -m pipeline.score results/mock__echo__zero_shot.jsonl

A reply that will not parse is not an error -- it is a result (valid=False,
exact=False, F1=0), and the "parsed" column says how often that happened.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphart.domain import GRAPH_HEADER, ParseError, from_text, to_text
from graphart.metrics import compare, compare_nodes, exact_topology

ROOT = Path(__file__).resolve().parents[1]
SIZE_BUCKETS = [(3, 8, "3-8 nodes"), (9, 15, "9-15"), (16, 25, "16-25"), (26, 10 ** 6, "26+")]
EMPTY = {"num_inputs": 0, "input_shapes": [], "shapes_available": False,
         "nodes": [], "edges": []}


def extract_graph_block(reply: str) -> str:
    """Slice out the graph, ignoring chain-of-thought, prose and code fences.

    Anything before the last header line is preamble, so a reply that writes
    its working out first -- which cot and cos both ask for -- still scores on
    the answer it finished with. The header comes from the domain; hardcoding
    it silently broke every reasoning prompt when the domain changed.
    """
    lines = reply.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().startswith(GRAPH_HEADER)]
    if not starts:
        return reply
    block = lines[starts[-1]:]
    for i, line in enumerate(block):
        if line.strip().startswith("```"):       # stop at a closing fence
            return "\n".join(block[:i])
    return "\n".join(block)


def _splice_nodes(reply: str, truth: dict) -> str:
    """For the edges-only ablation: put the given nodes back in front of the
    model's edge list, so there is a whole graph to parse and check."""
    edges = reply.split("EDGES", 1)[1] if "EDGES" in reply else reply
    return to_text({**truth, "edges": []}) + "\n" + edges.strip()


# `<id> call_module <path> <Class>(<params>)` -- the parenthesised part is the
# only piece of a node line written in our own notation.
_MODULE_PARAMS = re.compile(r"^(\s*\d+\s+call_module\s+\S+\s+[A-Za-z_]\w*)\([^)]*\)", re.M)
# `Sigmoid [] [1,16]`: an empty bracket pair where a parameterless layer has no
# parameters. The prompts write the format as `[<Class>(<params>)] [<shape>]`,
# brackets meaning "optional", and Qwen2.5-Coder 14B read them literally in 18/40
# zero-shot replies. Only stripped when a shape follows, so a genuine `[]` shape
# (a 0-d tensor) is never touched.
_EMPTY_PARAMS = re.compile(r"^(\s*\d+\s+call_module\s+\S+\s+[A-Za-z_]\w*)\s*\[\]\s+(?=\[)", re.M)


def _topology_fallback(text: str, truth: dict, mode: str) -> bool:
    """`exact_topo` for a reply that failed to parse only because of how it
    wrote module parameters (BatchNorm2d(8) for BatchNorm2d(features=8), or
    `Sigmoid []` for Sigmoid). Strip them and try again; any other parse failure
    still scores False."""
    try:
        stripped = from_text(_EMPTY_PARAMS.sub(r"\1 ", _MODULE_PARAMS.sub(r"\1", text)))
    except (ParseError, ValueError, IndexError):
        return False
    if not stripped["nodes"]:
        return False
    if mode == "nodes":
        return compare_nodes(stripped, truth)["exact_topo"]
    return exact_topology(stripped, truth)


def score_reply(reply: str, truth: dict, *, mode: str = "full") -> dict:
    """Parse one reply and compare it with the ground truth.

    mode "full"  -- the reply is a whole graph
         "nodes" -- the reply lists operations only; edges were not asked for
         "edges" -- the reply lists edges only; the nodes were given in the prompt
    """
    text = ""
    try:
        text = _splice_nodes(reply, truth) if mode == "edges" else extract_graph_block(reply)
        predicted = from_text(text)
    except (ParseError, ValueError, IndexError) as exc:
        # An empty graph trivially passes the structural checks, so `valid` has to
        # be forced down here -- a reply we cannot read is not a well-formed DAG.
        result = compare(EMPTY, truth) | {"valid": False,
                                          "exact_topo": _topology_fallback(text, truth, mode)}
        return {**result, "parsed": False, "parse_error": str(exc)}
    if not predicted["nodes"]:
        # Parses, but says nothing. An empty graph passes every structural check
        # vacuously, so it has to be failed explicitly rather than scored valid.
        return {**compare(EMPTY, truth), "valid": False, "parsed": False,
                "parse_error": "empty reply -- no nodes"}
    if mode == "nodes":
        return {**compare_nodes(predicted, truth), "parsed": True, "parse_error": None}
    return {**compare(predicted, truth), "parsed": True, "parse_error": None}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(scored: list[dict]) -> dict:
    keys = ("parsed", "valid", "exact", "exact_topo", "node_f1", "edge_f1")
    out = {}
    for k in keys:
        # None means "not asked for in this mode", which must not average as 0.
        values = [float(s[k]) for s in scored if s.get(k) is not None]
        out[k] = round(_mean(values), 3) if values else None
    return out | {"n": len(scored)}


def _cell(value) -> str:
    return "   -  " if value is None else f"{value:>6.2f}"


def table(title: str, groups: dict[str, list[dict]]) -> str:
    header = (f"\n{title}\n"
              f"{'group':<14}{'n':>5}{'parsed':>9}{'valid':>8}{'exact':>8}{'topo':>8}"
              f"{'node F1':>10}{'edge F1':>10}")
    lines = [header, "-" * len(header.splitlines()[-1])]
    for name, rows in groups.items():
        s = summarise(rows)
        lines.append(f"{name:<14}{s['n']:>5}   {_cell(s['parsed'])}  {_cell(s['valid'])}"
                     f"  {_cell(s['exact'])}  {_cell(s['exact_topo'])}"
                     f"    {_cell(s['node_f1'])}    {_cell(s['edge_f1'])}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", help="a results/*.jsonl file written by run_prompting.py")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--mode", choices=["full", "nodes", "edges"],
                    help="what the reply contains; inferred from the run's prompt name")
    ap.add_argument("--slides", action="store_true",
                    help="also print the per-tier rows as a RESULTS literal for make_slides.py")
    args = ap.parse_args()

    results_path = Path(args.results).resolve()
    records = [json.loads(line) for line in results_path.open(encoding="utf-8")]
    truths = {r["id"]: r["graph"] for r in
              (json.loads(line) for line in
               (ROOT / "data" / "generated" / f"{args.split}.jsonl").open(encoding="utf-8"))}

    mode = args.mode or ("nodes" if "nodes_only" in results_path.stem
                         else "edges" if "edges_only" in results_path.stem else "full")
    scored = []
    for record in records:
        truth = truths.get(record["id"])
        if truth is None:
            print(f"warning: {record['id']} is not in the {args.split} split -- skipped")
            continue
        scored.append({**record, **score_reply(record["reply"], truth, mode=mode)})

    if not scored:
        raise SystemExit("nothing to score")

    by_tier = {f"tier {t}": [s for s in scored if s["tier"] == t]
               for t in sorted({s["tier"] for s in scored})}
    by_size = {label: rows for lo, hi, label in SIZE_BUCKETS
               if (rows := [s for s in scored if lo <= s["num_nodes"] <= hi])}

    print(f"\n{results_path.name}" + (f"   (mode: {mode})" if mode != "full" else ""))
    print(table("by tier", by_tier))
    print(table("by graph size", by_size))
    print(table("overall", {"all": scored}))
    print("topo = exact, ignoring module parameters and function-vs-method spelling")

    unparsed = [s for s in scored if not s["parsed"]]
    if unparsed:
        print(f"\n{len(unparsed)} replies did not parse; first three reasons:")
        for s in unparsed[:3]:
            print(f"  {s['id']}: {s['parse_error']}")

    summary = {"results": results_path.name, "mode": mode,
               "overall": summarise(scored),
               "by_tier": {k: summarise(v) for k, v in by_tier.items()},
               "by_size": {k: summarise(v) for k, v in by_size.items()}}
    out = results_path.with_suffix(".summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")

    if args.slides:
        print("\nRESULTS = [")
        for name, rows in by_tier.items():
            s = summarise(rows)
            print(f'    ("{name.title()}", "{_cell(s["valid"]).strip()}", '
                  f'"{_cell(s["exact"]).strip()}", "{_cell(s["node_f1"]).strip()}", '
                  f'"{_cell(s["edge_f1"]).strip()}"),')
        print("]")


if __name__ == "__main__":
    main()

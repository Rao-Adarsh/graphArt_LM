"""Score a run produced by run_prompting.py, broken down by tier and circuit size.

    python -m pipeline.score results/mock__echo__zero_shot.jsonl

A reply that will not parse is not an error -- it is a result (valid=False,
exact=False, F1=0), and the "parsed" column says how often that happened.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphart.domain import ParseError, add_edges, from_text
from graphart.metrics import compare

ROOT = Path(__file__).resolve().parents[1]
SIZE_BUCKETS = [(3, 8, "3-8 nodes"), (9, 15, "9-15"), (16, 25, "16-25"), (26, 10 ** 6, "26+")]
EMPTY = {"num_qubits": 0, "num_clbits": 0, "nodes": [], "edges": []}


def extract_graph_block(reply: str) -> str:
    """Slice out the graph, ignoring chain-of-thought, prose and code fences.

    Anything before the last `QUBITS` line is preamble, so a CoT reply that
    writes its working first still scores on the answer it finished with.
    """
    lines = reply.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().startswith("QUBITS")]
    if not starts:
        return reply
    block = lines[starts[-1]:]
    for i, line in enumerate(block):
        if line.strip().startswith("```"):       # stop at a closing fence
            return "\n".join(block[:i])
    return "\n".join(block)


def score_reply(reply: str, truth: dict, *, connect: bool) -> dict:
    """Parse one reply and compare it with the ground truth."""
    try:
        predicted = from_text(extract_graph_block(reply))
    except (ParseError, ValueError, IndexError) as exc:
        # An empty graph trivially passes the structural checks, so `valid` has to
        # be forced down here -- a reply we cannot read is not a well-formed DAG.
        result = compare(EMPTY, truth) | {"valid": False}
        return {**result, "parsed": False, "parse_error": str(exc)}
    if not predicted["nodes"]:
        # Parses, but says nothing. An empty graph passes every structural check
        # vacuously, so it has to be failed explicitly rather than scored valid.
        return {**compare(EMPTY, truth), "valid": False, "parsed": False,
                "parse_error": "empty reply -- no nodes"}
    if connect:
        predicted = add_edges(predicted)
    return {**compare(predicted, truth), "parsed": True, "parse_error": None}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(scored: list[dict]) -> dict:
    keys = ("parsed", "valid", "exact", "node_f1", "edge_f1")
    return {k: round(_mean([float(s[k]) for s in scored]), 3) for k in keys} | {"n": len(scored)}


def table(title: str, groups: dict[str, list[dict]]) -> str:
    header = (f"\n{title}\n"
              f"{'group':<14}{'n':>5}{'parsed':>9}{'valid':>8}{'exact':>8}"
              f"{'node F1':>10}{'edge F1':>10}")
    lines = [header, "-" * len(header.splitlines()[-1])]
    for name, rows in groups.items():
        s = summarise(rows)
        lines.append(f"{name:<14}{s['n']:>5}{s['parsed']:>9.2f}{s['valid']:>8.2f}"
                     f"{s['exact']:>8.2f}{s['node_f1']:>10.2f}{s['edge_f1']:>10.2f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", help="a results/*.jsonl file written by run_prompting.py")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--connect", action="store_true",
                    help="derive the edges from the node order (for the gates_only ablation); "
                         "inferred automatically when the run's prompt is gates_only")
    ap.add_argument("--slides", action="store_true",
                    help="also print the per-tier rows as a RESULTS literal for make_slides.py")
    args = ap.parse_args()

    results_path = Path(args.results).resolve()
    records = [json.loads(line) for line in results_path.open(encoding="utf-8")]
    truths = {r["id"]: r["graph"] for r in
              (json.loads(line) for line in
               (ROOT / "data" / "generated" / f"{args.split}.jsonl").open(encoding="utf-8"))}

    connect = args.connect or "gates_only" in results_path.stem
    scored = []
    for record in records:
        truth = truths.get(record["id"])
        if truth is None:
            print(f"warning: {record['id']} is not in the {args.split} split -- skipped")
            continue
        scored.append({**record, **score_reply(record["reply"], truth, connect=connect)})

    if not scored:
        raise SystemExit("nothing to score")

    by_tier = {f"tier {t}": [s for s in scored if s["tier"] == t]
               for t in sorted({s["tier"] for s in scored})}
    by_size = {label: rows for lo, hi, label in SIZE_BUCKETS
               if (rows := [s for s in scored if lo <= s["num_nodes"] <= hi])}

    print(f"\n{results_path.name}"
          + (f"   (edges derived from the node order)" if connect else ""))
    print(table("by tier", by_tier))
    print(table("by circuit size", by_size))
    print(table("overall", {"all": scored}))

    unparsed = [s for s in scored if not s["parsed"]]
    if unparsed:
        print(f"\n{len(unparsed)} replies did not parse; first three reasons:")
        for s in unparsed[:3]:
            print(f"  {s['id']}: {s['parse_error']}")

    summary = {"results": results_path.name, "connect": connect,
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
            print(f'    ("{name.title()}", "{s["valid"]:.2f}", "{s["exact"]:.2f}", '
                  f'"{s["node_f1"]:.2f}", "{s["edge_f1"]:.2f}"),')
        print("]")


if __name__ == "__main__":
    main()

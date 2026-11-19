"""Split each run's misses into "listed the wrong nodes" and "wired the right nodes wrongly".

    python scripts/decompose.py results/<run>.jsonl [results/<run>.jsonl ...] [--per-sample]
    python scripts/decompose.py --torchvision          # every torchvision run in the comparison

`right graph` alone hides where a model fails. Here every reply is checked twice:

  nodes right   the reply's node multiset equals the reference's, labels compared as
                classify_errors compares them for a right graph (module parameters
                dropped, F.relu = nn.ReLU = relu). For a two-step run the step-1 reply
                is used when results/raw/<run>/<id>.step1.txt exists, so a step-2 slip
                does not hide good nodes.
  right graph   labelled isomorphism, forgiving notation, spelling and add/mul operand
                order: the same test as classify_errors' RIGHT GRAPH group.

`wired right | nodes right` is then the wiring success on samples whose nodes were right.

Edge recall split by adjacency asks whether a model wires by reading the code or by
connecting each node to the one listed before it: `adjacent` is the share of reference
edges whose source is the node listed just before the destination that the reply also
has (matched by node labels), `other` the share of all remaining edges. A model that
reads the dataflow scores both alike; one that chains its list scores `adjacent` high
and `other` low. Pooled over the replies that parse, so approximate.

A nodes_only run reports nodes only; an edges_only run (nodes given) reports wiring
only. Nothing here calls a model.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import classify_errors as ce                      # noqa: E402
from graphart.two_step import parse_step1          # noqa: E402
from pipeline.score import load_truths, run_split, score_reply   # noqa: E402

C = collections.Counter

TORCHVISION_RUNS = [
    "hf__qwen2.5-coder-1.5b-graphart-lora__zero_shot__torchvision",
    "hf__qwen2.5-coder-1.5b-graphart-lora-mixed__zero_shot__torchvision",
    "hf__qwen2.5-coder-1.5b-graphart-lora-mixed-v2__zero_shot__torchvision",
    "hf__qwen2.5-coder-1.5b-graphart-lora-two-step__two_step_brief__torchvision",
    "hf__qwen2.5-coder-1.5b-graphart-lora-two-step-full__two_step__torchvision",
    "hf__qwen2.5-coder-1.5b-graphart-lora-two-step-full-epoch__two_step__torchvision",
    "ollama__qwen2.5-coder-14b__zero_shot__torchvision",
    "ollama__qwen2.5-coder-14b__nodes_only_few_shot__torchvision",
    "ollama__qwen2.5-coder-14b__edges_only__torchvision",
    "ollama__qwen2.5-coder-14b__edges_only_cos__torchvision",
]


def labels(nodes: list[dict]) -> C:
    return C(ce.spelling_label(n) for n in nodes)


def edges_by_label(graph: dict, pick=lambda e: True) -> C:
    lab = {n["id"]: ce.spelling_label(n) for n in graph["nodes"]}
    return C((lab.get(e["src"]), lab.get(e["dst"])) for e in graph["edges"] if pick(e))


def analyse(path: Path) -> dict:
    run = path.stem
    truths = load_truths(run_split(path))
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    given = bool(records) and records[0].get("given_nodes")
    mode = "full" if given else "nodes" if "nodes_only" in run else "edges" if "edges_only" in run else "full"
    raw = ROOT / "results" / "raw" / run
    per, hit, tot = {}, C(), C()
    for rec in records:
        i, t = rec["id"], truths[rec["id"]]
        if mode != "full":
            s = score_reply(rec["reply"], t, mode=mode)
            per[i] = "R" if (s["exact"] or s.get("exact_topo")) else "."
            continue
        pred, _ = ce.parse(rec["reply"])
        step1 = raw / f"{i}.step1.txt"
        nodes = None
        if step1.exists():
            try:
                nodes = parse_step1(step1.read_text(encoding="utf-8"))[0]["nodes"]
            except Exception:
                nodes = None
        elif pred is not None:
            nodes = pred["nodes"]
        nodes_ok = nodes is not None and labels(nodes) == labels(t["nodes"])
        right = pred is not None and ce.iso(pred, t)
        per[i] = "R" if right else "n" if nodes_ok else "U" if pred is None else "."
        if pred is not None:
            p = edges_by_label(pred)
            for kind, pick in (("adjacent", lambda e: e["src"] == e["dst"] - 1),
                               ("other", lambda e: e["src"] != e["dst"] - 1)):
                ref = edges_by_label(t, pick)
                hit[kind] += sum((ref & p).values())
                tot[kind] += sum(ref.values())
    out = {"run": run, "mode": mode, "n": len(per), "per": per,
           "right": sum(v == "R" for v in per.values())}
    if mode == "full":
        out["nodes_right"] = sum(v in "Rn" for v in per.values())
        out["unparseable"] = sum(v == "U" for v in per.values())
        out["adjacent"] = hit["adjacent"] / tot["adjacent"] if tot["adjacent"] else None
        out["other"] = hit["other"] / tot["other"] if tot["other"] else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="*", help="results/*.jsonl files")
    ap.add_argument("--torchvision", action="store_true", help="every torchvision run in the comparison")
    ap.add_argument("--per-sample", action="store_true",
                    help="also print one row per sample (R right, n nodes right but miswired, "
                         ". nodes wrong, U unparseable)")
    args = ap.parse_args()
    paths = [Path(p) for p in args.results]
    if args.torchvision:
        paths += [ROOT / "results" / f"{r}.jsonl" for r in TORCHVISION_RUNS]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no results given (or none exist)")
    sys.stdout.reconfigure(errors="replace")
    runs = [analyse(p) for p in paths]

    print(f"{'run':78s} {'n':>3} {'nodes right':>11} {'wired | nodes':>13} {'right':>6} "
          f"{'unparsed':>8} {'edges: adjacent':>15} {'other':>6}")
    for r in runs:
        name = r["run"][:78]
        if r["mode"] != "full":
            what = "nodes listed right" if r["mode"] == "nodes" else "wired right (nodes given)"
            print(f"{name:78s} {r['n']:3d}   {what}: {r['right']}")
            continue
        wired = f"{r['right']}/{r['nodes_right']}"
        fmt = lambda v: "  -  " if v is None else f"{v:.2f}"
        print(f"{name:78s} {r['n']:3d} {r['nodes_right']:11d} {wired:>13} {r['right']:6d} "
              f"{r['unparseable']:8d} {fmt(r['adjacent']):>15} {fmt(r['other']):>6}")

    if args.per_sample:
        ids = sorted({i for r in runs for i in r["per"]})
        print("\n" + " " * 28 + " ".join(f"{k:>3d}" for k in range(1, len(runs) + 1)))
        for i in ids:
            print(f"{i:28s}" + " ".join(f"{r['per'].get(i, '-'):>3s}" for r in runs))
        print("columns:", "; ".join(f"{k} {r['run']}" for k, r in enumerate(runs, 1)))


if __name__ == "__main__":
    main()

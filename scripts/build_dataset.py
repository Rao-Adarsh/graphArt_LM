"""Generate the synthetic code -> forward-pass-graph dataset.

    python scripts/build_dataset.py --per-tier 500

Writes data/generated/{train,val,test}.jsonl, one JSON object per line.

This script is specific to the PyTorch domain -- it imports that generator and
extractor directly rather than through `graphart.domain`, which exists for the
scoring side of the harness (metrics and pipeline/), not for dataset building.

The per-sample metadata is chosen for what this domain and the research
questions actually need, not as a translation of the circuit dataset's fields.
`num_merges` and `max_fan_in` are the real measure of topological difficulty
and are what separate tier 3 from tiers 1-2; `num_modules` and
`hierarchy_depth` speak to the "identify layers and hierarchy" requirement;
`code_lines` against `num_nodes` separates how long the source is from how big
the answer is.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphart.extract import graph_from_code
from graphart.generate import TIERS, generate_program
from graphart.schema import to_text
from graphart.validate import check, to_networkx

MIN_NODES = 4
MAX_NODES = 80          # squeezenet1_1 (68) and resnet18 (71) are the smallest
                        # real architectures with genuine branching; above this
                        # `exact` collapses on output length rather than on
                        # difficulty, and Phase 2 stops fitting a free GPU.


def _stats(graph: dict) -> dict:
    incoming: Counter = Counter()
    for e in graph["edges"]:
        incoming[e["dst"]] += len(e["args"])
    module_paths = [n["target"] for n in graph["nodes"] if n["op"] == "call_module"]
    g = to_networkx(graph)
    return {
        "num_nodes": len(graph["nodes"]),
        "num_edges": len(graph["edges"]),
        "num_modules": len(module_paths),
        "num_merges": sum(1 for v in incoming.values() if v > 1),
        "max_fan_in": max(incoming.values(), default=0),
        "depth": nx.dag_longest_path_length(g) if graph["nodes"] else 0,
        "hierarchy_depth": max((p.count(".") + 1 for p in module_paths), default=0),
        "input_shape": graph["input_shapes"][0] if graph["input_shapes"] else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=500)
    ap.add_argument("--min-nodes", type=int, default=MIN_NODES)
    ap.add_argument("--max-nodes", type=int, default=MAX_NODES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/generated")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    seen_targets: set[str] = set()
    report: Counter = Counter()

    for tier in TIERS:
        samples, attempts = [], 0
        while len(samples) < args.per_tier and attempts < args.per_tier * 20:
            attempts += 1
            code = generate_program(rng, tier)
            try:
                graph, _ = graph_from_code(code)
            except Exception:
                report["failed_to_trace"] += 1
                continue
            if not args.min_nodes <= len(graph["nodes"]) <= args.max_nodes:
                report["size_filtered"] += 1
                continue
            if not graph["shapes_available"]:
                # The target carries shapes, so a sample without them would be
                # a different task wearing the same format.
                report["no_shapes"] += 1
                continue
            target = to_text(graph)
            if target in seen_targets:   # would otherwise leak across splits
                report["duplicate"] += 1
                continue
            assert not check(graph), check(graph)   # ground truth must pass our own checks
            seen_targets.add(target)
            samples.append({
                "id": f"t{tier}-{len(samples):05d}", "tier": tier, "code": code,
                "target": target, "graph": graph,
                "code_lines": len(code.strip().splitlines()),
                **_stats(graph),
            })

        rng.shuffle(samples)
        a, b = int(0.8 * len(samples)), int(0.9 * len(samples))
        splits["train"] += samples[:a]
        splits["val"] += samples[a:b]
        splits["test"] += samples[b:]
        nodes = [s["num_nodes"] for s in samples]
        merges = [s["num_merges"] for s in samples]
        print(f"tier {tier}: {len(samples):5d} unique from {attempts} attempts, "
              f"nodes min/avg/max = {min(nodes)}/{sum(nodes) / len(nodes):.1f}/{max(nodes)}, "
              f"merges avg {sum(merges) / len(merges):.2f}")

    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name:5s}: {len(rows)} samples -> {out / (name + '.jsonl')}")
    print("skipped:", dict(report))


if __name__ == "__main__":
    main()

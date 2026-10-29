"""Build the library-style dataset (tiers 5-6) from graphart/generate_large.py.

    python scripts/build_large_dataset.py --per-tier 400

Writes data/generated/large_{train,val,test}.jsonl in the same row format as
build_dataset.py, plus a `family` field (resnet, inception, ...). Living beside the
first dataset under its own split names, it is read unchanged by the runner and scorer
(`--split large_test`) and by the trainer (`--train-split train,large_train`).

Two things differ from build_dataset.py, both learned from the first dataset:

- Duplicates are removed by graph ISOMORPHISM, not by the text of the target. The
  first builder compared text, so the same graph written in another node order
  slipped through: 4 of its 160 test graphs repeat a training graph.
- Any program whose graph is topology-isomorphic to one of the 21 torchvision
  samples is rejected, so the torchvision set stays a real out-of-distribution test.

Families are drawn in rotation, so every family gets the same share of each tier.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_dataset import _stats
from graphart.extract import graph_from_code
from graphart.generate_large import FAMILIES, LARGE_TIERS, generate_large_program
from graphart.metrics import _isomorphic
from graphart.schema import to_text
from graphart.validate import check, node_label, to_networkx, topology_label

MIN_NODES = 8
# Above ~120 nodes an answer passes ~4,500 tokens; Phase 2 trained on answers of at
# most 475. Raise it once the trainer's --max-len and the GPU budget allow.
MAX_NODES = 120


def wl_hash(graph: dict, label) -> str:
    """Equal graphs always get equal hashes; used to bucket before the exact check."""
    g = to_networkx(graph)
    for n in graph["nodes"]:
        g.nodes[n["id"]]["h"] = str(label(n))
    for _, _, d in g.edges(data=True):
        d["h"] = str(d["label"])
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="h", edge_attr="h")


class Seen:
    """Graphs seen so far, by hash bucket, with an exact isomorphism check inside."""

    def __init__(self, label):
        self.label = label
        self.buckets: dict[str, list[dict]] = defaultdict(list)

    def find(self, graph: dict) -> bool:
        return any(_isomorphic(graph, other, self.label) for other in self.buckets[wl_hash(graph, self.label)])

    def add(self, graph: dict) -> None:
        self.buckets[wl_hash(graph, self.label)].append(graph)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-tier", type=int, default=400)
    ap.add_argument("--min-nodes", type=int, default=MIN_NODES)
    ap.add_argument("--max-nodes", type=int, default=MAX_NODES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--prefix", default="large_", help="split files are <prefix>train.jsonl etc.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torchvision = Seen(topology_label)
    tv_path = ROOT / "data" / "generated" / "torchvision.jsonl"
    if tv_path.exists():
        for r in map(json.loads, tv_path.open(encoding="utf-8")):
            torchvision.add(r["graph"])
    else:
        print("note: data/generated/torchvision.jsonl not found -- not filtering against it")
    seen = Seen(node_label)
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    report: Counter = Counter()
    families = sorted(FAMILIES)

    for tier in LARGE_TIERS:
        samples, attempts = [], 0
        while len(samples) < args.per_tier and attempts < args.per_tier * 20:
            family = families[attempts % len(families)]
            attempts += 1
            code = generate_large_program(rng, tier, family)
            try:
                graph, _ = graph_from_code(code)
            except Exception:
                report["failed_to_trace"] += 1
                continue
            if not args.min_nodes <= len(graph["nodes"]) <= args.max_nodes:
                report["size_filtered"] += 1
                continue
            if not graph["shapes_available"]:
                report["no_shapes"] += 1
                continue
            if torchvision.find(graph):
                report["same_as_torchvision"] += 1
                continue
            if seen.find(graph):
                report["duplicate"] += 1
                continue
            assert not check(graph), check(graph)   # ground truth must pass our own checks
            seen.add(graph)
            samples.append({
                "id": f"t{tier}-{len(samples):05d}", "tier": tier, "family": family, "code": code,
                "target": to_text(graph), "graph": graph,
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
        fams = Counter(s["family"] for s in samples)
        print(f"tier {tier} ({LARGE_TIERS[tier]}): {len(samples)} unique from {attempts} attempts, "
              f"nodes min/avg/max = {min(nodes)}/{sum(nodes) / len(nodes):.1f}/{max(nodes)}, "
              f"merges avg {sum(merges) / len(merges):.2f}, families {dict(sorted(fams.items()))}")

    for name, rows in splits.items():
        path = out / f"{args.prefix}{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name:5s}: {len(rows)} samples -> {path.relative_to(ROOT)}")
    print("skipped:", dict(report))


if __name__ == "__main__":
    main()

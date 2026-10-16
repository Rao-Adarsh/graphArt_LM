"""Generate the synthetic code -> circuit-DAG dataset.

    python scripts/build_dataset.py --per-tier 500

Writes data/generated/{train,val,test}.jsonl, one JSON object per line:
    id, tier, code, target (graph text), graph (JSON), num_qubits, num_nodes, num_edges, depth
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphart.legacy_qiskit.extract import graph_from_code
from graphart.legacy_qiskit.generate import TIERS, generate_program
from graphart.legacy_qiskit.schema import to_text
from graphart.legacy_qiskit.validate import check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=500)
    ap.add_argument("--max-nodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/generated")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": [], "test": []}
    seen_targets: set[str] = set()
    report = Counter()

    for tier in TIERS:
        samples, attempts = [], 0
        while len(samples) < args.per_tier and attempts < args.per_tier * 20:
            attempts += 1
            code = generate_program(rng, tier)
            try:
                graph, circuit = graph_from_code(code)
            except Exception:
                report["failed_to_run"] += 1
                continue
            target = to_text(graph)
            if not 3 <= len(graph["nodes"]) <= args.max_nodes:
                report["size_filtered"] += 1
                continue
            if target in seen_targets:  # same DAG already in the dataset -> would leak across splits
                report["duplicate"] += 1
                continue
            assert not check(graph), check(graph)  # ground truth must pass our own validator
            seen_targets.add(target)
            samples.append({
                "id": f"t{tier}-{len(samples):05d}", "tier": tier, "code": code, "target": target,
                "graph": graph, "num_qubits": graph["num_qubits"], "num_nodes": len(graph["nodes"]),
                "num_edges": len(graph["edges"]), "depth": circuit.depth(),
            })
        rng.shuffle(samples)
        a, b = int(0.8 * len(samples)), int(0.9 * len(samples))
        splits["train"] += samples[:a]
        splits["val"] += samples[a:b]
        splits["test"] += samples[b:]
        nodes = [s["num_nodes"] for s in samples]
        print(f"tier {tier}: {len(samples):5d} unique samples from {attempts} attempts, "
              f"nodes min/avg/max = {min(nodes)}/{sum(nodes) / len(nodes):.1f}/{max(nodes)}")

    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name:5s}: {len(rows)} samples -> {out / (name + '.jsonl')}")
    print("skipped:", dict(report))


if __name__ == "__main__":
    main()

"""Build the per-instance tree examples (graphart/hier_inst.py) from hier_train / hier_val.

    .venv/Scripts/python scripts/build_hier_inst.py                # both splits
    .venv/Scripts/python scripts/build_hier_inst.py --limit 50     # a quick try

Per program: build it with the constructor arguments recorded, one example per distinct
instance of a class in the code (the root with the whole reduced program), and a check that
the true answers, joined by hier_inst.assemble, give back exactly the stored tree's modules
and layer parameters (graphart/hier_score.modules). A program that fails the check is left
out and counted. Writes data/generated/hier_inst_<split>.jsonl with id, program, source,
kind, path, cls, user, answer; train/val only -- never a dev, test or frozen program.
"""

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from graphart.hier_inst import INST_PROMPT, assemble, instance_examples
from graphart.hier_score import modules
from graphart.reduce import reduce_source
from pipeline.run_prompting import load_prompt

DATA = ROOT / "data" / "generated"


def build(split: str, limit: int | None) -> None:
    _, template = load_prompt(INST_PROMPT)
    rows = [json.loads(l) for l in (DATA / f"{split}.jsonl").open(encoding="utf-8")]
    rows = rows[:limit] if limit else rows
    out, report, started = [], Counter(), time.time()
    for i, r in enumerate(rows, 1):
        try:
            timm_name = r["id"][len("timm-"):] if r["source"] == "timm" else None
            ex = instance_examples(r["code"], reduce_source(r["code"]), template, timm_name)
        except Exception as e:
            report[f"failed to build: {type(e).__name__}"] += 1
            continue
        by_desc = {e["head"]: e["answer"] for e in ex if e["path"]}
        root = next((e for e in ex if not e["path"]), None)
        try:
            tree = assemble(root["answer"], lambda path, desc: by_desc[desc])
            same = Counter(modules(tree)) == Counter(modules(r["hier"]["tree"]))
        except Exception as e:
            report[f"assembly failed: {type(e).__name__}"] += 1
            if report[f"assembly failed: {type(e).__name__}"] <= 3:
                print(f"  {r['id']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        if not same:
            report["assembled tree differs from the stored one"] += 1
            if report["assembled tree differs from the stored one"] <= 5:
                a, b = Counter(modules(tree)), Counter(modules(r["hier"]["tree"]))
                print(f"  {r['id']} differs: extra {list((a - b).items())[:3]} missing {list((b - a).items())[:3]}",
                      flush=True)
            continue
        report["programs"] += 1
        for k, e in enumerate(ex):
            out.append({"id": f"{r['id']}/{k}", "program": r["id"], "source": r["source"], "kind": r["kind"],
                        "path": e["path"], "cls": e["cls"], "user": e["user"], "answer": e["answer"]})
        if i % 200 == 0:
            print(f"  {split}: {i}/{len(rows)} ({time.time() - started:.0f}s)", flush=True)
    path = DATA / f"hier_inst_{split.removeprefix('hier_')}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    per = Counter(r["program"] for r in out)
    print(f"{split}: {len(out)} instance examples from {len(per)} programs "
          f"(mean {len(out) / max(len(per), 1):.1f}, max {max(per.values(), default=0)}); {dict(report)} -> {path.name}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", default="hier_train,hier_val")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    for split in args.splits.split(","):
        build(split, args.limit)


if __name__ == "__main__":
    main()

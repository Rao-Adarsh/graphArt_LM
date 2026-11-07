"""Build the two-step training data (graphart/two_step.py) from existing splits.

    python scripts/build_two_step.py                       # train, val, large_train, large_val
    python scripts/build_two_step.py --splits test

For every program it re-traces the code with graphart/cos_trace.py and writes
data/generated/two_step/<split>.jsonl: the row's id, tier, num_nodes and code, plus
`step1` (nodes with their source stacks) and `step2` (edges with their variables).
The splits themselves are not changed. A program whose trace does not reproduce the
stored ground truth, or fails `check_trace`, stops the build: no trace is written
that has not passed every check.

The frozen holdout is refused: nothing is derived from it until it is scored.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphart import cos_trace

DATA = ROOT / "data" / "generated"
OUT = DATA / "two_step"
FIELDS = ("id", "tier", "num_nodes", "code")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", default="train,val,large_train,large_val")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        if split == "torchvision_holdout":
            raise SystemExit("torchvision_holdout is frozen (data/generated/FROZEN.json) -- not building from it")
        rows = [json.loads(l) for l in (DATA / f"{split}.jsonl").open(encoding="utf-8")]
        out = []
        for r in rows:
            t = cos_trace.trace(r["code"])
            if t["graph"]["nodes"] != r["graph"]["nodes"] or t["graph"]["edges"] != r["graph"]["edges"]:
                raise SystemExit(f"{split}/{r['id']}: the trace does not reproduce the stored ground truth")
            problems = cos_trace.check_trace(r["code"], t)
            if problems:
                raise SystemExit(f"{split}/{r['id']}: {problems[:3]}")
            out.append({**{k: r[k] for k in FIELDS},
                        "step1": cos_trace.render_step1(t), "step2": cos_trace.render_step2(t)})
        path = OUT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in out:
                f.write(json.dumps(row) + "\n")
        print(f"{split:12s} {len(out):5d} programs -> {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()

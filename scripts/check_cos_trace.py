"""Check the CoS trace generator (graphart/cos_trace.py) before any trace is trained on.

    python scripts/check_cos_trace.py            # the hand-checked cases, then every dataset
    python scripts/check_cos_trace.py --cases    # the hand-checked cases only

1. Hand-checked cases: small programs for each construct that could fool it, with
   the step-2 line expected for one node, written down from reading the code.
2. Every program in every split except the frozen holdout: it must trace to exactly
   the stored ground truth and pass `check_trace`.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphart import cos_trace as ct, two_step

HEAD = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n\n"


def program(init: str, forward: str, shape: str = "1, 4, 8, 8", helpers: str = "") -> str:
    body = "\n".join("        " + l for l in init.strip().splitlines())
    fwd = "\n".join("        " + l for l in forward.strip().splitlines())
    return (HEAD + helpers + "class M(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
            f"{body}\n\n    def forward(self, x):\n{fwd}\n\n\nmodel = M()\nexample_input = torch.randn({shape})\n")


# (name, code, {dst id: expected step-2 line})
CASES = [
    ("reassignment and an alias: `out += identity` reads identity, not x",
     program("self.conv = nn.Conv2d(4, 4, 3, padding=1)\nself.bn = nn.BatchNorm2d(4)",
             "identity = x\nout = self.conv(x)\nout = self.bn(out)\nout += identity\nreturn F.relu(out)"),
     {1: "1 <- x=0 arg0", 2: "2 <- out=1 arg0", 3: "3 <- out=2 arg0; identity=0 arg1", 4: "4 <- out=3 arg0"}),
    ("nested calls on one line: inner results have no name",
     program("self.bn = nn.BatchNorm2d(4)\nself.fc = nn.Conv2d(4, 2, 1)",
             "return self.fc(F.relu(self.bn(x)))"),
     {1: "1 <- x=0 arg0", 2: "2 <- 1 arg0", 3: "3 <- 2 arg0"}),
    ("a loop over a ModuleList: each layer reads the loop's running variable",
     program("self.blocks = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(3)])",
             "h = x\nfor blk in self.blocks:\n    h = blk(h)\nreturn h"),
     {1: "1 <- h=0 arg0", 2: "2 <- h=1 arg0", 3: "3 <- h=2 arg0"}),
    ("nn.Sequential: only its first layer reads a variable",
     program("self.seq = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU(), nn.Conv2d(4, 4, 1))",
             "x = self.seq(x)\nreturn x"),
     {1: "1 <- x=0 arg0", 2: "2 <- 1 arg0", 3: "3 <- 2 arg0"}),
    ("branches built in a list comprehension, then concatenated",
     program("self.branches = nn.ModuleList([nn.Conv2d(4, 2, 1) for _ in range(2)])",
             "return torch.cat([b(x) for b in self.branches], 1)"),
     {1: "1 <- x=0 arg0", 2: "2 <- x=0 arg0", 3: "3 <- 1 arg0; 2 arg1"}),
    ("chunk and unpack, then swap the halves",
     program("pass", "a, b = x.chunk(2, dim=1)\nreturn torch.cat([b, a], 1)"),
     {2: "2 <- 1 arg0", 3: "3 <- 1 arg0", 4: "4 <- b=3 arg0; a=2 arg1"}),
    ("a module-level helper function reads its own parameter name",
     program("self.fc = nn.Conv2d(4, 4, 1)", "y = act(self.fc(x))\nreturn y",
             helpers="def act(t):\n    return F.gelu(t)\n\n\n"),
     {1: "1 <- x=0 arg0", 2: "2 <- t=1 arg0"}),
    ("one statement over several lines; one value filling two slots",
     program("pass", "y = torch.cat(\n    [x, x],\n    dim=1)\nreturn y"),
     {1: "1 <- x=0 arg0 arg1"}),
    ("torch._assert on a shape check",
     program("self.fc = nn.Linear(4, 4)", 'torch._assert(x.dim() == 2, "need 2-D")\nreturn self.fc(x)', shape="2, 4"),
     {1: "1 <- x=0 arg0", 2: "2 <- 1 arg0", 3: "3 <- 2 arg0", 4: "4 <- x=0 arg0"}),
    ("the same tensor in three slots of attention",
     program("self.proj = nn.Linear(4, 4)\nself.attn = nn.MultiheadAttention(4, 1, batch_first=True)",
             "h = self.proj(x)\nout, _ = self.attn(h, h, h, need_weights=False)\nreturn out", shape="1, 3, 4"),
     {2: "2 <- h=1 arg0 arg1 arg2"}),
]


def run_cases() -> int:
    failed = 0
    for name, code, expected in CASES:
        t = ct.trace(code)
        lines = {int(l.split(" <- ")[0]): l for l in ct.render_step2(t).splitlines()[1:]}
        problems = ct.check_trace(code, t)
        wrong = {dst: (want, lines.get(dst)) for dst, want in expected.items() if lines.get(dst) != want}
        ok = not problems and not wrong
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        for dst, (want, got) in wrong.items():
            print(f"       node {dst}: expected {want!r}, got {got!r}")
        for p in problems:
            print(f"       {p}")
        if not ok:
            print(two_step.numbered(code))
            print(ct.render_step1(t))
            print(ct.render_step2(t))
    return failed


def run_datasets() -> int:
    failed = 0
    data = ROOT / "data" / "generated"
    for split in ("train", "val", "test", "large_train", "large_val", "large_test", "torchvision", "edits", "rq6"):
        path = data / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        bad, named, total = [], 0, 0
        for r in rows:
            t = ct.trace(r["code"])
            if t["graph"]["nodes"] != r["graph"]["nodes"] or t["graph"]["edges"] != r["graph"]["edges"]:
                bad.append((r["id"], "differs from the stored ground truth"))
                continue
            bad += [(r["id"], p) for p in ct.check_trace(r["code"], t)]
            named += len(t["names"])
            total += len(t["graph"]["edges"])
        print(f"{split:12s} {len(rows):5d} programs, {len(bad)} problems, "
              f"{named / max(total, 1):.0%} of edges carry a variable name")
        for x in bad[:5]:
            print("      ", x)
        failed += len(bad)
    return failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", action="store_true", help="only the hand-checked cases")
    args = ap.parse_args()
    failed = run_cases()
    if not args.cases:
        failed += run_datasets()
    print("\nall good" if not failed else f"\n{failed} problem(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

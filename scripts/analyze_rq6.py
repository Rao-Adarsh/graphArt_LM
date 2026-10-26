"""Compare adapters on the RQ6 probe set (a construct held out of training).

    python scripts/analyze_rq6.py results/<run>__rq6.jsonl [results/<other run>__rq6.jsonl ...]

Accuracy per group (see build_rq6_probe.py), and for programs with a dead branch, how
many answers put the dead branch's layer into the graph anyway -- the specific mistake
of a model that has not learned that a False flag means the code never runs.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphart.schema import from_text
from pipeline.score import extract_graph_block, score_reply

GROUPS = ("control", "dead_branch", "dead_flag", "branch_on")


def dead_layers(code: str) -> set[str]:
    """Module attributes called inside an `if self.<flag>:` whose flag is False."""
    tree = ast.parse(code)
    false_flags = {t.attr for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Attribute)
                   and isinstance(n.value, ast.Constant) and n.value.value is False}
    return {c.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute) and n.test.attr in false_flags
            for stmt in n.body for c in ast.walk(stmt)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and getattr(c.func.value, "id", "") == "self"}


def included_dead_layer(reply: str, code: str, truth: dict) -> bool:
    """True when the answer calls a dead-branch layer more often than the real graph does.
    Counting, not just presence: the dead branch may reuse a layer that is also called
    outside it, and then writing the branch in shows up only as one call too many."""
    dead = dead_layers(code)
    if not dead:
        return False
    try:
        pred = from_text(extract_graph_block(reply))
    except Exception:
        return False
    count = lambda g, t: sum(1 for n in g["nodes"] if n["op"] == "call_module" and n["target"] == t)
    return any(count(pred, t) > count(truth, t) for t in dead)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="results/*__rq6.jsonl files")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    rows = {r["id"]: r for r in (json.loads(l) for l in (ROOT / "data/generated/rq6.jsonl").open(encoding="utf-8"))}

    print(f"{'run':52s}" + "".join(f"{g:>16s}" for g in GROUPS) + "   dead layer written in")
    for path in args.runs:
        recs = [json.loads(l) for l in Path(path).open(encoding="utf-8")]
        by = {g: [] for g in GROUPS}
        dead_written = dead_total = 0
        for rec in recs:
            row = rows.get(rec["id"])
            if row is None:
                continue
            s = score_reply(rec["reply"], row["graph"])
            by[row["group"]].append(s["exact_topo"])
            if row["group"] in ("dead_branch", "dead_flag"):
                dead_total += 1
                dead_written += included_dead_layer(rec["reply"], row["code"], row["graph"])
        cells = "".join(f"{(sum(v) / len(v) if v else float('nan')):>9.2f} ({len(v):3d})" for v in by.values())
        print(f"{Path(path).stem[:52]:52s}{cells}   {dead_written} of {dead_total}")
    print("\ncells are topo accuracy (n); the last column counts answers that include a layer "
          "from a branch that never runs")


if __name__ == "__main__":
    main()

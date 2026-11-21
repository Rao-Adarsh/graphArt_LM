"""Score hierarchical runs (pipeline/run_hier.py) against the truth.

    python -m pipeline.score_hier results/<run>.jsonl [more runs ...]
    python -m pipeline.score_hier results/<run>.jsonl --per-sample

The truth is the run's split (from its .meta.json) in data/generated/. Measures are
graphart/hier_score.py's; `right` (the flat graph after inline equals fx's, with layer
parameters) is the headline, shown with its Wilson 95% interval. Writes
results/<run>.score.json with every program's measures. The frozen sets need --frozen.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphart.hier_score import score_answer, wilson
from pipeline.run_hier import FROZEN
from pipeline.run_prompting import ROOT

DATA = ROOT / "data" / "generated"
COLS = [("parse", "parse"), ("tree_struct", "tree (no params)"), ("tree_exact", "tree"),
        ("variants", "variants"), ("wiring_all", "all wiring"), ("flat_topo", "flat topo"),
        ("right", "RIGHT (flat exact)")]
CATEGORIES = ("right", "parse", "tree", "params", "variants", "wiring", "assembly")


def score_run(path: Path, frozen_ok: bool) -> tuple[dict, list[dict]]:
    meta = json.loads(path.with_name(path.stem + ".meta.json").read_text(encoding="utf-8"))
    split = meta["split"]
    if split in FROZEN and not frozen_ok:
        raise SystemExit(f"{split} is frozen -- pass --frozen only for its one scored run")
    truth = {r["id"]: r for r in map(json.loads, (DATA / f"{split}.jsonl").open(encoding="utf-8"))}
    rows = []
    for rec in map(json.loads, path.open(encoding="utf-8")):
        t = truth[rec["id"]]
        s = score_answer(rec["reply"], t["hier"])
        rows.append({"id": rec["id"], "kind": t.get("kind"), "source": t.get("source", t.get("set")),
                     "family": t.get("family"), "run_error": rec.get("error"), **s})
    return meta, rows


def table(rows: list[dict], label: str) -> str:
    n = len(rows)
    if not n:
        return f"  {label}: no programs"
    cells = []
    for key, name in COLS:
        k = sum(bool(r[key]) for r in rows)
        cells.append(f"{name} {k}/{n}")
    k = sum(r["right"] for r in rows)
    lo, hi = wilson(k, n)
    wr = sum(r["wiring_right"] for r in rows), sum(r["wiring_total"] for r in rows)
    f1 = sum(r["tree_f1"] for r in rows) / n
    return (f"  {label} (n={n}): " + " | ".join(cells) +
            f"\n      right {k / n:.0%} [95% CI {lo:.0%}-{hi:.0%}]; tree F1 {f1:.3f}; "
            f"variant graphs right {wr[0]}/{wr[1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--per-sample", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # replies may hold any character
    for run in args.runs:
        path = Path(run)
        meta, rows = score_run(path, args.frozen)
        print(f"\n{path.stem}  ({meta.get('model')}, {meta.get('split')})")
        print(table(rows, "all"))
        for key in ("kind", "source"):
            groups = collections.defaultdict(list)
            for r in rows:
                groups[r[key]].append(r)
            if len(groups) > 1:
                for g in sorted(groups, key=str):
                    print(table(groups[g], f"{key}={g}"))
        cats = collections.Counter(r["category"] for r in rows)
        print("  first failing part: " + ", ".join(f"{c} {cats[c]}" for c in CATEGORIES if cats[c]))
        if args.per_sample:
            for r in rows:
                print(f"    {r['id']:40s} {r['category']:9s} tree F1 {r['tree_f1']:.2f} "
                      f"wiring {r['wiring_right']}/{r['wiring_total']}"
                      + (f"  ({r['error'] or r['run_error']})" if (r['error'] or r['run_error']) else ""))
        out = path.with_name(path.stem + ".score.json")
        k = sum(r["right"] for r in rows)
        out.write_text(json.dumps({"run": path.stem, "meta": meta, "n": len(rows), "right": k,
                                   "wilson95": wilson(k, len(rows)), "categories": dict(cats),
                                   "programs": rows}, indent=1), encoding="utf-8")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()

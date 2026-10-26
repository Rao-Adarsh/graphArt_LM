"""Read a model's answers on the controlled-edit set (RQ5).

    python scripts/analyze_edits.py results/<run>__edits.jsonl [results/<run>.jsonl]

For each edit, with `exact` and `topo` as everywhere else:

  neutral edits     the right answer is the original graph; a drop means the model is
                    thrown by surface changes that do not change the dataflow.
  meaningful edits  the right answer is a new graph. Among the wrong answers, "gave the
                    original" counts those that are exactly the *unedited* program's
                    graph: the model answered from what the code usually looks like, not
                    from what it now says.

Pass the same model's test-split run as the second argument to condition on the
programs it got right before any edit -- the cleanest view, since a miss on an edited
program it never solved says nothing about the edit.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx

from graphart.schema import from_text
from graphart.validate import to_networkx
from pipeline.score import extract_graph_block, load_truths, score_reply


def _iso(a: dict, b: dict) -> bool:
    match = lambda x, y: x["label"] == y["label"]
    return nx.is_isomorphic(to_networkx(a), to_networkx(b), node_match=match, edge_match=match)


def gave_original(reply: str, original: dict) -> bool:
    try:
        return _iso(from_text(extract_graph_block(reply)), original)
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("edits_run", help="a results/*__edits.jsonl file")
    ap.add_argument("test_run", nargs="?", help="the same model's test-split run, to condition on")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    rows = {r["id"]: r for r in (json.loads(l) for l in (ROOT / "data/generated/edits.jsonl").open(encoding="utf-8"))}
    originals = load_truths("test")
    solved = None
    if args.test_run:
        solved = {r["id"] for r in (json.loads(l) for l in Path(args.test_run).open(encoding="utf-8"))
                  if score_reply(r["reply"], originals[r["id"]])["exact"]}

    groups = defaultdict(list)
    for rec in (json.loads(l) for l in Path(args.edits_run).open(encoding="utf-8")):
        row = rows.get(rec["id"])
        if row is None or (solved is not None and row["base_id"] not in solved):
            continue
        s = score_reply(rec["reply"], row["graph"])
        orig = (row["kind"] == "meaningful" and not s["exact"]
                and gave_original(rec["reply"], originals[row["base_id"]]))
        for key in ((row["kind"], row["edit"]), (row["kind"], "ALL")):
            groups[key].append((s["exact"], s["exact_topo"], orig))

    print(f"{Path(args.edits_run).name}"
          + (f"  -- only programs the model solved unedited ({len(solved)}/40)" if solved is not None else ""))
    print(f"{'kind':11s}{'edit':11s}{'n':>4}{'exact':>8}{'topo':>7}   wrong answers that were the original graph")
    for kind in ("neutral", "meaningful"):
        for (k, edit), vals in sorted(groups.items(), key=lambda kv: (kv[0][1] == "ALL", kv[0][1])):
            if k != kind:
                continue
            n = len(vals)
            exact = sum(v[0] for v in vals) / n
            topo = sum(v[1] for v in vals) / n
            wrong = n - sum(v[0] for v in vals)
            tail = f"{sum(v[2] for v in vals):3d} of {wrong:3d}" if kind == "meaningful" else ""
            sep = "-" * 4 if edit == "ALL" else ""
            print(f"{kind:11s}{edit:11s}{n:4d}{exact:8.2f}{topo:7.2f}   {tail}{sep}")


if __name__ == "__main__":
    main()

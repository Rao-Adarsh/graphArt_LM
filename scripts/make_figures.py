"""Render one example per tier to docs/figures/ and run a tiny scoring demo.

    python scripts/make_figures.py
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphart.legacy_qiskit.extract import circuit_from_code
from graphart.legacy_qiskit.render import side_by_side
from graphart.metrics import compare

TIER_NAMES = {1: "flat gate list", 2: "loops", 3: "functions / if-else / nested loops", 4: "textbook algorithm"}


def main():
    rows = [json.loads(line) for line in open("data/generated/test.jsonl", encoding="utf-8")]
    figdir = Path("docs/figures")
    figdir.mkdir(parents=True, exist_ok=True)

    for tier, name in TIER_NAMES.items():
        # pick a mid-sized example so the figure is readable
        candidates = sorted((r for r in rows if r["tier"] == tier), key=lambda r: abs(r["num_nodes"] - 10))
        r = candidates[0]
        path = figdir / f"tier{tier}.png"
        side_by_side(r["code"], circuit_from_code(r["code"]), r["graph"],
                     title=f"Tier {tier}: {name}   ({r['id']})", path=str(path))
        print("wrote", path)

    # Scoring demo: perturb a ground-truth graph the way an LLM might get it wrong.
    truth = next(r["graph"] for r in rows if r["tier"] == 2 and r["num_nodes"] >= 8)
    wrong_gate = copy.deepcopy(truth)
    wrong_gate["nodes"][0]["gate"] = "x" if wrong_gate["nodes"][0]["gate"] != "x" else "h"
    missing_edge = copy.deepcopy(truth)
    missing_edge["edges"].pop()
    print("\nscoring demo")
    for label, pred in [("perfect copy", truth), ("one wrong gate", wrong_gate), ("one missing edge", missing_edge)]:
        s = compare(pred, truth)
        print(f"  {label:17s} valid={s['valid']!s:5s} exact={s['exact']!s:5s} "
              f"node_f1={s['node_f1']:.2f} edge_f1={s['edge_f1']:.2f}  {s['errors'][:1]}")


if __name__ == "__main__":
    main()

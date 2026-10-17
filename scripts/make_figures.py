"""Render one example per tier, plus the two figures that explain scoring.

    python scripts/make_figures.py

Writes docs/figures/tier{1..4}.png -- source beside the graph it traces to --
and docs/figures/comparison.png, which is deliverable 4: a reference graph and
a model-style answer side by side with every mismatch coloured.
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphart.metrics import compare
from graphart.render import compare_figure, side_by_side

TIER_NAMES = {1: "flat", 2: "loops", 3: "branches", 4: "blocks"}


def _pick(rows, tier, target_nodes=11):
    """A mid-sized example, so the figure stays readable."""
    candidates = [r for r in rows if r["tier"] == tier]
    return min(candidates, key=lambda r: abs(r["num_nodes"] - target_nodes))


def main() -> None:
    rows = [json.loads(line) for line in open("data/generated/test.jsonl", encoding="utf-8")]
    figdir = Path("docs/figures")
    figdir.mkdir(parents=True, exist_ok=True)

    for tier, name in TIER_NAMES.items():
        r = _pick(rows, tier)
        path = figdir / f"tier{tier}.png"
        side_by_side(r["code"], r["graph"],
                     title=f"Tier {tier}: {name}   ({r['id']})", path=str(path))
        print(f"wrote {path}  ({r['num_nodes']} nodes, {r['num_merges']} merges)")

    # Deliverable 4: what a wrong answer looks like next to the right one.
    # The damage is the kind a model actually makes -- a layer misread, a
    # branch dropped -- rather than something artificial.
    truth = _pick([r for r in rows if r["num_merges"] >= 1], 3, target_nodes=10)["graph"]
    pred = copy.deepcopy(truth)
    if pred["nodes"]:
        for node in pred["nodes"]:
            if node["op"] == "call_module" and node["params"].get("out"):
                node["params"]["out"] = node["params"]["out"] * 2    # misread a width
                break
    merge = next((e for e in pred["edges"] if e["args"] != [0]), None)
    if merge:
        pred["edges"].remove(merge)                                  # dropped a branch
    path = figdir / "comparison.png"
    compare_figure(truth, pred, title="Reference vs model output", path=str(path))
    score = compare(pred, truth)
    print(f"wrote {path}")
    print(f"  that prediction scores: valid={score['valid']} exact={score['exact']} "
          f"node_f1={score['node_f1']:.2f} edge_f1={score['edge_f1']:.2f}")

    # Scoring demo: a correct graph against two damaged copies.
    base = _pick(rows, 3, target_nodes=12)["graph"]
    wrong_layer = copy.deepcopy(base)
    for node in wrong_layer["nodes"]:
        if node["op"] == "call_module":
            node["target"] = node["target"] + "_x"
            break
    missing_edge = copy.deepcopy(base)
    missing_edge["edges"].pop()
    print("\nscoring demo")
    for label, pred in [("perfect copy", base), ("one layer renamed", wrong_layer),
                        ("one missing edge", missing_edge)]:
        s = compare(pred, base)
        print(f"  {label:18s} valid={s['valid']!s:5s} exact={s['exact']!s:5s} "
              f"node_f1={s['node_f1']:.2f} edge_f1={s['edge_f1']:.2f}  {s['errors'][:1]}")


if __name__ == "__main__":
    main()

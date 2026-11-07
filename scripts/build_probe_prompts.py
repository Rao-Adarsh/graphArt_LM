"""Write the prompts for the second 14B probe: better nodes without fine-tuning, and CoS
wiring given the true nodes.

    python scripts/build_probe_prompts.py

  edges_only_cos.txt             the edges-only task (true nodes given) with a CoS
                                 simulation before the edges
  nodes_only_few_shot.txt        nodes-only, with two worked examples
  nodes_only_stack.txt           nodes-only, each node with its @ source stack (the
                                 two-step step-1 prompt, under a name the scorer reads
                                 as nodes-only)
  nodes_only_stack_few_shot.txt  both

The examples are the two programs the 14B's failures call for: helper classes whose
layers must be listed one by one (it wrote 39 BasicConv2d boxes on torchvision). They
come from large_train only -- the shortest inception and inverted_residual programs
with 12-22 nodes and at least 3 layers inside helpers -- and their simulations, stacks
and node lists are rendered from the fx trace (graphart/cos_trace.py), not written by
hand.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphart import cos_trace, two_step
from graphart.schema import to_text

PROMPTS = ROOT / "prompts"
SEP = "---USER---"
FAMILIES = ("inception", "inverted_residual")


def in_helper(row: dict) -> int:
    return sum(1 for n in row["graph"]["nodes"] if n["op"] == "call_module"
               and len([p for p in n["target"].split(".") if not p.isdigit()]) >= 2)


def examples() -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "data" / "generated" / "large_train.jsonl").open(encoding="utf-8")]
    out = []
    for family in FAMILIES:
        pool = [r for r in rows if r["family"] == family and 12 <= r["num_nodes"] <= 22 and in_helper(r) >= 3]
        out.append(min(pool, key=lambda r: (len(r["code"]), r["id"])))
    return out


def simulation(row: dict, t: dict) -> str:
    """forward() statement by statement, stepping into every helper it calls: a call's
    own line once, then the lines inside it, indented; each node with its sources."""
    lines = row["code"].splitlines()
    names = t["names"]
    by_dst: dict[int, list] = {}
    for e in t["graph"]["edges"]:
        by_dst.setdefault(e["dst"], []).append(e)
    out, prev, prev_inst = [], [], None
    for n in t["graph"]["nodes"]:
        stack = t["stack"].get(n["id"]) or []
        # which helper instance the node is made in: two calls of one helper class (the
        # layers of an nn.Sequential) run the same lines, so the header names the instance
        inst = None
        if len(stack) > 1:
            inst = n["target"].rsplit(".", 1)[0] if n["op"] == "call_module" and "." in n["target"] \
                else (prev_inst if prev[:-1] == stack[:-1] else None)
        srcs = ", ".join(f"{names[(e['src'], n['id'])] + '=' if (e['src'], n['id']) in names else ''}n{e['src']} "
                         + " ".join(f"arg{a}" for a in e["args"])
                         for e in sorted(by_dst.get(n["id"], []), key=lambda e: min(e["args"])))
        node = f"n{n['id']} {n['op']} {n['target']}" + (f"   <- {srcs}" if srcs else "")
        if n["op"] == "placeholder":
            out.append(f"{n['target']} is the input   ->   {node}")
            continue
        if not stack:                                    # the output node
            out.append(f"return   ->   {node}")
            continue
        for depth in range(len(stack) - 1):
            deepest = depth == len(stack) - 2
            if prev[:depth + 1] != stack[:depth + 1] or (deepest and inst is not None and inst != prev_inst):
                into = f"steps into {inst}" if deepest and inst else "steps into it"
                out.append("    " * depth + lines[stack[depth] - 1].strip() + f"   ({into})")
        out.append("    " * (len(stack) - 1) + f"{lines[stack[-1] - 1].strip()}   ->   {node}")
        prev, prev_inst = stack, inst
    return "\n".join(out)


def system_of(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8").split(SEP)[0].rstrip()


def user_of(name: str) -> str:
    return SEP + (PROMPTS / f"{name}.txt").read_text(encoding="utf-8").split(SEP)[1]


def write(name: str, text: str) -> None:
    (PROMPTS / f"{name}.txt").write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote prompts/{name}.txt ({len(text)} chars)")


def main() -> None:
    exs = [(r, cos_trace.trace(r["code"])) for r in examples()]
    for r, t in exs:
        assert not cos_trace.check_trace(r["code"], t), r["id"]
        assert t["graph"]["nodes"] == r["graph"]["nodes"] and t["graph"]["edges"] == r["graph"]["edges"], r["id"]
    print("examples:", ", ".join(f"{r['id']} ({r['family']}, {r['num_nodes']} nodes)" for r, _ in exs))

    nodes_of = lambda r: r["target"].split("EDGES")[0].strip()
    edges_of = lambda r: "EDGES" + r["target"].split("EDGES")[1].rstrip()

    # --- nodes-only, few-shot -------------------------------------------------------
    shots = "".join(f"\n\nExample {i}\nCode:\n```python\n{r['code'].strip()}\n```\n{nodes_of(r)}"
                    for i, (r, _) in enumerate(exs, 1))
    write("nodes_only_few_shot", system_of("nodes_only") + "\n\nIn both examples below, each"
          " helper class is listed layer by layer, never as one node." + shots + "\n" + user_of("nodes_only"))

    # --- nodes-only with @ source stacks: the two-step step-1 prompt -----------------
    write("nodes_only_stack", (PROMPTS / "two_step_nodes.txt").read_text(encoding="utf-8"))
    shots = "".join(f"\n\nExample {i}\nCode:\n```python\n{two_step.numbered(r['code'])}\n```\n"
                    f"{cos_trace.render_step1(t)}" for i, (r, t) in enumerate(exs, 1))
    write("nodes_only_stack_few_shot", system_of("two_step_nodes") + shots + "\n" + user_of("two_step_nodes"))

    # --- edges-only (true nodes given) with a CoS simulation -------------------------
    rules = system_of("edges_only")
    head, _, _ = rules.rpartition("- Output only the edge list")
    how = """
HOW TO WORK IT OUT

Simulate forward() one statement at a time, as if you were the interpreter, keeping a
record of which node each variable holds. When a statement calls a helper -- a module
or function of this source, or an nn.Sequential of them -- step into it: its lines run
with its own variable names, its parameters holding what it was called with. Every
given node is made by exactly one line of this simulation.

Write the simulation under a heading SIMULATION, one line per node:
    <statement>   ->   n<id> <op> <target>   <- <variable>=n<id> arg<k>, ...
(an input no variable holds -- a nested call, the previous layer of an nn.Sequential --
is just n<id> arg<k>). Then write the edges under the heading EDGES, in exactly this
format, one edge per line, and nothing after them:

EDGES
0 -> 1 : arg0
1 -> 2 : arg0"""
    shots = "".join(
        f"\n\nExample {i}\nCode:\n```python\n{r['code'].strip()}\n```\n\nNodes:\n{nodes_of(r)}\n\n"
        f"SIMULATION\n{simulation(r, t)}\n{edges_of(r)}" for i, (r, t) in enumerate(exs, 1))
    write("edges_only_cos", head.rstrip() + "\n" + how + shots + "\n" + user_of("edges_only"))


if __name__ == "__main__":
    main()

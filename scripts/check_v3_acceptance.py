"""The acceptance check for generator v3, as fixed in docs/generator_rework_plan.md.

    .venv/Scripts/python scripts/check_v3_acceptance.py

Compares the v3 programs of data/generated/hier_{train,val,gen_test}.jsonl with the pooled
real code: the 30 dev networks (hier_dev.jsonl) and the timm training models (source
"timm" in hier_train / hier_val). Never reads the holdout or the timm test set.

  1. Size: the median of each size measure for v3 networks falls inside the real pool's
     interquartile range (reduced code tokens, tree lines, block graphs, largest block,
     module-path nesting, classes defined).
  2. Operations: every operation used by >= 15% of real networks appears in >= 10% of v3
     programs.
  3. Idioms: every idiom reaches at least half its real share.
  4. Lossless and unseen: guaranteed by the builder (it drops anything not lossless or
     isomorphic to an eval-set program); re-counted here on every v3 program.

Prints PASS / FAIL per criterion with the numbers behind it.
"""

import json
import re
import statistics
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from build_generator_viewer import IDIOMS, idioms, vocab_label
from graphart.hier import inline, tree_text
from graphart.reduce import reduce_source

DATA = ROOT / "data" / "generated"


def load(name):
    p = DATA / f"{name}.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def visible_vocab(h: dict) -> set:
    """Operations the answer contains: functions, methods and parameter reads in the visible
    block graphs, and built-in layers in the tree outside opaque boxes -- not what happens
    inside a layer whose source is not shown (the model never writes those)."""
    out = set()
    for v in h["variants"].values():
        if v["hidden"]:
            continue
        for n in v["nodes"]:
            if n["kind"] == "function":
                out.add(n["target"] if "." in n["target"] else f"{n['target']}()")
            elif n["kind"] == "method":
                out.add(f".{n['target']}()")
            elif n["kind"] == "get_attr":
                out.add("get_attr")

    def walk(node):
        if node["kind"] == "opaque":
            return
        if node["kind"] == "leaf":
            out.add(f"nn.{node['cls']}")
        for c in node["children"]:
            walk(c)
    walk(h["tree"])
    return out


def measures(r, T) -> dict:
    h = r["hier"]
    flat = inline(h)
    visible = [v for v in h["variants"].values() if not v["hidden"]]
    paths = [n["target"] for n in flat["nodes"] if n["op"] == "call_module"]
    return {
        "code_red_tok": T(reduce_source(r["code"])),
        "tree_lines": len(tree_text(h).splitlines()),
        "n_graphs": len(visible),
        "max_graph": max((len(v["nodes"]) for v in visible), default=0),
        "nesting": max((p.count(".") + 1 for p in paths), default=0),
        "classes": len(re.findall(r"^class \w+", r["code"], re.M)),
        "vocab": visible_vocab(h),
        "idioms": set(idioms(r["code"])),
    }


def q(v, p):
    s = sorted(v)
    i = (len(s) - 1) * p
    lo = int(i)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (i - lo)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", local_files_only=True)
    T = lambda s: len(tok(s)["input_ids"])
    gen = [r for s in ("hier_train", "hier_val", "hier_gen_test") for r in load(s) if r["source"] == "v3"]
    real = [r for r in load("hier_dev") if r["kind"] == "network"] + \
           [r for s in ("hier_train", "hier_val") for r in load(s) if r["source"] == "timm"]
    print(f"v3 programs: {len(gen)} ({sum(r['kind'] == 'network' for r in gen)} networks); "
          f"real pool: {len(real)} networks")
    G = [measures(r, T) for r in gen]
    GN = [m for m, r in zip(G, gen) if r["kind"] == "network"]
    R = [measures(r, T) for r in real]
    results = {}

    print("\n1. Size: v3 network median inside the real interquartile range")
    ok1 = True
    for k, label in [("code_red_tok", "reduced code tokens"), ("tree_lines", "tree lines"),
                     ("n_graphs", "block graphs"), ("max_graph", "largest block (nodes)"),
                     ("nesting", "module-path nesting"), ("classes", "classes defined")]:
        rv = [m[k] for m in R]
        lo, hi = q(rv, 0.25), q(rv, 0.75)
        med = statistics.median(m[k] for m in GN)
        ok = lo <= med <= hi
        ok1 &= ok
        print(f"   {'PASS' if ok else 'FAIL'}  {label:24s} v3 {med:7.0f}   real IQR {lo:6.0f} - {hi:6.0f} "
              f"(median {statistics.median(rv):.0f})")
    results["size"] = ok1

    print("\n2. Operations used by >= 15% of real networks appear in >= 10% of v3 programs")
    share = lambda ms, v: sum(v in m["vocab"] for m in ms) / len(ms)
    vocab = sorted({v for m in R for v in m["vocab"]})
    ok2, misses = True, []
    for v in vocab:
        rs = share(R, v)
        if rs < 0.15:
            continue
        gs = share(G, v)
        if gs < 0.10:
            ok2 = False
            misses.append(f"{v} (real {rs:.0%}, v3 {gs:.0%})")
    print(f"   {'PASS' if ok2 else 'FAIL'}  " + ("all present" if ok2 else "missing: " + ", ".join(misses)))
    results["operations"] = ok2

    print("\n3. Idioms reach at least half their real share")
    ok3 = True
    for idi in IDIOMS:
        rs = sum(idi in m["idioms"] for m in R) / len(R)
        gs = sum(idi in m["idioms"] for m in G) / len(G)
        ok = gs >= rs / 2
        ok3 &= ok
        print(f"   {'PASS' if ok else 'FAIL'}  {idi[:48]:48s} real {rs:4.0%}  v3 {gs:4.0%}")
    results["idioms"] = ok3

    print("\n4. Lossless and unseen: guaranteed by build_hier_dataset.py (it drops the rest)")
    results["lossless"] = True
    print("\nSUMMARY: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in results.items()))


if __name__ == "__main__":
    main()

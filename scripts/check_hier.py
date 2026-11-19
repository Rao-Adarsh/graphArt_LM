"""Check the hierarchical schema is lossless, and measure how big its answers are.

    .venv/Scripts/python scripts/check_hier.py                 # generator data, dev 21, full models
    .venv/Scripts/python scripts/check_hier.py --sets torchvision --show tv-resnet18

For every program: extract the tree and wiring graphs (graphart/hier.py), then check that
inlining them -- directly, and after writing them as text and parsing back -- gives exactly
the flat torch.fx graph. Prints failures with their reasons, and the size of the answers
(tree lines, graphs, largest graph, tokens) per set next to the flat graph's size.
The 17-sample holdout is never read (it is spent; see CLAUDE.md).
"""

import argparse
import inspect
import json
import statistics
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import torch

from graphart.hier import check_lossless, hier_from_code, hier_from_module, to_text, tree_text
from graphart.schema import to_text as flat_text

DATA = ROOT / "data" / "generated"
FULL_MODELS = [
    ("alexnet", {}, 224), ("vgg11", {}, 224), ("resnet18", {}, 224), ("resnet50", {}, 224),
    ("squeezenet1_1", {}, 224), ("googlenet", {"aux_logits": False, "init_weights": False}, 224),
    ("inception_v3", {"aux_logits": False, "init_weights": False}, 299),
    ("mobilenet_v2", {}, 224), ("mobilenet_v3_small", {}, 224), ("mnasnet1_0", {}, 224),
    ("shufflenet_v2_x1_0", {}, 224), ("efficientnet_b0", {}, 224), ("regnet_y_400mf", {}, 224),
    ("convnext_tiny", {}, 224), ("densenet121", {}, 224), ("vit_b_16", {}, 224),
]


def tokenizer():
    """Qwen's tokenizer if it is cached locally, else a characters/3.3 estimate."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", local_files_only=True)
        return lambda s: len(tok(s)["input_ids"]), "Qwen tokens"
    except Exception:
        return lambda s: round(len(s) / 3.3), "~tokens (chars/3.3)"


def programs(sets: list[str]):
    for name in sets:
        if name == "full":
            import torchvision.models as tvm
            for model_name, kw, side in FULL_MODELS:
                torch.manual_seed(0)
                model = getattr(tvm, model_name)(weights=None, **kw)
                src_mod = inspect.getmodule(type(model)).__name__
                yield "full", f"full-{model_name}", (model, (lambda c, m=src_mod: c.__module__ == m),
                                                     torch.randn(1, 3, side, side))
            continue
        path = DATA / f"{name}.jsonl"
        if name == "torchvision_holdout":
            raise SystemExit("the holdout is spent; not reading it")
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            yield name, r["id"], r["code"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=["large_train", "large_val", "large_test", "torchvision", "full"])
    ap.add_argument("--show", help="print the answer text for this program id")
    ap.add_argument("--limit", type=int, default=0, help="programs per set (0 = all)")
    ap.add_argument("--out", help="write per-program sizes and problems as jsonl")
    args = ap.parse_args()

    count_tokens, unit = tokenizer()
    rows, per_set = [], Counter()
    for set_name, pid, src in programs(args.sets):
        if args.limit and per_set[set_name] >= args.limit:
            continue
        per_set[set_name] += 1
        try:
            h, flat = hier_from_code(src) if isinstance(src, str) else hier_from_module(*src)
            issues = check_lossless(h, flat)
        except Exception as e:
            rows.append({"set": set_name, "id": pid, "error": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        text = to_text(h)
        visible = {t: v for t, v in h["variants"].items() if not v["hidden"]}
        rows.append({
            "set": set_name, "id": pid, "issues": issues,
            "flat_nodes": len(flat["nodes"]),
            "flat_tokens": count_tokens(flat_text(flat)),
            "tree_lines": len(tree_text(h).splitlines()),
            "graphs": text.count("\nGRAPH "),
            "graph_nodes": sum(len(v["nodes"]) for v in visible.values()),
            "max_graph": max((len(v["nodes"]) for v in visible.values()), default=0),
            "tokens": count_tokens(text),
            "tree_tokens": count_tokens(tree_text(h)),
        })
        if args.show == pid:
            print(text, "\n")

    print(f"sizes in {unit}; median / max per set\n")
    hdr = f"{'set':12s} {'n':>4s} {'fail':>4s} {'flat nodes':>12s} {'flat tok':>12s} {'tree lines':>12s} {'graphs':>9s} {'max graph':>10s} {'answer tok':>12s} {'tree tok':>11s}"
    print(hdr)
    for s in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == s]
        ok = [r for r in rs if "error" not in r]
        bad = [r for r in rs if "error" in r or r["issues"]]

        def mm(k):
            v = [r[k] for r in ok]
            return f"{statistics.median(v):g} / {max(v)}" if v else "-"
        print(f"{s:12s} {len(rs):4d} {len(bad):4d} {mm('flat_nodes'):>12s} {mm('flat_tokens'):>12s} "
              f"{mm('tree_lines'):>12s} {mm('graphs'):>9s} {mm('max_graph'):>10s} {mm('tokens'):>12s} {mm('tree_tokens'):>11s}")
    fails = [r for r in rows if "error" in r or r["issues"]]
    if fails:
        print(f"\n{len(fails)} program(s) with problems:")
        reasons = Counter()
        for r in fails:
            for i in ([r["error"]] if "error" in r else r["issues"]):
                reasons[i.split(":")[0] if "error" in r else i[:90]] += 1
        for reason, n in reasons.most_common(25):
            print(f"  {n:4d}  {reason}")
        print("  e.g.", ", ".join(r["id"] for r in fails[:12]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

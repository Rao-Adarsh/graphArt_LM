"""Build a browsable page of the torchvision-style generator's programs and their fx graphs.

    .venv/Scripts/python scripts/build_generator_viewer.py
    .venv/Scripts/python scripts/build_generator_viewer.py --fresh 50      # + 50 new programs per tier

Writes one self-contained HTML file (default results/raw/generator_viewer.html, gitignored)
with three views:

  Browse   - any program: its source next to the graph torch.fx traces from it
  Compare  - two graphs side by side (e.g. a generated network next to real ResNet-18)
  Stats    - generated vs real code on size and shape (nodes, depth, nesting, merges ...),
             which operations real code uses that the generator never writes, and which
             code idioms each side uses

What is in it:

  generator tier 5 / tier 6  data/generated/large_{train,val,test}.jsonl, the programs
                             graphart/generate_large.py wrote and the adapters trained on
  generator fresh            (--fresh N) programs generated now, from the current generator
  torchvision dev 21         data/generated/torchvision.jsonl, the development set
  torchvision full models    whole torchvision classification networks traced here, for
                             scale; their "code" is the model's source file

The 17-sample holdout is not included (CLAUDE.md: it is spent and never looked at to tune).
Graph layout runs in the browser (dagre from cdnjs), so the page needs a network connection.
"""

import argparse
import inspect
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx

from graphart.validate import to_networkx

DATA = ROOT / "data" / "generated"
TEMPLATE = ROOT / "interface" / "generator_viewer_template.html"

# Whole torchvision networks, traced for scale: (name, builder kwargs, input side).
FULL_MODELS = [
    ("alexnet", {}, 224), ("vgg11", {}, 224), ("resnet18", {}, 224), ("resnet50", {}, 224),
    ("squeezenet1_1", {}, 224), ("googlenet", {"aux_logits": False, "init_weights": False}, 224),
    ("inception_v3", {"aux_logits": False, "init_weights": False}, 299),
    ("mobilenet_v2", {}, 224), ("mobilenet_v3_small", {}, 224), ("mnasnet1_0", {}, 224),
    ("shufflenet_v2_x1_0", {}, 224), ("efficientnet_b0", {}, 224), ("regnet_y_400mf", {}, 224),
    ("convnext_tiny", {}, 224), ("densenet121", {}, 224), ("vit_b_16", {}, 224),
]

# Code idioms counted per program (regex over the source).
IDIOMS = {
    "several classes (≥3)": None,                     # computed, not a regex
    "helper functions (def at top level)": r"^def \w+",
    "nn.Sequential": r"nn\.Sequential\(",
    "nn.ModuleList": r"nn\.ModuleList\(",
    "OrderedDict / add_module": r"OrderedDict|add_module\(",
    "subclass of nn.Sequential": r"class \w+\(nn\.Sequential\)",
    "for-loop in __init__ / builder": r"^\s+for \w+",
    "_make_layer-style method": r"def _make_\w+",
    "config table (list of tuples/lists)": r"=\s*\[\s*\n?\s*[\[\(]\s*\d",
    "**kwargs": r"\*\*kwargs",
    "Optional / None default": r"Optional\[|=\s*None\b",
    "is None / is not None": r"is (not )?None",
    "functools.partial": r"partial\(",
    "Callable[...] (injected layer)": r"Callable\[",
    "torch.cat": r"torch\.cat\(",
    "F.* functional calls": r"\bF\.\w+\(",
    "tensor methods (.view/.reshape/.flatten/...)": r"\.(view|reshape|flatten|permute|transpose|chunk|mean)\(",
    "@staticmethod / @classmethod": r"@(staticmethod|classmethod)",
    "assert / torch._assert": r"\bassert\b|torch\._assert",
    "type annotations (-> Tensor)": r"->\s*Tensor",
    "_forward helper method": r"def _forward\b",
    "dropout": r"Dropout|dropout",
    "layer imported from torchvision (not in source)": r"torchvision\.ops|Conv2dNormActivation|SqueezeExcitation|StochasticDepth",
}


def short_params(node: dict) -> str:
    p, cls = node.get("params") or {}, node.get("cls") or ""
    if cls == "Linear":
        return f"{p.get('in')}→{p.get('out')}"
    if cls.startswith(("Conv", "ConvTranspose")):
        k = p.get("k")
        k = "x".join(map(str, k)) if isinstance(k, list) else k
        g = p.get("g") or p.get("groups")
        extra = f", s{p['s']}" if p.get("s") not in (None, 1, [1, 1]) else ""
        extra += f", g{g}" if g not in (None, 1) else ""
        return f"{p.get('in')}→{p.get('out')}, k{k}{extra}"
    if cls.startswith("BatchNorm") or cls == "GroupNorm":
        return str(p.get("features", ""))
    if cls == "LayerNorm":
        s = p.get("shape")
        return str(s[0]) if isinstance(s, list) and s else ""
    if "Pool" in cls and p.get("k") is not None:
        return f"k{p.get('k')}"
    return ""


def vocab_label(node: dict) -> str | None:
    """What an operation is, independent of where it sits: a module class, a function, a method."""
    if node["op"] == "call_module":
        return f"nn.{node['cls']}"
    if node["op"] == "call_function":
        t = node["target"]
        return t if "." in t else f"{t}()"
    if node["op"] == "call_method":
        return f".{node['target']}()"
    if node["op"] == "get_attr":
        return "get_attr"
    return None


def stats(graph: dict, code: str | None) -> dict:
    incoming: Counter = Counter()
    for e in graph["edges"]:
        incoming[e["dst"]] += len(e["args"])
    module_paths = [n["target"] for n in graph["nodes"] if n["op"] == "call_module"]
    g = to_networkx(graph)
    gens = list(nx.topological_generations(g)) if nx.is_directed_acyclic_graph(g) else []
    ops = Counter(n["op"] for n in graph["nodes"])
    n_nodes = len(graph["nodes"])
    out = {
        "nodes": n_nodes,
        "edges": len(graph["edges"]),
        "depth": nx.dag_longest_path_length(g) if gens else 0,
        "width": max((len(x) for x in gens), default=0),
        "merges": sum(1 for v in incoming.values() if v > 1),
        "fan_in": max(incoming.values(), default=0),
        "nesting": max((p.count(".") + 1 for p in module_paths), default=0),
        "modules": len(module_paths),
        "fn_share": round((ops["call_function"] + ops["call_method"]) / max(n_nodes, 1), 3),
        "distinct_ops": len({vocab_label(n) for n in graph["nodes"]} - {None}),
    }
    if code is not None:
        lines = [l for l in code.splitlines() if l.strip()]
        out["code_lines"] = len(lines)
        out["classes"] = len(re.findall(r"^class \w+", code, re.M))
        out["functions"] = len(re.findall(r"^def \w+", code, re.M))
    return out


def idioms(code: str) -> list[str]:
    found = []
    for name, pat in IDIOMS.items():
        if pat is None:
            if len(re.findall(r"^class \w+", code, re.M)) >= 3:
                found.append(name)
        elif re.search(pat, code, re.M):
            found.append(name)
    return found


_TOK = None


def ntok(text: str) -> int:
    """Qwen tokens when the tokenizer is cached locally, else characters / 3.3."""
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", local_files_only=True)
            _TOK = lambda s: len(tok(s)["input_ids"])
        except Exception:
            _TOK = lambda s: round(len(s) / 3.3)
    return _TOK(text)


def hier_graphs(h: dict) -> list[dict]:
    """The visible wiring graphs, in answer order, in the viewer's node/edge format."""
    from graphart.hier import _child, graph_text
    first = {}

    def walk(node):
        if node["kind"] == "opaque":
            return
        if node["kind"] == "block" and node["tag"] in h["variants"] and node["tag"] not in first:
            first[node["tag"]] = node
        for c in node["children"]:
            walk(c)
    walk(h["tree"])
    op = {"input": "placeholder", "output": "output", "module": "call_module", "function": "call_function",
          "method": "call_method", "get_attr": "get_attr"}
    out = []
    for tag, tnode in first.items():
        vnodes = h["variants"][tag]["nodes"]
        nodes, edges = [], {}
        for i, n in enumerate(vnodes):
            cls, params = "", ""
            if n["kind"] == "module":
                child = _child(tnode, n["target"])
                if child is not None:
                    cls = child["tag"] or child["cls"]
                    if child["kind"] == "leaf":
                        params = short_params({"cls": child["cls"], "params": child["params"]})
            nodes.append([i, op[n["kind"]], n["target"] + ("*" if n["pack"] else ""), cls, params, None])
            for slot, t in enumerate(n["src"]):
                src = int(re.match(r"\d+", t).group())
                edges.setdefault((src, i), []).append(slot)
        out.append({"tag": tag, "nodes": nodes, "edges": [[a, b, sl] for (a, b), sl in edges.items()],
                    "text": graph_text(vnodes)})
    return out


def pack(row_id: str, group: str, code: str, graph: dict, h: dict | None = None, *, family: str = "",
         split: str = "", code_is_file: bool = False) -> dict:
    from graphart.hier import to_text, tree_text
    from graphart.reduce import reduce_source
    nodes = [[n["id"], n["op"], n["target"], n.get("cls") or "", short_params(n),
              n.get("shape")] for n in graph["nodes"]]
    edges = [[e["src"], e["dst"], e["args"]] for e in graph["edges"]]
    try:
        reduced = reduce_source(code)
    except SyntaxError:
        reduced = code
    st = stats(graph, None if code_is_file else code)
    row = {
        "id": row_id, "group": group, "family": family, "split": split,
        "code": code, "code_reduced": reduced, "code_is_file": code_is_file,
        "nodes": nodes, "edges": edges,
        "vocab": sorted({vocab_label(n) for n in graph["nodes"]} - {None}),
        "idioms": [] if code_is_file else idioms(code),
    }
    st["code_tok"] = ntok(code)
    st["code_red_tok"] = ntok(reduced)
    if h is not None:
        answer = to_text(h)
        graphs = hier_graphs(h)
        row["tree"] = tree_text(h)
        row["hgraphs"] = graphs
        st["answer_tok"] = ntok(answer)
        st["tree_lines"] = len(row["tree"].splitlines())
        st["n_graphs"] = len(graphs)
        st["max_graph"] = max((len(g["nodes"]) for g in graphs), default=0)
        st["input_tok"] = st["code_red_tok"] + ntok(row["tree"])
    row["stats"] = st
    return row


def load_jsonl(name: str) -> list[dict]:
    path = DATA / f"{name}.jsonl"
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def generator_rows(sample: int = 0) -> list[dict]:
    from graphart.hier import hier_from_code
    rows = []
    src = [(split, r) for split in ("train", "val", "test") for r in load_jsonl(f"large_{split}")]
    if sample:
        src = random.Random(1).sample(src, sample)
    for split, r in src:
        h, _ = hier_from_code(r["code"])
        rows.append(pack(r["id"], f"gen-t{r['tier']}", r["code"], r["graph"], h,
                         family=r.get("family", ""), split=split))
    return rows


def fresh_rows(n: int, seed: int) -> list[dict]:
    from graphart.extract import graph_from_code
    from graphart.generate_large import FAMILIES, generate_large_program
    from graphart.hier import hier_from_code
    rng, rows, failed = random.Random(seed), [], 0
    families = sorted(FAMILIES)
    for tier in (5, 6):
        for i in range(n):
            fam = families[i % len(families)]
            code = generate_large_program(rng, tier, fam)
            try:
                graph, _ = graph_from_code(code)
                h, _ = hier_from_code(code)
            except Exception:
                failed += 1
                continue
            rows.append(pack(f"fresh-t{tier}-{i:03d}", f"fresh-t{tier}", code, graph, h, family=fam))
    print(f"fresh: {len(rows)} traced, {failed} failed to trace")
    return rows


def v3_rows(n: int, seed: int) -> list[dict]:
    """Programs from generator v3 (graphart/generate_v3.py): 70% networks, 30% blocks."""
    from graphart.extract import graph_from_code
    from graphart.generate_v3 import KINDS, generate_v3_program
    from graphart.hier import hier_from_code
    rng, rows, failed = random.Random(seed), [], 0
    for i in range(n):
        fam = KINDS[i % len(KINDS)]
        kind = "network" if rng.random() < 0.7 or fam == "vit" else "block"
        code = generate_v3_program(rng, kind, fam)
        try:
            graph, _ = graph_from_code(code)
            h, _ = hier_from_code(code)
        except Exception:
            failed += 1
            continue
        rows.append(pack(f"v3-{kind[:3]}-{fam}-{i:03d}", f"v3-{kind[:3]}", code, graph, h, family=fam))
    print(f"v3: {len(rows)} programs, {failed} failed to trace")
    return rows


def train_rows(n: int, seed: int) -> list[dict]:
    """Programs taken from the hierarchical training set itself (data/generated/hier_train.jsonl):
    generator v3 rows spread over families and network/block, and a few tiers 1-4 rows. The id
    names the round 2 patterns a program contains."""
    import collections
    from graphart.extract import graph_from_code
    rows = load_jsonl("hier_train")
    rng = random.Random(seed)
    rng.shuffle(rows)
    v3 = [r for r in rows if r["source"] == "v3"]
    by = collections.defaultdict(list)
    for r in v3:
        by[(r["family"], r["kind"])].append(r)
    keys = sorted(by)
    picked, i = [], 0
    while len(picked) < n and any(by.values()):
        k = keys[i % len(keys)]
        if by[k]:
            picked.append(by[k].pop())
        i += 1
    picked += [r for r in rows if r["source"] == "tiers14"][:max(2, n // 15)]

    def tags(code: str) -> str:
        t = []
        if "start=len(" in code:
            t.append("mixed")
        if "downsample=i > 0" in code:
            t.append("staged")
        if "down_type" in code:
            t.append("down_type")
        if re.search(r"\(self, block, ", code):
            t.append("block-arg")
        if "feature_info" in code:
            t.append("feature_info")
        if "grad_checkpointing" in code:
            t.append("ckpt")
        return ("-" + "-".join(t)) if t else ""

    out = []
    for r in picked:
        try:
            graph, _ = graph_from_code(r["code"])
        except Exception:
            continue
        group = "t14" if r["source"] == "tiers14" else f"v3-{r['kind'][:3]}"
        out.append(pack(f"{r['id']}-{r['family']}{tags(r['code'])}", group, r["code"], graph, r["hier"],
                        family=r["family"], split="train"))
    print(f"training set: {len(out)} programs")
    return out


def torchvision_rows() -> list[dict]:
    from graphart.hier import hier_from_code
    return [pack(r["id"], "tv-dev", r["code"], r["graph"], hier_from_code(r["code"])[0], family=r.get("kind", ""))
            for r in load_jsonl("torchvision")]


def _network_row(rid, group, family, model, builder, prefixes, example):
    from graphart.extract import graph_from_module
    from graphart.hier import hier_from_module
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_hier_eval import assemble
    code, _ = assemble(model, builder, "", prefixes)
    graph = graph_from_module(model, example_input=example)
    h, _ = hier_from_module(model, lambda c: c.__module__.startswith(prefixes), None)
    return pack(rid, group, code, graph, h, family=family)


def full_model_rows() -> list[dict]:
    """Whole torchvision networks, with their code assembled as in the dev set."""
    import torch
    import torchvision.models as tvm
    rows = []
    for name, kwargs, side in FULL_MODELS:
        try:
            torch.manual_seed(0)
            builder = tvm.get_model_builder(name)
            model = builder(weights=None, **kwargs)
            rows.append(_network_row(f"full-{name}", "tv-full", name, model, builder, ("torchvision.models",),
                                     torch.randn(1, 3, side, side)))
        except Exception as e:                      # a model fx cannot trace is reported, not fatal
            print(f"  full model {name}: skipped ({type(e).__name__}: {str(e)[:80]})")
    return rows


def timm_rows(n: int) -> list[dict]:
    """Real timm code of the kind used for training: one model from each of n allowed files
    (never a holdout architecture or a timm test family -- FROZEN_v2.json)."""
    import collections
    import torch
    import timm
    from timm.models import model_entrypoint
    frozen = json.loads((DATA / "FROZEN_v2.json").read_text(encoding="utf-8"))
    exclude = set(frozen["training_exclusions"]["timm_files"])
    by_file = collections.defaultdict(list)
    for m in timm.list_models():
        by_file[model_entrypoint(m).__module__.split(".")[-1]].append(m)
    files = [f for f in sorted(by_file) if f not in exclude]
    rows = []
    for f in files:
        if len(rows) >= n:
            break
        name = sorted(by_file[f], key=len)[0]
        try:
            torch.manual_seed(0)
            b = model_entrypoint(name)
            rows.append(_network_row(f"timm-{name}", "timm", f, b(), b, ("timm.models",), None))
        except Exception:
            continue
    print(f"timm: {len(rows)} programs")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/raw/generator_viewer.html")
    ap.add_argument("--fresh", type=int, default=0, help="also generate N new programs per tier")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-full", action="store_true", help="skip tracing whole torchvision models")
    ap.add_argument("--timm", type=int, default=25, help="real timm training-type programs to show (0 = none)")
    ap.add_argument("--sample", type=int, default=0, help="only this many generator programs (a light page)")
    ap.add_argument("--v3", type=int, default=0, help="also N programs from generator v3")
    ap.add_argument("--no-v2", action="store_true", help="leave out generator v2's dataset")
    ap.add_argument("--train-sample", type=int, default=0, help="also N programs from hier_train.jsonl")
    args = ap.parse_args()

    rows = [] if args.no_v2 else generator_rows(args.sample)
    if args.v3:
        rows += v3_rows(args.v3, args.seed)
    if args.train_sample:
        rows += train_rows(args.train_sample, args.seed)
    print(f"generator dataset: {len(rows)} programs")
    if args.fresh:
        rows += fresh_rows(args.fresh, args.seed)
    tv = torchvision_rows()
    print(f"torchvision dev: {len(tv)} programs")
    rows += tv
    if not args.no_full:
        rows += full_model_rows()
    if args.timm:
        rows += timm_rows(args.timm)

    payload = json.dumps({"rows": rows, "idioms": list(IDIOMS)}, separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB, {len(rows)} programs)")


if __name__ == "__main__":
    main()

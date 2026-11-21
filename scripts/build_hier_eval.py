"""Build the evaluation sets for the hierarchical schema, and freeze the two test sets.

    .venv/Scripts/python scripts/build_hier_eval.py            # build all three, write FROZEN_v2.json
    .venv/Scripts/python scripts/build_hier_eval.py --check    # rebuild, compare against FROZEN_v2.json

Writes data/generated/hier_{dev,holdout,timm_test}.jsonl. Each row is one program:

    id, set, family, kind ("network" | "block"), code   (what the model reads)
    target   the answer as text (graphart/hier.py to_text)
    hier     the full truth, including the hidden insides of modules whose source is not shown
    flat_nodes, code_chars, issues (empty when the lossless check passes)

The sets (decided 19 Nov 2026, docs/hierarchical_plan.md):

  dev        torchvision families already looked at: the 21 dev blocks (converted) and whole
             networks of those families and of the old holdout's families
  holdout    torchvision families never looked at: Swin (v1, v2), MaxViT, MViT, Swin3D, FCN,
             LRASPP -- whole networks, every constructor. Frozen; scored once.
  timm_test  three timm families never trained on: DLA, CSPNet, GhostNet, two sizes each (backups
             in a fixed order: HRNet, ReXNet, SelecSLS, used only if one fails to trace losslessly).
             Frozen; scored once.

A network's code is assembled from the real source: every class of the model that lives in
a torchvision.models / timm.models file, the functions and classes those reference, and the
builder function, file by file, with each file's import lines; then `model = <builder>(...)`.
Classes from torchvision.ops / timm.layers are not shown, so they are opaque boxes. The truth
comes from tracing the real object, with "in the source" = the files shown.

The script prints counts and sizes only -- it never prints holdout or timm-test content.
"""

import argparse
import ast
import enum
import hashlib
import inspect
import json
import re
import sys
import warnings
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import torch
from torch import nn

from graphart.hier import check_lossless, hier_from_code, hier_from_module, to_text

DATA = ROOT / "data" / "generated"
FROZEN = DATA / "FROZEN_v2.json"

# (builder, constructor arguments as written in the sample)
DEV_NETWORKS = [
    ("alexnet", ""), ("vgg11", ""), ("vgg16_bn", ""), ("resnet18", ""), ("resnet50", ""),
    ("resnext50_32x4d", ""), ("wide_resnet50_2", ""), ("squeezenet1_0", ""), ("squeezenet1_1", ""),
    ("googlenet", ""), ("inception_v3", ""), ("mobilenet_v2", ""), ("mobilenet_v3_small", ""),
    ("mobilenet_v3_large", ""), ("mnasnet0_5", ""), ("mnasnet1_0", ""), ("shufflenet_v2_x1_0", ""),
    ("efficientnet_b0", ""), ("efficientnet_v2_s", ""), ("regnet_x_400mf", ""), ("regnet_y_400mf", ""),
    ("convnext_tiny", ""), ("densenet121", ""), ("vit_b_16", ""),
    ("deeplabv3_resnet50", "weights_backbone=None"), ("deeplabv3_mobilenet_v3_large", "weights_backbone=None"),
    ("r3d_18", ""), ("mc3_18", ""), ("r2plus1d_18", ""), ("s3d", ""),
]
HOLDOUT_FAMILIES = {
    "swin": ["swin_t", "swin_s", "swin_b"],
    "swin_v2": ["swin_v2_t", "swin_v2_s", "swin_v2_b"],
    "maxvit": ["maxvit_t"],
    "mvit": ["mvit_v1_b", "mvit_v2_s"],
    "swin3d": ["swin3d_t", "swin3d_s", "swin3d_b"],
    "fcn": ["fcn_resnet50", "fcn_resnet101"],
    "lraspp": ["lraspp_mobilenet_v3_large"],
}
HOLDOUT_ARGS = {"fcn": "weights_backbone=None", "lraspp": "weights_backbone=None"}
TIMM_TEST = [("dla", ["dla34", "dla60"]), ("cspnet", ["cspresnet50", "cspdarknet53"]),
             ("ghostnet", ["ghostnet_100", "ghostnetv2_100"])]
TIMM_BACKUPS = [("hrnet", ["hrnet_w18_small", "hrnet_w18"]), ("rexnet", ["rexnet_100", "rexnet_150"]),
                ("selecsls", ["selecsls42b", "selecsls60"])]
# timm files never used for training: the holdout's architectures and the timm test families
TIMM_TRAIN_EXCLUDE_ARCH = ["swin_transformer", "swin_transformer_v2", "swin_transformer_v2_cr",
                           "maxxvit", "mvitv2"]


# --------------------------------------------------------------------------
# assembling a network's source
# --------------------------------------------------------------------------
def _defs(modname: str) -> tuple[dict, list[str], str]:
    """Top-level defs of a module: {name: (start, body_start, end, docstring line ranges)},
    import lines, source."""
    src = inspect.getsource(sys.modules[modname])
    tree = ast.parse(src)
    defs, imports = {}, []
    lines = src.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            docs = [(n.body[0].lineno, n.body[0].end_lineno) for n in ast.walk(node)
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.body
                    and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
                    and isinstance(n.body[0].value.value, str) and len(n.body) > 1]
            defs[node.name] = (start, node.lineno, node.end_lineno, docs)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # module-level constants (config tables such as VGG's `cfgs`, timm's `model_cfgs`):
            # shown when a shown def uses them (added 21 Nov, FROZEN_v2 amendment 1)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id not in defs:
                    defs[t.id] = (node.lineno, node.lineno, node.end_lineno, [])
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
    return defs, imports, src


def _shown(cls_or_fn, prefixes) -> bool:
    return getattr(cls_or_fn, "__module__", "").startswith(prefixes)


def _names(text: str) -> set[str]:
    """Identifiers used in code -- not words inside strings or comments."""
    import io
    import tokenize
    try:
        return {t.string for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type == tokenize.NAME}
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return set(re.findall(r"\b[A-Za-z_]\w*\b", text))


def _def_text(lines, d, *, decorators=True) -> str:
    """A def's source without its docstrings (and, for scanning, without decorators)."""
    start, body, end, docs = d
    skip = {i for a, b in docs for i in range(a, b + 1)}
    first = start if decorators else body
    return "\n".join(lines[i - 1] for i in range(first, end + 1) if i not in skip)


def assemble(model: nn.Module, builder, call: str, prefixes: tuple) -> tuple[str, set]:
    """Source of the model as a reader would need it; returns (code, modules shown).

    Shown files: those defining a class the model is built from, and the builder's file --
    not infrastructure (registries, weight loaders), whose code a reader does not need.
    References are followed within the shown files only, and not from decorator lines.
    Docstrings are dropped (torchvision builders carry long weight documentation)."""
    want: set[tuple[str, str]] = set()
    for m in model.modules():
        if _shown(type(m), prefixes):
            for k in type(m).__mro__:
                if _shown(k, prefixes) and issubclass(k, nn.Module):
                    want.add((k.__module__, k.__name__))
    want.add((builder.__module__, builder.__name__))
    shown_mods = {m for m, _ in want}
    cache = {}

    def get(mod):
        if mod not in cache:
            cache[mod] = _defs(mod)
        return cache[mod]

    todo = list(want)
    while todo:
        mod, name = todo.pop()
        defs, _, src = get(mod)
        if name not in defs:
            continue
        text = _def_text(src.splitlines(), defs[name], decorators=False)
        namespace = vars(sys.modules[mod])
        for ident in _names(text):
            if ident == name:
                continue
            obj = namespace.get(ident)
            # Functions and classes defined in a shown file -- nn.Modules and plain classes alike
            # (config dataclasses such as torchvision's MBConvConfig, which alone names MBConv);
            # never Enums: torchvision's weight enums are infrastructure, pages of metadata.
            # (Until 21 Nov only functions and nn.Module classes were followed: FROZEN_v2 amendment 1.)
            if (inspect.isfunction(obj) or (inspect.isclass(obj) and not issubclass(obj, enum.Enum))) \
                    and getattr(obj, "__module__", None) in shown_mods:
                key = (obj.__module__, obj.__name__)
                if key not in want and obj.__name__ in get(obj.__module__)[0]:
                    want.add(key)
                    todo.append(key)
                if ident != obj.__name__ and ident in defs and (mod, ident) not in want:
                    want.add((mod, ident))             # an alias: `_Outputs = Outputs`
                    todo.append((mod, ident))
            # module-level constants of the same file that this def reads
            elif ident in defs and not inspect.isclass(obj) and not inspect.isfunction(obj) \
                    and not inspect.ismodule(obj) and (mod, ident) not in want:
                want.add((mod, ident))
                todo.append((mod, ident))
    mods = sorted(shown_mods, key=lambda m: (m == builder.__module__, m == type(model).__module__, m))
    parts = []
    for mod in mods:
        defs, imports, src = get(mod)
        lines = src.splitlines()
        names = sorted((defs[n][0], n) for m, n in want if m == mod and n in defs)
        parts.append(f"# ---- {mod.replace('.', '/')}.py ----")
        parts += imports
        parts.append("")
        done = set()
        for start, n in names:
            if start in done:                  # `a = b = ...`: one statement, two names
                continue
            done.add(start)
            parts.append(_def_text(lines, defs[n]))
            parts.append("")
    parts.append(f"model = {builder.__name__}({call})")
    return "\n".join(parts) + "\n", set(mods)


# --------------------------------------------------------------------------
# samples
# --------------------------------------------------------------------------
def network_row(set_name, family, name, builder, call, prefixes) -> dict:
    torch.manual_seed(0)
    model = eval(f"builder({call})", {"builder": builder, "None": None})
    code, _ = assemble(model, builder, call, prefixes)
    h, flat = hier_from_module(model, lambda c: _shown(c, prefixes), None)
    return _row(set_name, f"{set_name}-{name}", family, "network", code, h, flat)


def _row(set_name, rid, family, kind, code, h, flat) -> dict:
    return {"id": rid, "set": set_name, "family": family, "kind": kind, "code": code,
            "target": to_text(h), "hier": h, "flat_nodes": len(flat["nodes"]),
            "code_chars": len(code), "issues": check_lossless(h, flat)}


def build_dev() -> list[dict]:
    import torchvision.models as tvm
    rows = []
    for line in (DATA / "torchvision.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        h, flat = hier_from_code(r["code"])
        rows.append(_row("dev", r["id"], r["id"].split("-")[1], "block", r["code"], h, flat))
    for name, call in DEV_NETWORKS:
        rows.append(network_row("dev", name.split("_")[0], name, tvm.get_model_builder(name), call,
                                ("torchvision.models",)))
    return rows


def build_holdout() -> list[dict]:
    import torchvision.models as tvm
    rows = []
    for fam, names in HOLDOUT_FAMILIES.items():
        for name in names:
            rows.append(network_row("holdout", fam, name, tvm.get_model_builder(name),
                                    HOLDOUT_ARGS.get(fam, ""), ("torchvision.models",)))
    return rows


def build_timm_test() -> tuple[list[dict], list[str]]:
    from timm.models import model_entrypoint
    rows, used, pool = [], [], list(TIMM_TEST)
    backups = list(TIMM_BACKUPS)
    while pool and len(used) < 3:
        fam, names = pool.pop(0)
        try:
            fam_rows = [network_row("timm_test", fam, n, model_entrypoint(n), "", ("timm.models",)) for n in names]
            ok = all(not r["issues"] for r in fam_rows)
        except Exception as e:
            print(f"  timm {fam}: failed to build ({type(e).__name__})")
            ok = False
        if ok:
            rows += fam_rows
            used.append(fam)
        else:
            print(f"  timm {fam}: not lossless -> replaced by the next backup")
            if backups:
                pool.append(backups.pop(0))
    return rows, used


def write(rows, name) -> Path:
    path = DATA / f"hier_{name}.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(name, rows):
    bad = [r for r in rows if r["issues"]]
    fl = sorted(r["flat_nodes"] for r in rows)
    cc = sorted(r["code_chars"] for r in rows)
    print(f"{name:10s} {len(rows):3d} programs, {len(bad)} not lossless; flat nodes median {fl[len(fl) // 2]} "
          f"max {fl[-1]}; code chars median {cc[len(cc) // 2]} max {cc[-1]}")
    if bad and name == "dev":
        for r in bad:
            print("   ", r["id"], r["issues"][:2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="rebuild and compare hashes with FROZEN_v2.json")
    ap.add_argument("--dry", action="store_true", help="build and report; do not write FROZEN_v2.json")
    ap.add_argument("--refreeze", metavar="REASON",
                    help="rebuild the frozen sets after a fix to how they are built, and append a dated "
                         "amendment with the old and new hashes -- only before any reply on them exists")
    args = ap.parse_args()
    if not (args.check or args.refreeze or args.dry) and FROZEN.exists():
        raise SystemExit("FROZEN_v2.json exists: the test sets are frozen. Use --check (or --refreeze).")

    import timm
    import torchvision
    dev = build_dev()
    summary("dev", dev)
    holdout = build_holdout()
    summary("holdout", holdout)
    timm_rows, timm_used = build_timm_test()
    summary("timm_test", timm_rows)

    paths = {"dev": write(dev, "dev"), "holdout": write(holdout, "holdout"), "timm_test": write(timm_rows, "timm_test")}
    if args.check:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        for k in ("holdout", "timm_test"):
            same = sha256(paths[k]) == frozen[k]["sha256"]
            print(f"{k}: {'matches' if same else 'DIFFERS FROM'} the frozen hash")
        return

    if args.dry:
        print("dry run: FROZEN_v2.json not written")
        return
    if args.refreeze:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        change = {"date": date.today().isoformat(), "kind": "re-freeze (rebuild)", "reason": args.refreeze,
                  "graphart_hier_sha256": sha256(ROOT / "graphart" / "hier.py"),
                  "scripts_build_hier_eval_sha256": sha256(Path(__file__))}
        for k, rows in (("holdout", holdout), ("timm_test", timm_rows)):
            change[k] = {"old_sha256": frozen[k]["sha256"], "new_sha256": sha256(paths[k]), "n": len(rows),
                         "lossless": sum(1 for r in rows if not r["issues"])}
            frozen[k].update(sha256=sha256(paths[k]), n=len(rows), lossless=change[k]["lossless"])
        frozen["dev"]["n"] = len(dev)
        frozen["amendments"].append(change)
        FROZEN.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        print(f"amendment {len(frozen['amendments'])} appended to {FROZEN.relative_to(ROOT)}: "
              + ", ".join(f"{k} {change[k]['old_sha256'][:12]} -> {change[k]['new_sha256'][:12]}"
                          for k in ("holdout", "timm_test")))
        return
    hier_py = ROOT / "graphart" / "hier.py"
    frozen = {
        "created": date.today().isoformat(),
        "schema": "hierarchical v1: module tree + one wiring graph per class forward() (graphart/hier.py)",
        "graphart_hier_sha256": sha256(hier_py),
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__},
        "holdout": {
            "file": "data/generated/hier_holdout.jsonl", "sha256": sha256(paths["holdout"]), "n": len(holdout),
            "families": HOLDOUT_FAMILIES,
            "lossless": sum(1 for r in holdout if not r["issues"]),
        },
        "timm_test": {
            "file": "data/generated/hier_timm_test.jsonl", "sha256": sha256(paths["timm_test"]),
            "n": len(timm_rows), "families": timm_used,
            "lossless": sum(1 for r in timm_rows if not r["issues"]),
            "rule": "DLA, CSPNet, GhostNet (two sizes each); a family that does not trace losslessly is "
                    "replaced by the next of HRNet, ReXNet, SelecSLS -- fixed before building",
        },
        "dev": {"file": "data/generated/hier_dev.jsonl", "n": len(dev),
                "note": "development data; not frozen"},
        "training_exclusions": {
            "timm_files": TIMM_TRAIN_EXCLUDE_ARCH + timm_used,
            "why": "holdout architectures (Swin, MaxViT, MViT) and the timm test families are never trained on",
            "generator": "programs whose variant graphs are isomorphic to a whole holdout/timm-test program's are rejected",
        },
        "rules": [
            "Holdout and timm_test are scored ONCE, on the frozen setup, after a dated amendment to this file "
            "fixes the pipeline, the models and the comparisons -- written before any reply on them exists.",
            "Never tune prompts, data, the generator or training on them; never generate replies on them before "
            "the amendment; never print or read their content while developing (the builder prints counts only).",
            "The 17-sample flat holdout (FROZEN.json) stays spent and is not part of this protocol.",
            "Caveat recorded at freezing: the holdout constructors were traced once on 19 Nov in a count of all "
            "torchvision models (counts only, no content looked at).",
        ],
        "amendments": [],
    }
    FROZEN.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {FROZEN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Ask a model for the hierarchical answer (graphart/hier.py): the tree, then one wiring graph
per block variant, assembled by code; save the replies.

    python -m pipeline.run_hier --backend hf --model lora/qwen2.5-coder-1.5b-graphart-lora-hier --split hier_dev
    python -m pipeline.run_hier --backend ollama --model qwen2.5-coder:14b --num-ctx 20480 --split hier_dev
    python -m pipeline.score_hier results/hf__qwen2.5-coder-1.5b-graphart-lora-hier__hier__hier_dev.jsonl

Per program (docs/hierarchical_plan.md, "Pipeline"):
  1. the `hier_tree` prompt on the reduced code -> the module tree;
  2. for every variant tag in THAT tree (pre-order, first instance), the `hier_wire` prompt
     with the class's code and the instance's subtree as the model's own tree writes it ->
     the variant's wiring graph. A tag whose class is not in the code gets no call;
  3. code joins them into TREE + GRAPH sections (the answer format of hier.to_text).
Never the reference tree: step 2 sees exactly what step 1 produced, as in training's
inference story. `--oracle-tree` is the one exception, for measuring wiring alone: step 2
gets the truth's variants and subtrees, and the saved answer is the truth's tree with the
model's graphs, so only wiring can be wrong.

Replies are cached under results/raw/<run>/<id>.tree.txt and <id>.wire.<tag>.txt, so an
interrupted run resumes. The frozen sets (hier_holdout, hier_timm_test; FROZEN_v2.json) are
refused unless --frozen is given: they are scored once, on the final model, after a dated
amendment.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphart.hier import HierError, _parse_tree, to_text, tree_text
from graphart.hier_tasks import (TREE_PROMPT, WIRE_PROMPT, _fill, class_source, instance_subtree,
                                 wiring_tasks)
from graphart.reduce import reduce_source
from pipeline import cost as cost_module
from pipeline.backends import BackendError, DEFAULT_MODELS, make_backend
from pipeline.run_prompting import RESULTS, ROOT, load_prompt

DATA = ROOT / "data" / "generated"
FROZEN = ("hier_holdout", "hier_timm_test")


def clean(reply: str) -> list[str]:
    """Reply lines without code fences and blank lines."""
    return [l.rstrip() for l in reply.strip().splitlines() if l.strip() and not l.strip().startswith("```")]


def tree_lines(reply: str) -> list[str]:
    """The tree part of a tree reply: from TREE (if written) up to any GRAPH line."""
    lines = clean(reply)
    if lines and lines[0].strip() == "TREE":
        lines = lines[1:]
    end = next((i for i, l in enumerate(lines) if l.startswith("GRAPH")), len(lines))
    return lines[:end]


def graph_lines(reply: str) -> list[str]:
    """The graph part of a wiring reply: numbered lines; a GRAPH header the model repeats is
    dropped, and anything after a second header is ignored."""
    lines = clean(reply)
    if lines and lines[0].startswith("GRAPH"):
        lines = lines[1:]
    end = next((i for i, l in enumerate(lines) if l.startswith("GRAPH")), len(lines))
    return lines[:end]


def block_tags(tree: dict) -> list[str]:
    """Variant tags of block instances outside opaque boxes, in pre-order. `Class#0` marks a
    module that is built but never run (hier.py) -- it has no graph, so it gets no call."""
    seen = []

    def walk(node):
        if node["kind"] == "opaque":
            return
        if (node["kind"] == "block" and node.get("tag") and node["tag"] not in seen
                and not node["tag"].endswith("#0")):
            seen.append(node["tag"])
        for c in node["children"]:
            walk(c)
    walk(tree)
    return seen


# How each layer's parameters are written (graphart/extract.py _layer_params), for prompted
# models: the worked examples cannot show every layer. Tuned adapters learnt it from data.
LAYER_FORMATS = """Layer parameters are written exactly like this (any other torch.nn layer by its name alone, e.g. ReLU):
Conv1d/Conv2d/Conv3d/ConvTranspose2d(in=,out=,k=,s=,p=) plus g= only when groups is not 1; Linear(in=,out=,bias=True|False);
BatchNorm1d/BatchNorm2d/BatchNorm3d(features=); GroupNorm(features=<channels>); LayerNorm(shape=[64]);
MaxPool2d/AvgPool2d(k=,s=); AdaptiveAvgPool2d/AdaptiveMaxPool2d(out=); Dropout(p=); Embedding(num=,dim=); Flatten(start=,end=).
A kernel, stride or padding that is the same in every dimension is one number (k=3), otherwise a list with x between the values (k=[1x7]); LayerNorm over two dims is shape=[64x64]."""


def shots(k: int, ttpl: str, wtpl: str) -> tuple[str, str]:
    """Worked examples for a prompted (untuned) model, from the TRAIN split only, chosen by a
    fixed rule: k=1 the smallest generator network with a collapsed repeat (`0-2:`) and >= 3
    variants; k=2 adds the smallest generator block with >= 2 variants first. The wiring
    examples are that network's first two variant graphs. Returns (tree prefix, wire prefix)."""
    from graphart.hier_tasks import tree_task
    if k <= 0:
        return "", ""
    saved = ROOT / "prompts" / f"hier_shots{k}.json"      # written once here, shipped to Kaggle
    if saved.exists():
        d = json.loads(saved.read_text(encoding="utf-8"))
        return d["tree"], d["wire"]
    rows =[json.loads(l) for l in (DATA / "hier_train.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["source"] == "v3"]
    rows.sort(key=lambda r: (r["tree_task_tokens"], r["id"]))

    def n_tags(r):
        return len(block_tags(r["hier"]["tree"]))
    net = next(r for r in rows if r["kind"] == "network" and n_tags(r) >= 3
               and any("-" in l.split(":")[0] for l in tree_text(r["hier"]).splitlines()))
    picks = ([next(r for r in rows if r["kind"] == "block" and n_tags(r) >= 2)] if k >= 2 else []) + [net]
    tparts = []
    for i, r in enumerate(picks, 1):
        t = tree_task(r["code"], ttpl, r["hier"])
        tparts.append(f"Worked example {i}:\n{t['user']}\n{t['answer']}")
    wparts = []
    for i, w in enumerate(wiring_tasks(net["code"], net["hier"], wtpl)[:2], 1):
        wparts.append(f"Worked example {i}:\n{w['user']}\n{w['answer']}")
    tail = "\n\nNow the task:\n"
    out = {"tree": LAYER_FORMATS + "\n\n" + "\n\n".join(tparts) + tail, "wire": "\n\n".join(wparts) + tail,
           "from_train": [r["id"] for r in picks]}
    saved.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    return out["tree"], out["wire"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="mock", choices=["anthropic", "openrouter", "ollama", "hf", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--split", default="hier_dev", help="a data/generated/<split>.jsonl with code + hier")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="only these program ids, comma-separated")
    ap.add_argument("--tree-tokens", type=int, default=8192, help="output cap of the tree call")
    ap.add_argument("--wire-tokens", type=int, default=2048, help="output cap of each wiring call")
    ap.add_argument("--num-ctx", type=int, default=None, help="Ollama only: context window")
    ap.add_argument("--oracle-tree", action="store_true",
                    help="wiring calls get the truth's variants and subtrees (wiring measured alone)")
    ap.add_argument("--shots", type=int, default=0,
                    help="worked examples from the train split before each prompt (prompted models; 0 for adapters)")
    ap.add_argument("--part", help="suffix of the result file, so two processes can split one run (merge after)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--yes", action="store_true", help="required to proceed when the run would cost money")
    ap.add_argument("--frozen", action="store_true",
                    help="allow a frozen test set (FROZEN_v2.json): the one scored run of the final model")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # replies may hold any character

    if args.split in FROZEN and not args.frozen:
        raise SystemExit(f"{args.split} is frozen (data/generated/FROZEN_v2.json): it is scored once, on the "
                         "final model, after a dated amendment. Pass --frozen only for that run.")
    path = DATA / f"{args.split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in want]
    rows = rows[:args.limit] if args.limit else rows

    (tsys, ttpl), (wsys, wtpl) = load_prompt(TREE_PROMPT), load_prompt(WIRE_PROMPT)
    model_name = args.model or DEFAULT_MODELS[args.backend]
    # priced as one call per program with the whole answer; the wiring calls add their inputs
    est_rows = [{"code": reduce_source(r["code"]), "target": r.get("target") or to_text(r["hier"])} for r in rows]
    estimate = cost_module.estimate(args.backend, model_name, est_rows, tsys, ttpl)
    if args.cached_only:
        print("  --cached-only: no provider is contacted, so this cannot cost anything.")
    else:
        print(cost_module.render(estimate, args.backend, model_name))
        print("  (input counted for the tree calls only; the wiring calls add theirs)")
    if not args.cached_only and not cost_module.is_free(estimate) and not args.yes:
        raise SystemExit("\n  STOPPED: this run would cost money (or its price is unknown).\n"
                         "  Nothing has been sent and nothing has been charged.\n"
                         "  Re-run with --yes only if you accept the cost above.\n")
    extra = {} if args.num_ctx is None else {"num_ctx": args.num_ctx}
    try:
        backend = None if args.cached_only else make_backend(args.backend, args.model, effort=args.effort,
                                                             max_tokens=args.tree_tokens, **extra)
    except BackendError as exc:
        raise SystemExit(f"\n  {exc}\n")
    slug = (backend.model if backend else model_name).replace(":", "-").replace("/", "-")
    prompt_name = ("hier_oracle_tree" if args.oracle_tree else "hier") + (f"_{args.shots}shot" if args.shots else "")
    tshot, wshot = shots(args.shots, ttpl, wtpl)
    run_id = f"{args.backend}__{slug}__{prompt_name}__{args.split}"
    cache_dir = RESULTS / "raw" / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    def ask(name: str, system: str, user: str, cap: int) -> tuple[str, bool]:
        p = cache_dir / name
        if p.exists() and not args.refresh:
            return p.read_text(encoding="utf-8"), True
        if args.cached_only:
            raise LookupError
        if hasattr(backend, "max_tokens"):
            backend.max_tokens = cap
        reply = backend(system, user)
        p.write_text(reply, encoding="utf-8")
        return reply, False

    out = RESULTS / (f"{run_id}__part{args.part}.jsonl" if args.part else f"{run_id}.jsonl")

    def save() -> None:
        """Results so far, rewritten after every program: a run stopped by a time limit keeps its work."""
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        meta = {"backend": args.backend, "model": (backend.model if backend else model_name), "prompt": prompt_name,
                "split": args.split, "n": len(records), "of": len(rows), "failed": failed, "calls": calls,
                "cached_calls": cached_calls, "tree_tokens": args.tree_tokens, "wire_tokens": args.wire_tokens,
                "num_ctx": args.num_ctx}
        out.with_name(out.stem + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    records, failed, calls, cached_calls = [], 0, 0, 0
    for i, row in enumerate(rows, 1):
        started, error, reply = time.time(), None, ""
        try:
            code = row["code"]
            reduced = reduce_source(code)
            graphs = {}
            if not args.oracle_tree:              # the oracle run takes the truth's tree: no tree call
                tree_reply, hit = ask(f"{row['id']}.tree.txt", tsys, tshot + _fill(ttpl, code=reduced.strip()),
                                      args.tree_tokens)
                calls += 1
                cached_calls += hit
                lines = tree_lines(tree_reply)
            if args.oracle_tree:
                truth = row["hier"]
                tree_part = tree_text(truth)
                for w in wiring_tasks(code, truth, wtpl, with_answers=False):
                    g, hit = ask(f"{row['id']}.wire.{w['tag']}.txt", wsys, wshot + w["user"], args.wire_tokens)
                    calls += 1
                    cached_calls += hit
                    graphs[w["tag"]] = graph_lines(g)
            else:
                tree_part = "\n".join(lines)
                try:
                    tree = _parse_tree(lines)
                except HierError as e:
                    raise ValueError(f"tree reply unreadable: {e}")
                h = {"tree": tree, "variants": {}, "problems": []}
                for tag in block_tags(tree):
                    src = class_source(reduced, tag.split("#")[0])
                    if not src:
                        continue                 # a class not in the code: no call, no graph
                    user = _fill(wtpl, cls=tag.split("#")[0], code=src.strip(),
                                 subtree=instance_subtree(h, tag), tag=tag)
                    g, hit = ask(f"{row['id']}.wire.{tag}.txt", wsys, wshot + user, args.wire_tokens)
                    calls += 1
                    cached_calls += hit
                    graphs[tag] = graph_lines(g)
            reply = "\n".join(["TREE", tree_part] + [x for tag, gl in graphs.items()
                                                     for x in ["", f"GRAPH {tag}", *gl]])
        except LookupError:
            continue                             # --cached-only and not on disk
        except BackendError as exc:
            raise SystemExit(f"\n{exc}")
        except Exception as exc:                 # one bad program must not lose the rest
            error = f"{type(exc).__name__}: {exc}"
            failed += 1
        elapsed = time.time() - started
        if not args.cached_only:
            print(f"  [{i}/{len(rows)}] {row['id']} {elapsed:6.1f}s {'FAILED: ' + error[:160] if error else ''}",
                  flush=True)
        records.append({"id": row["id"], "kind": row.get("kind"), "family": row.get("family"),
                        "source": row.get("source", row.get("set")), "reply": reply,
                        "elapsed_s": round(elapsed, 2), "error": error})
        save()

    save()
    print(f"\n{len(records)} programs, {calls} calls ({cached_calls} from cache), {failed} failed -> {out}")
    print(f"score with:  python -m pipeline.score_hier {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

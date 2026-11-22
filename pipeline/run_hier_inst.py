"""The hierarchical answer built per instance (graphart/hier_inst.py), then wired; save it.

    python -m pipeline.run_hier_inst --tree-model lora/qwen2.5-coder-1.5b-graphart-lora-hier-inst \\
        --wire-model lora/qwen2.5-coder-1.5b-graphart-lora-hier --split hier_dev
    python -m pipeline.score_hier results/hf__qwen2.5-coder-1.5b-graphart-lora-hier-inst__hier_inst__hier_dev.jsonl

Per program:
  1. the root call (`hier_inst` prompt, the whole reduced program) -> the root's children;
     every child written `Class(args)` for a class in the code gets its own call (its class's
     code, its arguments), once per distinct `Class(args)`; code joins the answers into the
     module tree (hier_inst.assemble);
  2. wiring, deepest block instances first: one `hier_wire` call per distinct instance, with
     its subtree as built (descendants already tagged); two instances of a class whose graphs
     are the same share a variant tag (as graphart/hier.py numbers variants);
  3. a block that its nearest block ancestor's graph never calls (nor a container holding it)
     is built but never run: `Class#0`, no graph -- again as hier.py marks it;
  4. TREE + GRAPH text (hier.to_text), scored by pipeline/score_hier.py like any run.
Never the truth: every step sees only what the model wrote. Replies are cached under
results/raw/<run>/; results are saved after every program. The frozen sets need --frozen.
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphart.hier import HierError, _parse_graph, _subtree_lines, _tree_desc, to_text
from graphart.hier_inst import EMPTY, INST_PROMPT, ROOT_LABEL, assemble, inst_source
from graphart.hier_score import same_wiring
from graphart.hier_tasks import WIRE_PROMPT, _fill, class_source
from graphart.reduce import reduce_source
from pipeline.backends import BackendError, make_backend
from pipeline.run_hier import FROZEN, graph_lines
from pipeline.run_prompting import RESULTS, ROOT, load_prompt

DATA = ROOT / "data" / "generated"


def class_kinds(reduced: str) -> dict[str, str]:
    """Classes defined in the code -> 'chain' (an nn.Sequential subclass without its own
    forward), 'list' (ModuleList / ModuleDict likewise) or 'block'. Read from the code, not run."""
    out = {}
    for node in ast.parse(reduced).body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {ast.unparse(b).split(".")[-1] for b in node.bases}
        own_forward = any(isinstance(f, ast.FunctionDef) and f.name == "forward" for f in node.body)
        if "Sequential" in bases and not own_forward:
            out[node.name] = "chain"
        elif bases & {"ModuleList", "ModuleDict"} and not own_forward:
            out[node.name] = "list"
        else:
            out[node.name] = "block"
    return out


def walk(node: dict, path: str = ""):
    yield path, node
    if node["kind"] in ("leaf", "opaque"):
        return
    for c in node["children"]:
        yield from walk(c, f"{path}.{c['name']}" if path else c["name"])


def build_answer(reduced: str, ask_inst, ask_wire, max_calls: int = 400) -> tuple[str, dict]:
    """(TREE + GRAPH text, counts) for one program. `ask_inst(path, head, code)` and
    `ask_wire(path, cls, code, subtree, tag)` return reply text."""
    kinds = class_kinds(reduced)
    calls = {"inst": 1, "wire": 0}

    def ask(path, desc):
        cls = desc.split("(", 1)[0]
        src = inst_source(reduced, cls)
        if not src:
            return EMPTY
        calls["inst"] += 1
        return ask_inst(path, f"{path}: {desc}", src)

    root_reply = ask_inst("", ROOT_LABEL, reduced)
    tree = assemble(root_reply, ask, max_calls=max_calls, defined=lambda c: c in kinds)

    # kinds: classes in the code that are containers are chains / lists, not blocks
    for _, node in walk(tree):
        if node["kind"] == "block" and node is not tree and kinds.get(node["cls"]) in ("chain", "list"):
            node["kind"] = kinds[node["cls"]]
        # a class in the code written by bare name has no __init__ of its own (e.g. LayerNorm2d
        # over nn.LayerNorm): no children, but its own forward(), so it is a block to wire
        elif node["kind"] == "opaque" and kinds.get(node["cls"]) == "block" and "desc" not in node:
            node["kind"] = "block"
            node["desc"] = node["cls"]
    tree["kind"] = "block"

    # wiring, deepest first; tags by graph per class
    blocks = [(p, n) for p, n in walk(tree) if n["kind"] == "block" and n["cls"] in kinds]
    blocks.sort(key=lambda pn: -pn[0].count(".") - (1 if pn[0] else 0))
    variants: dict[str, dict] = {}
    per_class: dict[str, list[str]] = {}
    tag_of: dict[str, str] = {}                      # description -> tag
    for path, node in blocks:
        key = node["cls"] + "|" + node.get("desc", "<root>")
        if key in tag_of:
            node["tag"] = tag_of[key]
            continue
        cls = node["cls"]
        provisional = f"{cls}#{len(per_class.get(cls, [])) + 1}"
        node["tag"] = provisional
        head = f"{node['name']}: {_tree_desc(node)}" if node["name"] else _tree_desc(node)
        subtree = "\n".join([head] + _subtree_lines(node, 1))
        src = class_source(reduced, cls)
        reply = ask_wire(path, cls, src, subtree, provisional)
        calls["wire"] += 1
        try:
            nodes = _parse_graph(graph_lines(reply))
        except HierError:
            nodes = None
        tag = None
        if nodes is not None:
            for t in per_class.get(cls, []):
                if same_wiring(variants[t]["nodes"], nodes):
                    tag = t
                    break
        if tag is None:
            tag = provisional
            per_class.setdefault(cls, []).append(tag)
            variants[tag] = {"nodes": nodes or [], "hidden": False}
        node["tag"] = tag
        tag_of[key] = tag

    # never run: a block its nearest block ancestor's graph does not reach
    def reached(anc_path, anc, path):
        rel = path[len(anc_path) + 1:] if anc_path else path
        targets = [n["target"] for n in variants.get(anc["tag"], {}).get("nodes", []) if n["kind"] == "module"]
        return any(rel == t or rel.startswith(t + ".") for t in targets)

    def never_run(node, path, anc, anc_path, dead):
        for c in node["children"] if node["kind"] not in ("leaf", "opaque") else []:
            cp = f"{path}.{c['name']}" if path else c["name"]
            if c["kind"] == "block" and c.get("tag"):
                d = dead or not reached(anc_path, anc, cp)
                if d:
                    c["tag"] = f"{c['cls']}#0"
                never_run(c, cp, c, cp, d)
            else:
                never_run(c, cp, anc, anc_path, dead)
    never_run(tree, "", tree, "", False)
    used = {n.get("tag") for _, n in walk(tree)}
    variants = {t: v for t, v in variants.items() if t in used and v["nodes"]}
    text = to_text({"tree": tree, "variants": variants, "problems": []})
    return text, calls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree-model", required=True, help="adapter directory for the instance calls")
    ap.add_argument("--wire-model", required=True, help="adapter directory for the wiring calls")
    ap.add_argument("--split", default="hier_dev")
    ap.add_argument("--ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--inst-tokens", type=int, default=2048)
    ap.add_argument("--wire-tokens", type=int, default=2048)
    ap.add_argument("--max-calls", type=int, default=300, help="instance calls per program at most")
    ap.add_argument("--part", help="suffix of the result file, so two processes can split one run")
    ap.add_argument("--frozen", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.split in FROZEN and not args.frozen:
        raise SystemExit(f"{args.split} is frozen (data/generated/FROZEN_v2.json)")
    rows = [json.loads(l) for l in (DATA / f"{args.split}.jsonl").open(encoding="utf-8")]
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in want]
    rows = rows[:args.limit] if args.limit else rows
    print("  local model on this machine: free (no provider is contacted)", flush=True)

    (isys, itpl), (wsys, wtpl) = load_prompt(INST_PROMPT), load_prompt(WIRE_PROMPT)
    try:
        tree_be = make_backend("hf", args.tree_model, max_tokens=args.inst_tokens)
        wire_be = tree_be if args.wire_model == args.tree_model else make_backend("hf", args.wire_model,
                                                                                    max_tokens=args.wire_tokens)
    except BackendError as exc:
        raise SystemExit(f"\n  {exc}\n")
    run_id = f"hf__{tree_be.model}__hier_inst__{args.split}"
    cache = RESULTS / "raw" / run_id
    cache.mkdir(parents=True, exist_ok=True)
    out = RESULTS / (f"{run_id}__part{args.part}.jsonl" if args.part else f"{run_id}.jsonl")
    records = []

    def cached(name, fn):
        p = cache / name
        if p.exists():
            return p.read_text(encoding="utf-8")
        reply = fn()
        p.write_text(reply, encoding="utf-8")
        return reply

    import hashlib
    h = lambda s: hashlib.sha1(s.encode()).hexdigest()[:12]
    for i, row in enumerate(rows, 1):
        started, error, reply, counts = time.time(), None, "", {}
        reduced = reduce_source(row["code"])
        rid = row["id"]

        def ask_inst(path, head, code):
            return cached(f"{rid}.inst.{h(head)}.txt",
                          lambda: tree_be(isys, _fill(itpl, code=code.strip(), instance=head)))

        def ask_wire(path, cls, code, subtree, tag):
            return cached(f"{rid}.wire.{h(cls + subtree)}.txt",
                          lambda: wire_be(wsys, _fill(wtpl, cls=cls, code=code.strip(), subtree=subtree, tag=tag)))
        try:
            reply, counts = build_answer(reduced, ask_inst, ask_wire, args.max_calls)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - started
        print(f"  [{i}/{len(rows)}] {rid} {elapsed:6.1f}s {counts} {'FAILED: ' + error[:160] if error else ''}",
              flush=True)
        records.append({"id": rid, "kind": row.get("kind"), "family": row.get("family"),
                        "source": row.get("source", row.get("set")), "reply": reply,
                        "elapsed_s": round(elapsed, 2), "error": error, "calls": counts})
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        meta = {"backend": "hf", "model": tree_be.model, "wire_model": wire_be.model, "prompt": "hier_inst",
                "split": args.split, "n": len(records), "of": len(rows),
                "failed": sum(1 for r in records if r["error"]), "inst_tokens": args.inst_tokens,
                "wire_tokens": args.wire_tokens}
        out.with_name(out.stem + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n{len(records)} programs -> {out}")
    print(f"score with:  python -m pipeline.score_hier {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

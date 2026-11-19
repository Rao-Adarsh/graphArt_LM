"""Send a prompt to a model for every sample in a split, and save the replies.

    python -m pipeline.run_prompting --prompt zero_shot --backend mock --dry-run
    python -m pipeline.run_prompting --prompt zero_shot --backend anthropic
    python -m pipeline.run_prompting --prompt cot --backend ollama --limit 20

This does no scoring -- it only collects raw replies, so a bad scorer never
costs a second run. Every reply is cached on disk under results/raw/, keyed by
backend, model, prompt and sample id, and a re-run skips what is already there.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import cost as cost_module
from pipeline.backends import BackendError, DEFAULT_MODELS, make_backend

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
RESULTS = ROOT / "results"
SPLIT_SEPARATOR = "---USER---"


def load_prompt(name: str) -> tuple[str, str]:
    """Return (system, user_template). The system half is the cacheable prefix."""
    path = Path(name) if Path(name).exists() else PROMPTS / f"{name}.txt"
    if not path.exists():
        raise SystemExit(f"no such prompt: {path}")
    text = path.read_text(encoding="utf-8")
    if SPLIT_SEPARATOR in text:
        system, user = text.split(SPLIT_SEPARATOR, 1)
        return system.strip(), user.strip()
    return "", text.strip()


def _render(template: str, row: dict) -> str:
    """Fill a prompt template for one sample.

    `{nodes}` is for the edges-only ablation, which hands the model the correct
    operations and asks only how they connect.
    """
    text = template.replace("{code}", row["code"].strip())
    if "{nodes}" in text:
        # `given_nodes` (--nodes-from): another model's node list, or the reference's
        # reordered; otherwise the reference as traced.
        given = row.get("given_nodes")
        text = text.replace("{nodes}", node_block(given) if given else row["target"].split("EDGES")[0].strip())
    # The two-step prompts (graphart/two_step.py). Not stripped: the line numbers must
    # be those of the source exactly as it was traced. `step1` is the reference step-1
    # answer when training and the model's own step-1 reply when evaluating.
    if "{numbered_code}" in text or "{step1_nodes}" in text:
        from graphart import two_step
        text = text.replace("{numbered_code}", two_step.numbered(row["code"]))
        if "{step1_nodes}" in text:
            text = text.replace("{step1_nodes}", two_step.step2_input(row["step1"]))
    return text


def node_block(graph: dict) -> str:
    """A graph's INPUTS/NODES lines, as `{nodes}` shows them (the reference's own text
    when the graph is the reference)."""
    from graphart.schema import to_text
    return to_text({**graph, "edges": []}).split("EDGES")[0].strip()


def shuffled(graph: dict, key: str) -> dict:
    """The same graph with its nodes in a random valid topological order, renumbered:
    inputs first and the output last as always, everything else in whatever order the
    dependencies allow -- so parallel branches interleave instead of following each
    other. Deterministic per `key` (seed and sample id). Used to build node lists for
    --nodes-from (scripts/make_kaggle_notebook.py), so the reference edges never leave
    the laptop."""
    import random
    rng = random.Random(key)
    nodes = {n["id"]: n for n in graph["nodes"]}
    preds = {i: set() for i in nodes}
    for e in graph["edges"]:
        preds[e["dst"]].add(e["src"])
    first = [i for i in nodes if nodes[i]["op"] == "placeholder"]
    last = [i for i in nodes if nodes[i]["op"] == "output"]
    order, placed = list(first), set(first)
    rest = [i for i in nodes if i not in placed and i not in last]
    while rest:
        ready = [i for i in rest if preds[i] <= placed]
        pick = rng.choice(ready)
        order.append(pick); placed.add(pick); rest.remove(pick)
    order += last
    new = {old: k for k, old in enumerate(order)}
    return {**graph,
            "nodes": [{**nodes[old], "id": new[old]} for old in order],
            "edges": sorted(({**e, "src": new[e["src"]], "dst": new[e["dst"]]} for e in graph["edges"]),
                            key=lambda e: (e["dst"], e["src"]))}


def splice(reply: str, given: dict) -> str:
    """The given nodes followed by the edge list of an edges-only reply: a whole graph
    the scorer reads as is. Same rule as the scorer's edges mode (pipeline/score.py)."""
    from graphart.schema import to_text
    edges = reply.split("EDGES", 1)[1] if "EDGES" in reply else reply
    return to_text({**given, "edges": []}) + "\n" + edges.strip()


def load_split(split: str, tiers: list[int] | None, limit: int | None,
               stratify: int | None) -> list[dict]:
    path = ROOT / "data" / "generated" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/build_dataset.py first")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if tiers:
        rows = [r for r in rows if r["tier"] in tiers]
    if stratify:
        # The split was shuffled at build time, so the first N of a tier is an
        # arbitrary but reproducible sample -- and the same one for every model,
        # which is what makes the runs comparable.
        picked, seen = [], {}
        for row in rows:
            if seen.get(row["tier"], 0) < stratify:
                seen[row["tier"]] = seen.get(row["tier"], 0) + 1
                picked.append(row)
        rows = picked
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", default="zero_shot",
                    help="a name in prompts/ (zero_shot, few_shot, cot, gates_only) or a path")
    ap.add_argument("--backend", default="mock",
                    choices=["anthropic", "openrouter", "ollama", "hf", "mock"])
    ap.add_argument("--model", default=None, help="override the backend's default model")
    ap.add_argument("--effort", default="high",
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="hosted models only; part of the experimental condition")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override the backend's own default output cap")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="Ollama only: context window; prompt + reply must fit in it")
    ap.add_argument("--split", default="test",
                    help="train, val, test, or any evaluation set in data/generated/<name>.jsonl "
                         "(torchvision, edits, ...)")
    ap.add_argument("--tier", type=int, action="append", help="restrict to a tier (repeatable)")
    ap.add_argument("--stratify", type=int, metavar="N",
                    help="N samples per tier -- the same N for every model, so runs compare")
    ap.add_argument("--limit", type=int, help="only the first N samples")
    ap.add_argument("--refresh", action="store_true", help="ignore cached replies and re-ask")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first fully-rendered prompt and exit without calling anything")
    ap.add_argument("--yes", action="store_true",
                    help="required to proceed when the run would cost money")
    ap.add_argument("--cached-only", action="store_true",
                    help="rebuild results from replies already on disk; never contacts a "
                         "provider, so it cannot cost anything")
    ap.add_argument("--nodes-from", metavar="RESULTS_JSONL",
                    help="edges-only prompts: hand the model the nodes of this run's replies (id, reply) "
                         "instead of the reference ones -- another model's, or the reference reordered. "
                         "Samples it has no reply for are left out; each result is those nodes plus the "
                         "model's edges, scored as a whole graph")
    args = ap.parse_args()

    system, user_template = load_prompt(args.prompt)
    rows = load_split(args.split, args.tier, args.limit, args.stratify)
    prompt_name = Path(args.prompt).stem
    if args.nodes_from:
        if "{nodes}" not in user_template:
            raise SystemExit(f"{prompt_name} takes no node list ({{nodes}}); --nodes-from is for edges-only prompts")
        from graphart.two_step import parse_step1      # bundled on Kaggle; reads the nodes of a whole reply too
        source = Path(args.nodes_from)
        replies = {r["id"]: r["reply"] for r in map(json.loads, source.open(encoding="utf-8"))}
        left_out = [r["id"] for r in rows if r["id"] not in replies]
        rows = [r for r in rows if r["id"] in replies]
        if left_out:
            print(f"  --nodes-from: {len(left_out)} samples have no reply there and are left out", flush=True)
        for row in rows:
            try:
                graph = parse_step1(replies[row["id"]])[0]
                row["given_nodes"] = {**graph, "edges": []} if graph["nodes"] else None
            except Exception:
                row["given_nodes"] = None          # that reply has no readable node list
            row["given_nodes_missing"] = row["given_nodes"] is None
        # e.g. hf__qwen2.5-coder-1.5b-graphart-lora-mixed-v2__zero_shot__torchvision -> mixed-v2,
        #      reference__shuffled1__torchvision -> shuffled1
        prompt_name += "_nodes_from_" + source.stem.split("__")[1].split("lora-")[-1]

    model_name = args.model or DEFAULT_MODELS[args.backend]
    estimate = cost_module.estimate(args.backend, model_name, rows, system, user_template)

    if args.dry_run:
        print(f"=== SYSTEM ({len(system)} chars) ===\n{system}")
        print(f"\n=== USER (sample {rows[0]['id']}) ===")
        print(_render(user_template, rows[0]))
        print(f"\n=== {len(rows)} samples would be sent to {args.backend}/{model_name} ===")
        print(cost_module.render(estimate, args.backend, model_name))
        print("\n  --dry-run: nothing was sent, nothing was charged.")
        return

    # Spending gate: never make a paid call without showing the price first and
    # being told explicitly to go ahead.
    if args.cached_only:
        print("  --cached-only: no provider is contacted, so this cannot cost anything.")
    else:
        print(cost_module.render(estimate, args.backend, model_name))
    if not args.cached_only and not cost_module.is_free(estimate) and not args.yes:
        raise SystemExit(
            "\n  STOPPED: this run would cost money (or its price is unknown).\n"
            "  Nothing has been sent and nothing has been charged.\n"
            "  Re-run with --yes only if you accept the cost above.\n")

    try:
        extra = {} if args.max_tokens is None else {"max_tokens": args.max_tokens}
        if args.num_ctx is not None:
            extra["num_ctx"] = args.num_ctx
        backend = None if args.cached_only else make_backend(
            args.backend, args.model, effort=args.effort, **extra)
    except BackendError as exc:
        raise SystemExit(f"\n  {exc}\n")
    slug = (backend.model if backend else model_name).replace(":", "-").replace("/", "-")
    # The test split keeps the original run names; any other set gets its own suffix,
    # so its replies and results never mix with the test split's.
    run_id = f"{args.backend}__{slug}__{prompt_name}" + ("" if args.split == "test" else f"__{args.split}")
    cache_dir = RESULTS / "raw" / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    records, cached, failed = [], 0, 0
    for i, row in enumerate(rows, 1):
        cache_file = cache_dir / f"{row['id']}.txt"
        if row.get("given_nodes_missing"):
            # Nothing to hand the model: the source reply had no readable nodes. Kept, so
            # the sample counts as a failure rather than vanishing from the run.
            records.append({"id": row["id"], "tier": row["tier"], "num_nodes": row["num_nodes"], "reply": "",
                            "elapsed_s": 0.0, "error": "no readable node list in --nodes-from", "given_nodes": True})
            continue
        if cache_file.exists() and not args.refresh:
            reply, elapsed, error = cache_file.read_text(encoding="utf-8"), 0.0, None
            cached += 1
        elif args.cached_only:
            continue                      # nothing on disk, and we may not ask for it
        else:
            started = time.time()
            try:
                reply = backend(system, _render(user_template, row))
                error = None
                cache_file.write_text(reply, encoding="utf-8")
            except BackendError as exc:      # configuration problems: stop, do not burn the run
                raise SystemExit(f"\n{exc}")
            except Exception as exc:         # one bad sample must not lose the other 159
                reply, error = "", f"{type(exc).__name__}: {exc}"
                failed += 1
            elapsed = time.time() - started
            print(f"  [{i}/{len(rows)}] {row['id']} tier {row['tier']} "
                  f"{elapsed:5.1f}s {'FAILED: ' + error if error else ''}", flush=True)

        record = {"id": row["id"], "tier": row["tier"], "num_nodes": row["num_nodes"],
                  "reply": reply, "elapsed_s": round(elapsed, 2), "error": error}
        if row.get("given_nodes"):
            # The nodes came from elsewhere, so `reply` is made a whole graph -- those
            # nodes plus the model's edges -- and the scorer reads it in full mode.
            record.update(reply=splice(reply, row["given_nodes"]) if reply else "",
                          raw_reply=reply, given_nodes=True)
        records.append(record)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{run_id}.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    meta = {"backend": args.backend, "model": (backend.model if backend else model_name), "prompt": prompt_name,
            "split": args.split, "effort": args.effort, "n": len(records),
            "cached": cached, "failed": failed,
            **({"nodes_from": args.nodes_from} if args.nodes_from else {})}
    (RESULTS / f"{run_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{len(records)} replies ({cached} from cache, {failed} failed) -> {out}")
    print(f"score with:  python -m pipeline.score {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

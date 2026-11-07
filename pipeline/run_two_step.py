"""Ask a model for the graph in two steps (graphart/two_step.py), and save the replies.

    python -m pipeline.run_two_step --backend hf --model lora/<adapter> --split torchvision
    python -m pipeline.run_two_step --backend ollama --model qwen2.5-coder:14b --stratify 10

Per program: the `two_step_nodes` prompt, then the `two_step_edges` prompt given that
reply's own nodes (with --brief, their rule-free versions that a --two-step adapter is
trained on) -- never the reference ones. The two replies are joined into the
ordinary INPUTS/NODES/EDGES answer and written to results/<run>.jsonl exactly as
run_prompting.py writes its runs, so `python -m pipeline.score` and the error
classifier read it unchanged. A pair that cannot be joined (either reply unreadable)
is saved as the two raw replies one after the other, which the scorer counts as
unparseable, as it would a one-step reply.

Both raw replies are cached under results/raw/<run>/<id>.step1.txt and .step2.txt.

`--step1-from results/<run>.jsonl` skips step 1 and hands step 2 the nodes of another
run's replies instead -- e.g. a one-step model's -- to measure step 2 on those nodes:

    python -m pipeline.run_two_step --backend hf --model lora/<adapter> --brief --split torchvision         --step1-from results/hf__qwen2.5-coder-1.5b-graphart-lora-mixed-v2__zero_shot__torchvision.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphart.schema import ParseError
from graphart.two_step import assemble
from pipeline import cost as cost_module
from pipeline.backends import BackendError, DEFAULT_MODELS, make_backend
from pipeline.run_prompting import RESULTS, ROOT, _render, load_prompt, load_split

STEPS = ("two_step_nodes", "two_step_edges")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="mock", choices=["anthropic", "openrouter", "ollama", "hf", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-tokens", type=int, default=None, help="output cap for each of the two calls")
    ap.add_argument("--num-ctx", type=int, default=None, help="Ollama only: context window")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tier", type=int, action="append")
    ap.add_argument("--stratify", type=int, metavar="N")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--yes", action="store_true", help="required to proceed when the run would cost money")
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--brief", action="store_true",
                    help="the brief prompts, without the rules: what a --two-step adapter was trained on")
    ap.add_argument("--step1-from", metavar="RESULTS_JSONL",
                    help="take step 1 from this run's replies (their NODES section); only step 2 is asked")
    args = ap.parse_args()
    suffix = "_brief" if args.brief else ""
    prompt_name = "two_step" + suffix
    given = None
    if args.step1_from:
        source = Path(args.step1_from)
        given = {r["id"]: r["reply"] for r in map(json.loads, source.open(encoding="utf-8"))}
        # e.g. hf__qwen2.5-coder-1.5b-graphart-lora-mixed-v2__zero_shot__torchvision -> mixed-v2
        prompt_name += "_nodes_from_" + source.stem.split("__")[1].split("lora-")[-1]

    if args.split == "torchvision_holdout":
        print("  note: torchvision_holdout is frozen -- score it once, on the final model only.")
    (sys1, tpl1), (sys2, tpl2) = (load_prompt(name + suffix) for name in STEPS)
    rows = load_split(args.split, args.tier, args.limit, args.stratify)
    model_name = args.model or DEFAULT_MODELS[args.backend]
    # Both calls, priced together; the step-2 prompt is estimated without step 1's
    # nodes, so this undercounts its input a little.
    estimate = cost_module.estimate(args.backend, model_name, rows, sys1 + sys2, tpl1 + tpl2)
    if args.cached_only:
        print("  --cached-only: no provider is contacted, so this cannot cost anything.")
    else:
        print(cost_module.render(estimate, args.backend, model_name))
    if not args.cached_only and not cost_module.is_free(estimate) and not args.yes:
        raise SystemExit("\n  STOPPED: this run would cost money (or its price is unknown).\n"
                         "  Nothing has been sent and nothing has been charged.\n"
                         "  Re-run with --yes only if you accept the cost above.\n")

    extra = {} if args.max_tokens is None else {"max_tokens": args.max_tokens}
    if args.num_ctx is not None:
        extra["num_ctx"] = args.num_ctx
    try:
        backend = None if args.cached_only else make_backend(args.backend, args.model, effort=args.effort, **extra)
    except BackendError as exc:
        raise SystemExit(f"\n  {exc}\n")
    slug = (backend.model if backend else model_name).replace(":", "-").replace("/", "-")
    run_id = f"{args.backend}__{slug}__{prompt_name}" + ("" if args.split == "test" else f"__{args.split}")
    cache_dir = RESULTS / "raw" / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    def ask(row: dict, step: int, system: str, template: str) -> tuple[str, bool]:
        """One call's reply, from the cache when it is there; (reply, was_cached)."""
        path = cache_dir / f"{row['id']}.step{step}.txt"
        if path.exists() and not args.refresh:
            return path.read_text(encoding="utf-8"), True
        if args.cached_only:
            raise LookupError
        reply = backend(system, _render(template, row))
        path.write_text(reply, encoding="utf-8")
        return reply, False

    records, cached, failed, joined = [], 0, 0, 0
    for i, row in enumerate(rows, 1):
        started, error = time.time(), None
        try:
            if given is None:
                step1, hit1 = ask(row, 1, sys1, tpl1)
            elif row["id"] in given:
                step1, hit1 = given[row["id"]], True
            else:
                raise KeyError(f"{row['id']} is not in {args.step1_from}")
            step2, hit2 = ask({**row, "step1": step1}, 2, sys2, tpl2)
            cached += hit1 and hit2
            try:
                reply = assemble(step1, step2)
                joined += 1
            except ParseError:
                reply = step1.strip() + "\n" + step2.strip()
        except LookupError:
            continue                          # --cached-only and not on disk
        except BackendError as exc:
            raise SystemExit(f"\n{exc}")
        except Exception as exc:              # one bad sample must not lose the rest
            reply, error = "", f"{type(exc).__name__}: {exc}"
            failed += 1
        elapsed = time.time() - started
        if not args.cached_only:
            print(f"  [{i}/{len(rows)}] {row['id']} tier {row['tier']} {elapsed:5.1f}s "
                  f"{'FAILED: ' + error if error else ''}", flush=True)
        records.append({"id": row["id"], "tier": row["tier"], "num_nodes": row["num_nodes"],
                        "reply": reply, "elapsed_s": round(elapsed, 2), "error": error})

    out = RESULTS / f"{run_id}.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    meta = {"backend": args.backend, "model": (backend.model if backend else model_name), "prompt": prompt_name,
            "split": args.split, "effort": args.effort, "n": len(records),
            "cached": cached, "failed": failed, "joined": joined}
    (RESULTS / f"{run_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n{len(records)} programs ({cached} from cache, {failed} failed, {joined} joined) -> {out}")
    print(f"score with:  python -m pipeline.score {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

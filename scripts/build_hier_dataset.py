"""Build the training data for the hierarchical schema: generator v3 + real timm code + tiers 1-4.

    .venv/Scripts/python scripts/build_hier_dataset.py                 # ~1,500 v3 programs
    .venv/Scripts/python scripts/build_hier_dataset.py --v3 200 --timm-files 10   # a quick try

Writes data/generated/hier_{train,val,gen_test}.jsonl. A row is one program:

    id, source ("v3" | "timm" | "tiers14"), family, kind ("network" | "block"), code,
    target (the answer text, graphart/hier.py), hier (the full truth), flat_nodes,
    tree_task_tokens (reduced code + tree, Qwen tokens), split

Sources (docs/generator_rework_plan.md):
  v3       graphart/generate_v3.py, 70% whole networks, families in rotation
  timm     real timm models, up to --timm-sizes per allowed file, code assembled exactly as
           the dev/test sets' (build_hier_eval.assemble); never a file FROZEN_v2.json excludes
  tiers14  the small single-class programs of generator v1 (data/generated/train.jsonl),
           ~15% of the examples, so the small programs are not forgotten (finding 8)

Every program passes the lossless check (graphart/hier.check_lossless). Duplicates go
(isomorphic flat graphs, labels with parameters), and so does any program whose flat graph
is topology-isomorphic to a dev, holdout or timm-test program -- the frozen sets are read
only for that comparison, never printed. Examples longer than --max-tokens (the tree
task: reduced code + tree) are left out; the memory probe fits 20k on a T4.
"""

import argparse
import collections
import json
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import torch

from build_large_dataset import Seen
from graphart.hier import check_lossless, hier_from_code, hier_from_module, inline, to_text, tree_text
from graphart.reduce import reduce_source
from graphart.validate import node_label, topology_label

DATA = ROOT / "data" / "generated"
OVERHEAD = 300                      # prompt instruction tokens per example (assumed)


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", local_files_only=True)
    return lambda s: len(tok(s)["input_ids"])


class Builder:
    def __init__(self, max_tokens: int):
        self.T = tokenizer()
        self.max_tokens = max_tokens
        self.seen = Seen(node_label)
        self.frozen = Seen(topology_label)
        self.report = collections.Counter()
        for name in ("dev", "holdout", "timm_test"):
            for line in (DATA / f"hier_{name}.jsonl").open(encoding="utf-8"):
                self.frozen.add(inline(json.loads(line)["hier"]))

    def add(self, rows: list, rid: str, source: str, family: str, kind: str, code: str, h: dict, flat: dict) -> bool:
        try:
            return self._add(rows, rid, source, family, kind, code, h, flat)
        except Exception as e:                      # one odd program must not stop the build
            self.report[f"{source}: error {type(e).__name__}: {str(e)[:60]}"] += 1
            return False

    def _add(self, rows, rid, source, family, kind, code, h, flat) -> bool:
        if check_lossless(h, flat):
            self.report[f"{source}: not lossless"] += 1
            return False
        if self.frozen.find(flat):
            self.report[f"{source}: same graph as an eval-set program"] += 1
            return False
        if self.seen.find(flat):
            self.report[f"{source}: duplicate"] += 1
            return False
        tokens = OVERHEAD + self.T(reduce_source(code)) + self.T(tree_text(h))
        if tokens > self.max_tokens:
            self.report[f"{source}: over {self.max_tokens} tokens"] += 1
            return False
        self.seen.add(flat)
        rows.append({"id": rid, "source": source, "family": family, "kind": kind, "code": code,
                     "target": to_text(h), "hier": h, "flat_nodes": len(flat["nodes"]),
                     "tree_task_tokens": tokens})
        return True


def v3_rows(b: Builder, n: int, seed: int, network_share: float) -> list:
    from graphart.generate_v3 import SCHEDULE as KINDS, generate_v3_program
    rng, rows, attempts = random.Random(seed), [], 0
    while len(rows) < n and attempts < n * 3:
        fam = KINDS[attempts % len(KINDS)]
        attempts += 1
        kind = "network" if rng.random() < network_share else "block"
        code = generate_v3_program(rng, kind, fam)
        try:
            h, flat = hier_from_code(code)
        except Exception:
            b.report["v3: failed to trace"] += 1
            continue
        b.add(rows, f"v3-{len(rows):05d}", "v3", fam, kind, code, h, flat)
        if len(rows) % 250 == 0 and rows:
            print(f"  v3: {len(rows)} / {n}", flush=True)
    return rows


def timm_rows(b: Builder, files_limit: int, sizes: int) -> list:
    import timm
    from timm.models import model_entrypoint
    from build_hier_eval import assemble
    frozen = json.loads((DATA / "FROZEN_v2.json").read_text(encoding="utf-8"))
    exclude = set(frozen["training_exclusions"]["timm_files"])
    by_file = collections.defaultdict(list)
    for m in timm.list_models():
        by_file[model_entrypoint(m).__module__.split(".")[-1]].append(m)
    files = [f for f in sorted(by_file) if f not in exclude][: files_limit or None]
    import re
    # Building a model allocates its weights: a billion-parameter one (eva_giant, ...) takes
    # gigabytes and killed the first build with no traceback. Small sizes have the same code.
    # Extreme depths too (densenet264d crashed the process natively): a shallower size has
    # the same classes and the same wiring.
    big = re.compile(r"giant|gigantic|enormous|huge|xxlarge|xlarge|large|_g_|_h_|bigG|_e_|_448|_512|_1024"
                     r"|264|201|161|152|200|269|270|350|420")
    small = re.compile(r"atto|femto|pico|nano|tiny|mini|small|_xs|_s\b|_t\b|18|26|_050|_075|_100\b|b0")
    jobs = []
    for f in files:
        names = [n for n in by_file[f] if not big.search(n)] or sorted(by_file[f], key=len)[:1]
        names = sorted(names, key=lambda s: (not small.search(s), len(s), s))
        jobs += [(f, name) for name in names[:sizes]]

    # Each model is built and traced in a worker process: a model that crashes the process
    # natively (no Python traceback) or hangs costs that model, not the build.
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
    from concurrent.futures.process import BrokenProcessPool
    rows, pool = [], None
    for f, name in jobs:
        if pool is None:
            pool = ProcessPoolExecutor(max_workers=1, max_tasks_per_child=8)
        print(f"    timm {name}", flush=True)
        try:
            code, h, flat = pool.submit(_timm_one, name).result(timeout=600)
        except (BrokenProcessPool, FutureTimeout) as e:
            b.report[f"timm: worker {'crashed' if isinstance(e, BrokenProcessPool) else 'timed out'}"] += 1
            for p in list(getattr(pool, "_processes", {}).values()):
                p.terminate()
            pool.shutdown(wait=False, cancel_futures=True)
            pool = None
            continue
        except Exception:
            b.report["timm: failed to trace"] += 1
            continue
        b.add(rows, f"timm-{name}", "timm", f, "network", code, h, flat)
    if pool is not None:
        pool.shutdown()
    print(f"  timm: {len(rows)} programs from {len(files)} files", flush=True)
    return rows


def _timm_one(name: str):
    """Build one timm model, assemble its code, extract the truth (runs in a worker)."""
    import warnings
    warnings.filterwarnings("ignore")
    from timm.models import model_entrypoint
    from build_hier_eval import assemble
    torch.manual_seed(0)
    builder = model_entrypoint(name)
    model = builder()
    code, _ = assemble(model, builder, "", ("timm.models",))
    h, flat = hier_from_module(model, lambda c: c.__module__.startswith("timm.models"), None)
    return code, h, flat


def tiers14_rows(b: Builder, n: int, seed: int) -> list:
    src = [json.loads(l) for l in (DATA / "train.jsonl").open(encoding="utf-8")]
    random.Random(seed).shuffle(src)
    rows = []
    for r in src:
        if len(rows) >= n:
            break
        try:
            h, flat = hier_from_code(r["code"])
        except Exception:
            b.report["tiers14: failed to trace"] += 1
            continue
        b.add(rows, f"t14-{r['id']}", "tiers14", f"tier{r['tier']}", "block", r["code"], h, flat)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v3", type=int, default=1500)
    ap.add_argument("--network-share", type=float, default=0.7)
    ap.add_argument("--timm-files", type=int, default=0, help="limit timm files (0 = all allowed)")
    ap.add_argument("--timm-sizes", type=int, default=3)
    ap.add_argument("--tiers14-share", type=float, default=0.15)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prefix", default="hier_")
    ap.add_argument("--resume", action="store_true", help="reuse sources already built (results/raw/hier_build_cache)")
    args = ap.parse_args()

    t0 = time.time()
    b = Builder(args.max_tokens)
    cache = ROOT / "results" / "raw" / "hier_build_cache"
    cache.mkdir(parents=True, exist_ok=True)

    def cached(name, make):
        """Rows of one source, saved as soon as they exist; reused on a rerun (--resume)."""
        path = cache / f"{args.prefix}{name}.jsonl"
        if args.resume and path.exists():
            rows = [json.loads(l) for l in path.open(encoding="utf-8")]
            for r in rows:                         # keep duplicate checks across sources
                b.seen.add(inline(r["hier"]))
            print(f"  {name}: {len(rows)} rows reused from {path.relative_to(ROOT)}", flush=True)
            return rows
        rows = make()
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return rows

    v3 = cached("v3", lambda: v3_rows(b, args.v3, args.seed, args.network_share))
    tm = cached("timm", lambda: timm_rows(b, args.timm_files, args.timm_sizes))
    n14 = round(args.tiers14_share / (1 - args.tiers14_share) * (len(v3) + len(tm)))
    t14 = cached("tiers14", lambda: tiers14_rows(b, n14, args.seed))

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "gen_test": []}
    rng.shuffle(v3)
    a, c = int(0.9 * len(v3)), int(0.95 * len(v3))
    splits["train"] += v3[:a]
    splits["val"] += v3[a:c]
    splits["gen_test"] += v3[c:]
    files = sorted({r["family"] for r in tm})
    val_files = set(rng.sample(files, max(1, len(files) // 20))) if files else set()
    splits["train"] += [r for r in tm if r["family"] not in val_files]
    splits["val"] += [r for r in tm if r["family"] in val_files]
    splits["train"] += t14[: int(0.95 * len(t14))]
    splits["val"] += t14[int(0.95 * len(t14)):]
    for name, rows in splits.items():
        rng.shuffle(rows)
        for r in rows:
            r["split"] = name
        path = DATA / f"{args.prefix}{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        by = collections.Counter(r["source"] for r in rows)
        tok = [r["tree_task_tokens"] for r in rows]
        print(f"{name:8s} {len(rows):5d} programs {dict(by)}; tree-task tokens median "
              f"{statistics.median(tok) if tok else 0:.0f}, max {max(tok, default=0)}, total "
              f"{sum(tok) / 1e6:.2f}M -> {path.relative_to(ROOT)}")
    print("left out:", dict(b.report))
    print(f"done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()

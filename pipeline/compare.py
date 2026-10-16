"""Put several runs side by side -- the headline table for the report.

    python -m pipeline.compare results/*.jsonl
    python -m pipeline.compare results/*__zero_shot.jsonl --markdown

Each row is one (model, prompt) run; the tier columns are `exact`, which is the
number that matters. Runs scored here must share a split to be comparable, and
the runner's --stratify picks the same samples for every model so they do.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.score import score_reply, summarise

ROOT = Path(__file__).resolve().parents[1]


def load_truths(split: str) -> dict:
    path = ROOT / "data" / "generated" / f"{split}.jsonl"
    return {r["id"]: r["graph"] for r in
            (json.loads(line) for line in path.open(encoding="utf-8"))}


def score_run(path: Path, truths: dict) -> dict | None:
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    connect = "gates_only" in path.stem
    scored = [{**r, **score_reply(r["reply"], truths[r["id"]], connect=connect)}
              for r in records if r["id"] in truths]
    if not scored:
        return None

    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    parts = path.stem.split("__")
    overall = summarise(scored)
    tiers = {t: summarise([s for s in scored if s["tier"] == t])["exact"]
             for t in sorted({s["tier"] for s in scored})}
    return {"model": meta.get("model") or (parts[1] if len(parts) > 1 else path.stem),
            "prompt": meta.get("prompt") or (parts[2] if len(parts) > 2 else "?"),
            "overall": overall, "tiers": tiers, "file": path.name}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="+", help="results/*.jsonl files to compare")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--markdown", action="store_true", help="emit a markdown table")
    ap.add_argument("--sort", choices=["exact", "model"], default="exact")
    args = ap.parse_args()

    paths = [Path(p) for p in args.results if not p.endswith((".meta.json", ".summary.json"))]
    truths = load_truths(args.split)
    runs = [r for r in (score_run(p, truths) for p in paths) if r]
    if not runs:
        raise SystemExit("no scorable runs found")

    all_tiers = sorted({t for r in runs for t in r["tiers"]})
    runs.sort(key=(lambda r: -r["overall"]["exact"]) if args.sort == "exact"
              else (lambda r: r["model"]))

    if args.markdown:
        head = ["model", "prompt", "n", "parsed", "valid", "**exact**"] + \
               [f"t{t} exact" for t in all_tiers]
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join(["---"] * len(head)) + "|")
        for r in runs:
            o = r["overall"]
            cells = [f"`{r['model']}`", r["prompt"], str(o["n"]), f"{o['parsed']:.2f}",
                     f"{o['valid']:.2f}", f"**{o['exact']:.2f}**"] + \
                    [f"{r['tiers'].get(t, float('nan')):.2f}" for t in all_tiers]
            print("| " + " | ".join(cells) + " |")
    else:
        header = (f"{'model':<40}{'prompt':<12}{'n':>5}{'parsed':>8}{'valid':>7}"
                  f"{'exact':>7}" + "".join(f"{'t' + str(t):>7}" for t in all_tiers))
        print(header)
        print("-" * len(header))
        for r in runs:
            o = r["overall"]
            print(f"{r['model'][:39]:<40}{r['prompt'][:11]:<12}{o['n']:>5}{o['parsed']:>8.2f}"
                  f"{o['valid']:>7.2f}{o['exact']:>7.2f}"
                  + "".join(f"{r['tiers'].get(t, float('nan')):>7.2f}" for t in all_tiers))
        print("\ntier columns are `exact`; t1 flat, t2 loops, t3 branches, t4 blocks")

    out = ROOT / "results" / "comparison.json"
    out.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Look up model ids and prices on OpenRouter. No API key needed.

    python -m pipeline.list_models qwen coder
    python -m pipeline.list_models --free
    python -m pipeline.list_models deepseek --sort price

Model slugs and prices drift, so look them up rather than trusting a hardcoded
list -- an id that is wrong by one character fails only after the run starts.
"""

import argparse
import json
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch() -> list[dict]:
    request = urllib.request.Request(MODELS_URL, headers={"User-Agent": "graphArt-LM"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())["data"]


def per_million(price: str | None) -> float:
    """Price per million tokens. Router entries use a negative sentinel for
    'varies by which model it routes to' -- report those as unknown."""
    try:
        value = float(price) * 1_000_000
    except (TypeError, ValueError):
        return float("nan")
    return float("nan") if value < 0 else value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("terms", nargs="*", help="all terms must appear in the model id")
    ap.add_argument("--free", action="store_true", help="only models that cost nothing")
    ap.add_argument("--sort", choices=["id", "price"], default="id")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    rows = []
    for model in fetch():
        model_id = model.get("id", "")
        if not all(t.lower() in model_id.lower() for t in args.terms):
            continue
        pricing = model.get("pricing") or {}
        prompt_price = per_million(pricing.get("prompt"))
        completion_price = per_million(pricing.get("completion"))
        if args.free and not (prompt_price == 0 and completion_price == 0):
            continue
        rows.append((model_id, model.get("context_length") or 0,
                     prompt_price, completion_price))

    rows.sort(key=(lambda r: (r[2], r[0])) if args.sort == "price" else (lambda r: r[0]))

    if not rows:
        print("no models matched")
        return
    print(f"{'model id':<52}{'context':>10}{'$/1M in':>10}{'$/1M out':>10}")
    print("-" * 82)
    for model_id, context, prompt_price, completion_price in rows[:args.limit]:
        print(f"{model_id:<52}{context:>10,}{prompt_price:>10.3f}{completion_price:>10.3f}")
    if len(rows) > args.limit:
        print(f"... and {len(rows) - args.limit} more (raise --limit)")


if __name__ == "__main__":
    main()

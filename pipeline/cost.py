"""Estimate what a run will cost, before it runs.

The runner refuses to make a paid call unless it has printed an estimate and
been given --yes. Free models, Ollama and the mock backend are free by
construction and still print $0.00 so the number is always on screen.
"""

# Anthropic first-party rates, $ per million tokens (input, output).
ANTHROPIC_RATES = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

CHARS_PER_TOKEN = 4          # rough, but consistent across models
OUTPUT_SLACK = 1.8           # replies run longer than the bare target text


def rates_for(backend: str, model: str) -> tuple[float, float] | None:
    """($/1M input, $/1M output), or None when we cannot find out."""
    if backend in ("mock", "ollama"):
        return (0.0, 0.0)
    if backend == "anthropic":
        return ANTHROPIC_RATES.get(model)
    if backend == "openrouter":
        try:
            from pipeline.list_models import fetch, per_million
            for entry in fetch():
                if entry.get("id") == model:
                    pricing = entry.get("pricing") or {}
                    return (per_million(pricing.get("prompt")),
                            per_million(pricing.get("completion")))
        except Exception:
            return None
    return None


def estimate(backend: str, model: str, rows: list[dict], system: str,
             user_template: str) -> dict:
    """Token and dollar estimate for sending `rows` through this prompt."""
    prefix_tokens = len(system) / CHARS_PER_TOKEN
    input_tokens = sum(
        prefix_tokens + len(user_template.replace("{code}", r["code"].strip())) / CHARS_PER_TOKEN
        for r in rows)
    output_tokens = sum(len(r["target"]) / CHARS_PER_TOKEN * OUTPUT_SLACK for r in rows)

    rates = rates_for(backend, model)
    if rates is None:
        dollars = None
    else:
        dollars = (input_tokens / 1e6) * rates[0] + (output_tokens / 1e6) * rates[1]
    return {"n": len(rows), "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens), "rates": rates, "dollars": dollars}


def render(est: dict, backend: str, model: str) -> str:
    lines = [
        "",
        "  " + "=" * 62,
        f"  COST ESTIMATE   {backend} / {model}",
        f"    {est['n']} samples   ~{est['input_tokens']:,} input tokens"
        f"   ~{est['output_tokens']:,} output tokens",
    ]
    if est["rates"] is None:
        lines.append("    rate: UNKNOWN -- could not look up this model's price")
        lines.append("    ESTIMATED COST: unknown. Treat as if it costs money.")
    else:
        rate_in, rate_out = est["rates"]
        lines.append(f"    rate: ${rate_in:.3f}/1M in, ${rate_out:.3f}/1M out")
        lines.append(f"    ESTIMATED COST: ${est['dollars']:.4f}"
                     + ("   (free model)" if est["dollars"] == 0 else ""))
    lines.append("    Estimates are approximate. The real bill is whatever the provider charges.")
    lines.append("  " + "=" * 62)
    return "\n".join(lines)


def is_free(est: dict) -> bool:
    return est["dollars"] == 0

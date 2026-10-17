# Handoff — state and agenda

*Written 17 Oct 2026, at the end of the domain rewrite. Read this first in a new session.*

## Where things stand

The PyTorch rewrite is **complete and committed**. `graphart/` has `extract`, `schema`,
`validate`, `generate`, `render`, `connect` and `metrics` on the forward-pass domain;
`pipeline/` is domain-agnostic; the dataset is built (1280 / 160 / 160).

**No model has ever been called. Nothing has been spent. Ever.**

```
bc57f39  renderer, and the reference-vs-prediction figure
8aacdb8  six prompts, and the plumbing two of them need
3305df4  build the PyTorch dataset
0fb7a77  tiered program generator
053e0fe  structural validator, and switch domain.py over
6dfa407  record every argument slot an input fills
2221468  text schema for the forward-pass graph
8eac712  fx-based forward-pass extractor
```

Everything is **local only — nothing has been pushed** to the GitHub remote.

## Environment

| | |
|---|---|
| Machine | Windows, Python 3.11, `.venv` in the repo. **No usable GPU** (Intel Iris Xe, CPU-only) |
| Local model | Ollama running, `qwen2.5-coder:1.5b` pulled. ~90 s per sample on CPU |
| OpenRouter | Key is in the **Windows user registry**, not the environment. `backends.resolve_key()` reads it without a shell restart |
| Anthropic | No key. A Claude Pro subscription does **not** include API access |
| Free tier | OpenRouter free models: 50 requests/day with no credits, 1000/day with $10 |

## Rules that still apply

- **Never spend money without showing an estimate and getting an explicit yes.** The gate
  is in code (`pipeline/cost.py`, `--yes`, `--cached-only`); keep it intact.
- **Never propose having the model call `torch.fx` itself.** That is the answer key. The
  stated objective is whether models can *reason* about topology.
- Few-shot examples come from the **train** split only.

## Agenda, in priority order

### 1. Confirm the setup, spend nothing

```bash
.venv/Scripts/python -m pipeline.run_prompting --prompt cos --backend mock --dry-run
```

Shows the rendered prompt and a `$0.0000` estimate, sends nothing. Do this for `cos` and
`edges_only` — the latter is the one that fills `{nodes}`, so it proves the substitution
works before a paid run depends on it.

### 2. The first real numbers — free

```bash
.venv/Scripts/python -m pipeline.run_prompting --prompt zero_shot --backend openrouter --model "qwen/qwen3.8-27b:free" --stratify 10
```

40 samples, under the 50/day free cap. Gives the tier-1→4 difficulty curve, which is
research question 1. Expect this to be slow (free tier queues, roughly 1–3 min a sample).

### 3. The decomposition — the highest-value experiment

`nodes_only` then `edges_only`, same model, same 40 samples. This is research question 2
and it is the experiment that cannot be got any other way:

- both halves high, full task low → the bottleneck is **doing both at once** (stamina,
  context), not either skill
- `nodes_only` low → it cannot read the code
- `edges_only` low → it cannot build the graph

Scoring modes are inferred from the filename, so nothing extra is needed.

### 4. Does simulation prompting help?

`cos` against `cot` against `zero_shot`, same model, same samples. **`cot` is the control
and is not optional** — without it, a `cos` win cannot be attributed to the simulation
structure rather than to simply thinking more.

### 5. Mid-size models on a free Colab GPU

The gap between the local 1.5B and the hosted 27B/32B is unfilled, and the laptop cannot
close it (a 7B would take ~5 hours for 40 samples on CPU). A free T4 runs a 7B–14B at
20–40 tok/s.

Design, agreed but not built: a notebook that clones the repo, installs
`transformers`/`accelerate`, and adds a **fourth backend** — a local HF-pipeline backend —
so the *same* `run_prompting.py` and `score.py` drive it. It must write into the same
`results/raw/` layout so the outputs come back and `pipeline.compare` works with no
special-casing. The principle: **inference venue changes, scoring path does not.**

### 6. Write up

`pipeline.compare results/*.jsonl --markdown` produces the headline table.
`pipeline.score ... --slides` prints rows ready to paste into `RESULTS` in
`scripts/make_slides.py`.

## Known gaps

- **`docs/llm_primer.md`, `docs/presentation_outline.md` and the 22-slide deck in
  `scripts/make_slides.py` still describe quantum circuits.** The code is current; that
  prose is not. The deck matters for the next TA submission.
- **Spatial shape arithmetic is not validated.** `Conv2d(k=7,s=2,p=3)` claiming 224→200
  would pass. Add it only if models are seen to make that error.
- **Tool-augmented prompting was considered and dropped** — tracing is the answer key.
  A no-tracers "scratchpad" variant remains interesting but needs an agentic runner.
- **Self-check / revise prompting** is unbuilt for the same reason.
- `graph_edit_distance` is exponential; a large-graph split would need the approximate
  version or a cheaper proxy.

## Things that bit us, worth not repeating

- `extract_graph_block` hardcoded the schema header, so every reasoning-prompt reply
  failed to parse and looked like a model failure. Domain-specific strings belong in
  `graphart/domain.py`.
- An empty or unparseable reply scored `valid=True`, because an empty graph passes every
  structural check vacuously. Check `parsed` before believing `valid`.
- `fx.all_input_nodes` deduplicates, so `attn(x, x, x)` looked like one input. Ground
  truth said a correct answer was wrong.
- Heredocs in this environment mangle backslashes. Use the Write/Edit tools for patches.

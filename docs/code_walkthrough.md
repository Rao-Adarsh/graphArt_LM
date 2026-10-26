# Code walkthrough — recording outline (≤ 10 minutes)

The order the course asks for: data collection → preprocessing → splits → the network →
training (loss, optimiser, learning rate, schedule) → metrics → validation. Each block lists
what to open and what to point at; say it in your own words. Times add up to about 9:30.

Before recording: face camera on, editor font at 16+, close notifications, open the files
below in tabs in this order.

---

**0:00–0:30 · The task and the repository** — `README.md` (top), the repository-layout tree.
- Code in → forward-pass graph out, no execution; `torch.fx` gives the exact answer.
- Three parts: `graphart/` (domain: data, truth, checks, metrics), `pipeline/` (models,
  scoring), `scripts/` (dataset builders, training, analyses, report), `interface/`.

**0:30–1:50 · Data collection: the generator** — `graphart/generate.py`
- Docstring: tiers are about how much the code *hides* the graph, not size.
- `class Style` (line 63): naming, imports, activation idiom chosen independently of the
  network — against "house style". (Mention honestly: it varied spelling, not code
  organisation — what the closing experiments exposed.)
- One tier function, e.g. `tier3_branches` (line 304): residuals, concat, dead branch, reuse.
- `generate_program` (line 451).

**1:50–2:40 · Ground truth and preprocessing** — `graphart/extract.py`
- `graph_from_code` → `graph_from_module` (line 186): `GranularTracer` + `symbolic_trace`,
  `ShapeProp` for shapes, `ALIASES` normalising `x + y` / `torch.add` / `x.add` to `add`.
- `_input_slots`: edges carry argument slots (order matters for `cat`, `sub`).
- `graphart/schema.py` `to_text` / `from_text` (lines 115, 191): the text format the model
  reads and writes.

**2:40–3:20 · Splits** — `scripts/build_dataset.py`
- Generate → trace → validate → dedup (line ~96: duplicates would leak across splits) →
  80/10/10 per tier (lines 110–112) → 1280 / 160 / 160.
- Evaluation sets: `build_torchvision_set.py` (torchvision's own source, verified
  isomorphic to tracing the real object), `build_edit_set.py`, `build_rq6_probe.py`.

**3:20–4:10 · The models and prompts** — `pipeline/backends.py`, `prompts/`
- One interface, five backends; `HFBackend` (line 240) loads the base model and merges the
  LoRA adapter.
- `prompts/zero_shot.txt` (the rules and label table) and `prompts/cos.txt` — our
  chain-of-simulation prompt: track which node each variable holds.
- `pipeline/run_prompting.py`: `load_prompt` / `_render` (lines 29, 41), caching, cost gate.

**4:10–6:10 · Training: LoRA fine-tuning** — `scripts/train_lora.py`
- Base: Qwen2.5-Coder-1.5B-Instruct, a decoder-only transformer.
- `encode` (line 75): one chat = zero-shot prompt + reference graph; **loss mask**: prompt
  tokens get label `IGNORE = -100` (line 39, applied line 88), so cross-entropy is computed
  on the answer tokens only — the report's Eq. (2).
- `LoraConfig` (lines 174–176): r = 16, α = 32, dropout 0.05, on q, k, v, o, gate, up, down
  projections → 18.5 M trainable of 1.56 B (1.2 %); Eq. (1), B initialised to zero.
- `TrainingArguments` (lines 180–190): **AdamW**, **learning rate 2e-4**, **cosine schedule**
  with **3 % warm-up**, effective batch 16 (1 × 16 gradient accumulation), 2 epochs,
  fp16, gradient checkpointing, max length 2560, seed 0.
- **Validation**: val loss every 40 steps, best checkpoint kept (`load_best_model_at_end`).
- `class Guard` (line 101): stops on NaN loss or no learning (exit codes 4 / 3) and on the
  time budget — why two Kaggle runs failed safely.
- `--exclude-construct dead_branch`: the RQ6 held-out adapter.

**6:10–7:40 · Evaluation metrics** — `graphart/validate.py`, `graphart/metrics.py`
- `check` (validate.py line 175): no reference needed — acyclic, one output, reachability,
  arities, module-path hierarchy, shape consistency.
- `compare` (metrics.py line 125): `exact` = labelled isomorphism (VF2); `topo` = without
  layer settings; node / edge F1; `reach_f1` = connectivity preservation (line 53);
  shape accuracy.
- `pipeline/score.py` `score_reply` (line 85): extract the graph block, parse, parameter-
  stripping retry; per-tier tables.

**7:40–8:40 · Analysis: why a miss is a miss** — `scripts/classify_errors.py`
- `classify` (line 156): fixed order of isomorphism tests → right graph / narrow error / wrong.
- `coarse_category` (line 127): re-trace the real code with the drawn blocks kept whole.
- `gave_original`: the edit-set signature. Quick run on a result file to show the table.

**8:40–9:30 · Interface** — `interface/app.py`
- `run` (line ~168): trace the reference, take the saved answer or run the model live,
  parse, `compare` + `classify`, `compare_figure`.
- `generate` (line ~101): a fresh program, checked against every training graph.
- Launch it (`.venv/Scripts/python interface/app.py`) and show one example.

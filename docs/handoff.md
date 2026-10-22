# Handoff — state and agenda

*Updated 22 Oct 2026, after the first Phase 1 results. Read this first in a new session.*

## Where things stand

The PyTorch domain, dataset (1280 / 160 / 160) and harness are done. **Phase 1 has its
first numbers**: zero-shot on the stratified 40 for two models (table in the README,
*Results so far*):

| model | exact | topo |
|---|---|---|
| Gemma 4 31B, OpenRouter free tier | 0.75 | 0.93 |
| Qwen2.5-Coder 1.5B, Ollama on the laptop | 0.00 | 0.05 |

**Spent so far: $0.** About 9 minutes of Kaggle GPU quota (one aborted run, below).

Everything is **local only — nothing has been pushed** to the GitHub remote. Ask before
pushing.

## Environment

| | |
|---|---|
| Machine | Windows, Python 3.11, `.venv` in the repo. **No usable GPU** (Intel Iris Xe, CPU-only) |
| Local model | Ollama 0.40, `qwen2.5-coder:1.5b`. ~60–90 s per sample on CPU |
| OpenRouter | Key is in the **Windows user registry** and the environment; `backends.resolve_key()` reads either. Free tier, no credits |
| Free tier | 50 requests/day across *all* free models, reset at UTC midnight. Upstream 429s do not count against it |
| Kaggle | Account is phone-verified (GPU + internet allowed). 30 GPU-hours/week on 2× T4 |
| Anthropic | No key. A Claude Pro subscription does **not** include API access |

## Rules that still apply

- **Never spend money without showing an estimate and getting an explicit yes.** The gate
  is in code (`pipeline/cost.py`, `--yes`, `--cached-only`); keep it intact. The user has
  said no to paid runs for now.
- **Be careful with Kaggle GPU hours** and keep things simple: one unattended run per
  model, no idle interactive sessions.
- **Never propose having the model call `torch.fx` itself.** That is the answer key.
- Few-shot examples come from the **train** split only.

## Agenda, in priority order

Phase 1 (prompting) is done: zero-shot for four models, all six prompts on Qwen 14B. Its
Kaggle notebook is `notebooks/kaggle_ollama.ipynb`, built by
`scripts/make_kaggle_notebook.py --models ... --prompts ...`; results come back as a zip
the user extracts, then `run_prompting --cached-only` + `pipeline.score` as usual.

### 1. Phase 2 — LoRA fine-tune Qwen2.5-Coder 1.5B (planned 22 Oct, not built)

**Question.** Can a fine-tuned 1.5B — zero-shot topo 0.05 today — match the prompted 14B
(0.60) or Gemma 31B (0.93)? This is research question 3, answered on free hardware.

**Model.** `Qwen/Qwen2.5-Coder-1.5B-Instruct` from Hugging Face (Apache-2.0) — the same
weights as Ollama's `qwen2.5-coder:1.5b`, in fp16 rather than Q4.

**Data.** The 1,280 train samples, one chat example each: system + user = the `zero_shot`
prompt exactly as the runner renders it; assistant = the reference graph text. Loss on the
assistant tokens only. Same prompt before and after, so the comparison is clean. The val
split (160) picks the checkpoint; **the test split is never seen in training or
selection.** ~1.6M tokens per epoch, ~85% of them prompt (masked out of the loss).

**Training.** LoRA r=16, alpha=32, dropout 0.05, on all attention and MLP projections;
lr 2e-4 cosine, 3% warmup, 2 epochs, effective batch 16, max length 2048, fp16 with
gradient checkpointing. One Kaggle T4. Rough estimate 1–1.5 h training; check it in the
smoke step.

**Code to build** (all tested locally on a tiny model before any GPU time):
1. An **`hf` backend** in `pipeline/backends.py`: base model + optional LoRA adapter via
   `transformers`/`peft`, the model's own chat template, greedy decoding, the same
   1500-token cap as Ollama. `run_prompting` then drives it unchanged.
2. **`scripts/train_lora.py`**: builds the chat examples from the train split, masks the
   prompt, trains, keeps the best checkpoint by val loss.
3. A **fine-tuning mode in the notebook generator**: bundles train/val (full targets) and
   the test 40 (edges stripped); the notebook pip-installs pinned `peft`/`trl`, runs a
   ~20-step smoke train that **stops if the loss is NaN or not falling**, trains, then
   runs `zero_shot` on the test 40 for both the **untuned base** (the "before", in the
   same venue) and the **tuned adapter**, and zips the adapter and `results/`.

**Scoring.** Unchanged: replies land as `results/raw/hf__<model>__zero_shot/`, then
`--cached-only` + `pipeline.score` + `pipeline.compare` against the Phase 1 rows.

**Risks.** Qwen2.5 can overflow in fp16 on a T4 (the smoke step catches it; fall back to
fp32, ~2× slower). Kaggle library versions drift (pin them). A tuned model may learn the
generator's house style rather than the rule — Phase 4's torchvision set is the test.

**Later, only if time allows.** A training-size curve (160 / 640 / 1280 samples) for the
compute-vs-accuracy question; the 0.5B; all 160 test samples for the write-up.

### 2. Write up

`pipeline.compare results/*.jsonl --markdown` produces the headline table (now with a
`topo` column). `pipeline.score ... --slides` prints rows for `scripts/make_slides.py`.

## Findings worth carrying forward

**Qwen2.5-Coder 7B, zero_shot + few_shot, done 22 Oct.** Size curve by zero-shot `topo`:
1.5B 0.05 → 7B 0.25 → 14B 0.60 → (Gemma 31B 0.93). The 7B gets 0/20 on tiers 3–4
zero-shot and 1/20 with `few_shot`. **Phase 1 is complete enough to move on**; next is
Phase 2, LoRA fine-tuning the 1.5B on the train split (agreed plan, not started), then
torchvision as the harder benchmark. Before the write-up, rerun the final configurations
on all 160 test samples.

**Qwen2.5-Coder 14B, all six prompts, done 22 Oct** (README has the table). By `topo`:
zero_shot 0.60, few_shot 0.70, cot 0.53, cos 0.72, nodes_only 0.70, edges_only 0.75.
Open issues before the prompting comparison is final:
- `cos.txt` lacked the "expand `nn.Sequential`" rule, so `cos` collapsed on tier 2 (0.30)
  while scoring 0.90 on tier 3. **Fixed 22 Oct** (rule restored, procedure now says one
  line per layer); old run archived in `results/archive/cos-without-sequential-rule/`.
  **Re-run done 22 Oct: topo 0.68** (t1–t4: 1.00 / 0.30 / 0.70 / 0.70), vs 0.72 before.
  Tier 2 did not improve: the model now keeps an extra node for the Sequential container,
  or expands nested containers one level only. Do not keep iterating the prompt on the
  test 40 — that tunes it to the test set; use the val split if `cos` is revisited.
- The format line `[<Class>(<params>)] [<shape>]` makes the 14B write a literal `[]` for
  parameterless layers (18/40 zero-shot). `topo` tolerates it; fixing the wording would
  need every prompt re-run, so decide whether `exact` matters enough.

- Gemma 4 31B gets the **topology** of every tier 1–3 sample right. Its strict misses there
  are all `x.tanh()` written as `call_function` (fx records `call_method`).
- Its tier-4 misses are all transformer blocks: `a, q = self.attn(h, h, h)` traces to two
  `getitem` nodes (including the unused weights) that the model leaves out.
- The 1.5B model gets nothing exact; 12/40 replies run away to 130+ nodes until the token
  cap.

## Known gaps

- **`docs/llm_primer.md`, `docs/presentation_outline.md` and the 22-slide deck in
  `scripts/make_slides.py` still describe quantum circuits.** The deck matters for the next
  TA submission.
- **Spatial shape arithmetic is not validated.** Add it only if models are seen to err there.
- The cost estimate ignores `{nodes}` in `edges_only` and the reasoning text `cot`/`cos`
  write, so it **underestimates those prompts**. Irrelevant while everything is free.
- The OpenRouter Gemma results came from the free endpoint; the Kaggle ones will be a
  4-bit quantised copy. The Kaggle `zero_shot` run doubles as a check that the two agree.

## Things that bit us, worth not repeating

- **The prompts named the parameter format for two layer classes only**, so the first real
  reply — a correct graph — failed to parse on `BatchNorm2d(8)`. All node-producing prompts
  now carry a full label table.
- **The reply cache is keyed by prompt *name*, not prompt text.** Editing a prompt does not
  invalidate cached replies; move the old `results/raw/<run>/` aside (see
  `results/archive/`) or pass `--refresh`.
- **Gemma 4 is a thinking model in Ollama.** Without `"think": false` it spent the whole
  token budget on hidden reasoning and returned an empty reply (288 s on 2× T4). The
  backend now sends `think: false` and treats an empty reply as a failure.
- **Ollama's default context window varies by version and VRAM** and silently drops the
  start of an over-long prompt. The backend sets `num_ctx` explicitly (8192).
- **OpenRouter's quota counter lags successes by minutes.** Compare it against total
  replies, not against the change since a loop started.
- The free endpoint for Gemma was congested: about one reply per 15–20 minutes for days.
  Free OpenRouter is fine for 40 samples, impractical for 200.
- The laptop sleeps overnight and kills long local loops; the app's keep-awake only holds
  while the session is active.
- The in-app browser cannot upload files or paste, and Kaggle's editor stops accepting
  input when the pane is hidden. Import the `.ipynb` by hand instead.
- Earlier: `extract_graph_block` once hardcoded the schema header; an empty reply once
  scored `valid=True`; `fx.all_input_nodes` deduplicates. All fixed — check `parsed` before
  believing `valid`.

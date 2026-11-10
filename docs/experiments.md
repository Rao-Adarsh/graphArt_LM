# Experiments after the closing runs: real code, CoS and the two-step format

*Updated 10 Nov 2026. Covers everything from Session 1 (the first training on real-style
code) to the two-GPU two-step run, with the earlier phases summarised for context. The
narrative up to the closing experiments is in `research_story.md`; this file is the
catalogue of what we tried after it, what each attempt scored, and what we take from it.*

## Why this stretch exists

The closing experiments (RQ4–RQ6) showed that the Phase 2 adapter had learned our
generator rather than the rule: 40/40 on our test programs, but **0/21** right graphs on
real torchvision code, where it drew a plain chain. Everything below tries to move that
torchvision number, first by making the training data look more like real code, then, after
we decided that kind of fix was fitting the test set, by changing *how* the model is asked
to produce the graph.

## How to read the numbers

| Set | What it is | Status |
|---|---|---|
| **test 40** | 10 synthetic programs per tier 1–4, the same 40 for every run since Phase 1 | checks the basic skill survives |
| **torchvision 21** | real blocks and small networks from torchvision, 9–71 nodes (median 19) | the target. **Development data now**: prompt and data choices were tuned on it |
| **torchvision holdout 17** | 10 model families the 21 never use, frozen (sha256 in `data/generated/FROZEN.json`) | never looked at; scored **once**, on the final setup |

- **right graph** (`scripts/classify_errors.py`): exact, or isomorphic once notation
  (module parameters), spelling (`F.relu` vs `nn.ReLU`) or the operand order of `add`/`mul`
  is forgiven. The fairest single "did it get the graph" number.
- **exact**: labelled-graph isomorphism with the torch.fx reference.
- **node F1 / edge F1**: partial credit; edge F1 is averaged per sample.
- `nodes`/`edges` runs score only the half they were asked for.

## What each variant is

**Training data** (all generated, traced with torch.fx for the answers):
- *tiers 1–4* (`graphart/generate.py`): small synthetic programs, 1280 train.
- *tiers 5–6* (`graphart/generate_large.py`): library-style code — helper classes,
  containers, `BasicConv2d`-style units, inception branches; 640 train, tier 6 up to 119
  nodes. **Generator v2** added the torchvision idioms behind Session 1's misses.

**Answer formats:**
- *one step* (`zero_shot`): the whole graph in one reply.
- *two step* (`graphart/two_step.py`): call 1 lists the nodes, each with its **CoS stack**
  (`@64>56>16`: the source lines of the call chain that made it); call 2 is given the
  code with line numbers plus the node list and writes the edges, each input tagged with
  the variable it is read from (`21 <- out=3 arg0`). `assemble()` joins the two. The
  stacks and variable names were recorded during fx tracing (`graphart/cos_trace.py`,
  checked by `scripts/check_cos_trace.py`), not written by hand.
- *brief* vs *full* prompts: the full ones spell out every convention (~1,200 tokens more
  per example); the brief ones only sketch the format.

**Prompting styles** (Qwen2.5-Coder 14B, no training): `zero_shot`; `cot` (think step by
step); `cos` (simulate the code line by line, tracking which node each variable holds);
`few_shot` (worked examples from the train split); `nodes_only` / `edges_only` (one half
of the task, the other half given); `edges_only_cos` (edges given the true nodes, with a
CoS simulation); `nodes_only_few_shot` (two worked tier-5 examples); `*_stack` (nodes with
their CoS stacks).

## Results

### The master table

| # | Method | Model | Trained on | Test 40: right (exact) | Torchvision 21: right (exact) | TV node F1 | TV edge F1 |
|---|---|---|---|---|---|---|---|
| 1 | zero-shot | Qwen 1.5B, untuned | — | 4 (0) | not run | — | — |
| 2 | zero-shot | Qwen 14B | — | 28 (7) | 4 (1) | 0.31 | 0.20 |
| 3 | zero-shot | Gemma 4 31B | — | 37 (30) | not run | — | — |
| 4 | CoS, whole graph | Qwen 14B | — | 27 (12) | — (1 exact) | 0.34 | 0.23 |
| 5 | Phase 2, one step | 1.5B LoRA | tiers 1–4, 2 epochs | **40 (40)** | **0 (0)** | 0.36 | 0.17 |
| 6 | Session 1 "mixed", one step | 1.5B LoRA | tiers 1–6, generator v1 | 40 (40) | 7 (4) | 0.70 | 0.55 |
| 7 | Session 2 "mixed-v2", one step | 1.5B LoRA | tiers 1–6, generator v2 | 40 (40) | 7 (5) | **0.86** | **0.71** |
| 8 | two step, brief prompts | 1.5B LoRA | tiers 1–4 every 2nd + 5–6 | 40 (35) | 7 (5) | 0.69 | 0.58 |
| 9 | two step, full prompts, 65% of an epoch | 1.5B LoRA | tiers 1–4 every 8th + 5–6 | 30 (24) | **9** (5) | 0.77 | 0.64 |
| 10 | two step, full prompts, full epoch (2 GPUs) | 1.5B LoRA | tiers 1–4 every 8th + 5–6 | 33 (26) | 7 (4) | 0.62 | 0.53 |

**Half-task probes, Qwen 14B on torchvision 21** (the other half given correct):

| Probe | Exact | F1 |
|---|---|---|
| edges, given the true nodes | 10/21 | edge 0.91 |
| edges, given the true nodes, with CoS | **14/21** | edge 0.93 |
| nodes only | 1/21 | node 0.30 |
| nodes only, with stacks | 1/21 | node 0.34 |
| nodes only, few-shot | 6/21 (11 topo) | node **0.69** |
| nodes only, stacks + few-shot | 3/21 (8 topo) | node 0.73 |

### Training runs

| Run | Examples | Steps | Minutes | GPUs | Best val loss |
|---|---|---|---|---|---|
| Phase 2 | 1280 × 2 epochs | 160 | 98 | 1 | 0.00009 |
| Session 1 mixed | 1920 | 95 | 96 | 1 | 0.0010 |
| Session 2 mixed-v2 | 1920 | 96 | 96 | 1 | 0.0021 |
| two step, brief | 2560 | 138 | 96 | 1 | 0.0015 |
| two step, full, 65% | 1590 | 64 of ~100 | 90 | 1 | 0.0049 |
| two step, full, epoch | 1590 | 100 | 82 | 2 | 0.0034 |

All are fresh LoRAs (rank 16, alpha 32, lr 2e-4, effective batch 16) on
Qwen2.5-Coder-1.5B-Instruct, one seed each. The full-prompt runs leave out the 10 longest
tier-6 examples (over 6,144 tokens; 8,192 did not fit in a T4's memory).

## What we take from it

1. **Training on code that looks real moves real code, once.** Adding library-style tiers
   5–6 took torchvision from 0 to 7 right graphs (node F1 0.36 → 0.70) without costing
   the test set. That is the largest single gain in this stretch.

2. **After that, the 1.5B sits at 7–9 of 21, whatever we change.** Generator v2, the
   two-step format, brief or full prompts, a full epoch: all land at 7–9. On 21 samples a
   difference of two is within run-to-run noise. Which samples are right also moves
   between runs: across the four tier 1–6 runs (7–10), 12 samples are right at least once,
   only 2 (ResNet BasicBlock, SqueezeNet Fire) are right every time, and 9 are never right.

3. **Patching the generator for each miss was fitting the test set.** Generator v2 added
   exactly the idioms behind Session 1's misses; node and edge F1 rose (0.86 / 0.71) but
   right graphs stayed at 7. We stopped generator rounds there (31 Oct) and changed the
   format instead.

4. **Wiring is the easier half; naming the nodes is the bottleneck.** Given the true
   nodes, the prompted 14B wires 10/21 torchvision graphs exactly, 14/21 with CoS. On its
   own it lists the nodes badly (node F1 0.30), mostly by drawing a helper module as one
   box instead of expanding it. In the brief two-step run, every sample whose step-1 nodes
   were fully right came out exact. The 1.5B's remaining misses are almost all **module
   paths**: `layers.0` written `layers.0.0`, `branch1x1.conv` shortened to `branch1x1`,
   nested blocks of ResNet-18 and SqueezeNet dropped, and (full epoch only) a space for a
   dot in three Inception-v3 samples, which made them unparseable.

5. **CoS helps the step it fits.** As a whole-graph prompt on torchvision it does nothing
   (1/21 exact for the 14B). Applied to wiring alone, given the nodes, it is the best
   result we have (14/21). Recording the CoS stacks during tracing gave us training
   answers that are correct by construction and scale to 100-node programs.

6. **Few-shot is the cheap fix for node listing.** Two worked examples took the 14B's
   node F1 from 0.30 to 0.69. Adding the stacks on top did not help exact matches (6 → 3)
   — the extra output to write seems to cost more than the scaffolding gives.

7. **Rules in the prompt fix the slips they name, and nothing more.** The full-rules
   prompts stopped the brief adapter's `_forward` slip (a method name written into module
   paths on Inception-v3), and the 65% run had the best torchvision score (9). Finishing
   the epoch lowered the validation loss (0.0049 → 0.0034) but not the torchvision score:
   **a lower loss on our generator's data no longer predicts real code.**

8. **Cutting tiers 1–4 costs the small programs.** With every 8th tier 1–4 program (40 a
   tier), test fell from 40 to 30–33. The misses are idioms the small tiers teach:
   `torch.tanh` written as a module call, stray `view`/`size` operations. The brief run,
   with every 2nd program, kept 40/40.

9. **Engineering that now works.** Two-GPU training (DistributedDataParallel via torchrun)
   finishes an epoch in 82 min instead of ~140; the memory-saving loss goes through the
   model's forward so the GPUs stay in sync, and is scaled by the GPU count. Notebooks
   carry the code and data inside them (xz + base64, ~0.5 MB) and download only the base
   model.

## Caveats

- 21 torchvision samples and one training seed per run: differences of one or two graphs
  are not evidence on their own.
- The 21 are development data now. Only the frozen holdout can give an unbiased number.
- Gemma 31B was never scored on torchvision (the free endpoint was rate-limited).
- The 14B probes ran 4-bit on Ollama; the 1.5B runs in fp16 on Hugging Face.

## Where this leaves the plan

Two directions, to run side by side, pending an outside opinion on next steps:

1. **A fully prompted 14B two-step**: few-shot node listing (finding 6), then CoS wiring
   on its own node list (finding 5). No training; tests whether the two best half-task
   results compose.
2. **The two-step format on Qwen2.5-Coder 3B**: whether a larger fine-tuned model names
   module paths better (finding 4), now that two GPUs make a full epoch affordable.

Then: freeze one setup, score the holdout once, run the remaining closing sets
(large_test, edits, RQ6), and write up.

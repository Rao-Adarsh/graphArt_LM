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

## A closer look: which half fails, per model (10 Nov)

`scripts/decompose.py --torchvision` checks every reply twice: is its node list right
(labels as for a right graph), and is the graph right? It also splits edge recall by
whether the reference edge comes from the node listed just before.

| Run | Nodes right | Wired right, of those | Right graphs | Edge recall: adjacent / other |
|---|---|---|---|---|
| Session 1 mixed, one step | 7 | 7 | 7 | 0.71 / 0.54 |
| Session 2 mixed-v2, one step | **15** | 7 | 7 | 0.87 / **0.58** |
| two step, brief (step-1 nodes) | 8 | 7 | 7 | 0.85 / 0.85 |
| two step, full, 65% | 12 | 9 | 9 | 0.77 / 0.72 |
| two step, full, epoch | 10 | 7 | 7 | 0.75 / 0.79 |

This corrects finding 4 for the one-step model. **mixed-v2 lists the right nodes on 15 of
21 but miswires 8 of them**: it connects a node to the one listed before it far more
reliably than to anything else (0.87 vs 0.58) — branch heads fed from the previous node
instead of the block input. The two-step models wire both kinds alike (variable-tagged
edges), but list nodes worse. The best node lister and the best wirer are different
models, and they have not been combined.

The 14B's CoS wiring (14/21 on true nodes) is weaker than it looks on branching code: on
the 11 samples whose nodes can be listed in more than one valid order (branches), it was
exact on 5.

## Where this leaves the plan

After an outside review (Fable 5.2, 10 Nov) checked against the saved replies, the 3B
training is dropped — it addresses neither failure above, and a full-prompt epoch would
not fit a session — and the prompted 14B two-step becomes a composition test. Each
criterion is fixed before the run.

| | Experiment | Run on | Success | Failure means |
|---|---|---|---|---|
| **E1** | 14B CoS wiring handed **mixed-v2's node lists** (`run_prompting --nodes-from`) | torchvision 21 | ≥ 12 right graphs, and ≥ 10 of the 15 node-right samples wired right (ceiling 13) | wiring a model's node list is harder than wiring fx's; the 14/21 probe overstated it |
| **E2** | the 193 controlled edits on **mixed-v2 (one step)** and the **brief two-step** adapter, both 40/40 on test | edits | two-step answers meaningful edits with the unedited graph ≥ 20 points less often (exact McNemar p < 0.05, n = 69), neutral edits still ≥ 85% right | both formats pattern-match; the honest negative for RQ5 |
| **E3** | 14B CoS wiring on the **true nodes in a random valid order**, branches interleaved (2 seeds) | the 11–12 samples whose order can change | keeps ≥ 4 of its 5 (seed 1) / 6 (seed 2) true-order successes | ≤ 2: it leans on listing order, not on tracking variables |
| E4 | only if E1 passes: a 1.5B step-2 wirer trained on stack-free node lists, fed mixed-v2's nodes | torchvision 21 | ≥ 11 right, ≥ 38/40 test | keep the 14B wirer |

Notebooks: `kaggle_probe_14b_compose.ipynb` (E1 and E3, Ollama) and
`kaggle_edits_two_adapter.ipynb` (E2, both adapters attached as inputs).

### E2 result (10 Nov): passed

Both adapters ran all 193 edits on two T4s in one session (`kaggle_results/graphart_edits_eval`),
scored with `classify_errors.py --solved` on each adapter's own test run (both 40/40, so all
193 edits count). Wilson 95% intervals in brackets.

| Edits | | mixed-v2, one step | two step, brief | Paired (exact McNemar) |
|---|---|---|---|---|
| meaningful (69) | gave the **unedited** graph | 39 — 57% (45–68%) | **14 — 20% (12–31%)** | 26 vs 1, p ≈ 4e-7 |
| | right graph | 17 — 25% (16–36%) | **47 — 68% (56–78%)** | 0 vs 30, p ≈ 2e-9 |
| neutral (124) | right graph | 111 — 90% (83–94%) | 109 — 88% (81–93%) | 13 vs 11, p = 0.84 |

The criterion was ≥ 20 points fewer unedited graphs at p < 0.05 with neutral edits ≥ 85%;
the gap is 36 points. Every meaningful edit the one-step model got right, two-step also got
right.

By edit type (right / unedited, of n):

| Edit | n | mixed-v2 | two step |
|---|---|---|---|
| reroute (an input now comes from elsewhere) | 27 | 1 / 17 | 17 / 6 |
| swap_cat (concatenation order swapped) | 9 | 0 / 9 | 9 / 0 |
| drop_call (a call removed) | 29 | 16 / 9 | 21 / 4 |
| branch_on (a flag switches a branch on) | 4 | 0 / 4 | 0 / 4 |
| alias, dead_flag, rename, reorder (neutral) | 124 | 111 / – | 109 / – |

Reading it:
- The one-step model mostly recalls the program it was trained on: rerouted inputs and swapped
  concatenations come back with the original wiring. The two-step model, which must name the
  variable each edge reads, follows the edit far more often — the same mechanism that gave it
  0.85 recall on non-adjacent edges above.
- Both miss all four `branch_on` edits; neither reads a flag's value to decide what runs.
- Two-step's neutral errors are mostly on `alias` (29/40 vs 37/40): it lists the alias line
  (`h1_ref = h1`) as an extra node. Step 1 ties nodes to source lines, and an assignment that
  computes nothing still looks like one.
- Caution: the two adapters differ in training data as well as output format (mixed-v2: all
  of tiers 1–6 from generator v2; brief two-step: tiers 1–4 every 2nd program plus 5–6), so
  this compares the two adapters, not the format alone.

### E1 and E3 results (10 Nov): E1 passed, E3 failed

One Kaggle session, 14B 4-bit on Ollama (`kaggle_results/graphart_14b_probe_compose`).
One E1 reply timed out after 600 s (`tv-googlenet-aux`); mixed-v2's node list for it was
already wrong, so it does not change the count.

**E1, mixed-v2 lists the nodes, the 14B wires them with CoS: 13 of 21 right graphs**
(Wilson 41–79%), 13 of the 15 node-right samples wired right — exactly the ceiling. Both
misses (`inceptionv3-e`, `shufflenet-unit`) were also missed when the 14B wired the true
nodes.

| Compared with | Right | Only E1 right / only the other | Exact McNemar |
|---|---|---|---|
| mixed-v2 alone, one step | 7 | 6 / 0 | p = 0.031 |
| best single 1.5B (two step, full, 65%) | 9 | 5 / 1 | p = 0.22 |
| 14B zero-shot | 4 | 10 / 1 | p = 0.012 |

The composition fixes exactly what the decomposition predicted: the six samples gained over
mixed-v2 alone are all ones where mixed-v2 listed the right nodes and wired them by
adjacency (the GoogLeNet Inception block, Inception-v3 A–D, AlexNet).

**E3, the 14B wiring the true nodes listed in a random valid order: it relies on order.**

| Seed | Samples reordered | True-order successes among them | Kept after reordering |
|---|---|---|---|
| 1 | 11 | 5 | **1** |
| 2 | 12 | 6 | **3** |

The criterion was to keep ≥ 4 of 5 / ≥ 5 of 6; it kept 4 of 11. When the branches of an
Inception block are interleaved, the 14B wires each node to the one listed before it more
than to the variable it reads. Its CoS stacks do not make it track variables on their own.

What the two say together: the composition works **because** mixed-v2 lists nodes in the
order the code is written, and the 14B leans on that order. That is a property the pipeline
needs, and it is worth stating as a limit: the wirer is not order-independent. The 1.5B
two-step adapters, which name the variable on every edge, may be — E2 suggests they read
variables — but that was not tested here.

Caution: mixed-v2's node lister comes from a generator patched on these 21 samples, so 13/21
is a development number; only the frozen holdout can confirm it.

### E4 (set up 10 Nov): a 1.5B wirer

E1 passed, so E4 runs: `notebooks/kaggle_e4_wirer.ipynb` (`make_kaggle_notebook.py --e4-wirer`).
A fresh LoRA trained on step 2 alone (`train_lora.py --wirer`): the brief edges prompt,
given the reference nodes with their stacks removed — the node list as mixed-v2 writes it —
answered with the variable-tagged edges. Same programs as the brief two-step adapter (every
2nd tier 1–4 program, all of 5–6): 1,280 examples, ~1,150 tokens each, one epoch on two T4s.
Checked locally: on all 40 test programs the node list it trains on is, line for line, the
one it is handed from mixed-v2's replies.

Fixed before the run:

| Run | Handed | Success |
|---|---|---|
| **E4** | mixed-v2's node lists, torchvision 21 | ≥ 11 right graphs |
| | mixed-v2's node lists, the 40 test programs | ≥ 38 right |
| diagnostic | the true nodes, torchvision 21 | none — compared with the 14B's 14 |
| diagnostic | the true nodes reordered (E3's two seeds) | keeps ≥ 80% of its own true-order successes on those samples, where the 14B kept 4 of 11 |

If E4 passes, the pipeline is two 1.5B adapters (mixed-v2 lists, the wirer wires) and is the
candidate to freeze. If it fails, the 14B wirer of E1 is the candidate.

### E4 result (15 Nov): passed; the order check failed

Training took 80 steps, 21.6 min on two T4s, and reached val loss 0.0026 (`kaggle_results/graphart_e4_wirer`).

| Wirer handed | Right | Criterion |
|---|---|---|
| mixed-v2's node lists, torchvision 21 | **11** (Wilson 32–72%) | ≥ 11: met |
| mixed-v2's node lists, 40 test programs | **40** (all exact) | ≥ 38: met |
| the true nodes, torchvision 21 | **16** (the 14B: 14) | diagnostic |
| the true nodes reordered, seeds 1 / 2 | 0 / 1 | keeps 0 of its 7 + 7 true-order successes |

Two torchvision replies with mixed-v2's nodes ran out of GPU memory: three scoring jobs
shared GPU 0. Both were samples where mixed-v2's nodes were already wrong (`googlenet-aux`,
`resnet18`), so the count is unchanged. The holdout run should use one job per GPU.

Paired comparisons (exact McNemar, discordant pairs):

| | Right | Only this / only the other | p |
|---|---|---|---|
| E4 vs mixed-v2 alone | 11 vs 7 | 5 / 1 | 0.22 |
| E4 vs E1 (14B wirer) | 11 vs 13 | 1 / 3 | 0.63 |
| wirer vs 14B, both on true nodes | 16 vs 14 | 5 / 3 | 0.73 |

Reading it:
- **The composition loses nothing.** With mixed-v2's nodes, the wirer is right exactly on the
  samples where mixed-v2's nodes are right *and* the wirer is right on the true nodes (11 of
  them). The same held for the 14B in E1.
- **The 1.5B wirer is as good as the 14B at wiring** (16 vs 14 on true nodes; the difference
  is noise). Its misses on true nodes are Inception-v3 C, D and E, the ViT block, and the
  downsampling basic block.
- **The node lister is now the limit.** The wirer wires the true nodes of `resnet18`,
  `resnet-bottleneck`, `squeezenet1_1`, `shufflenet-unit-down` and `googlenet-aux`, but
  mixed-v2 lists the wrong nodes for all five.
- **Neither wirer is order-independent.** Reordered, the wirer misses edges even between
  nodes with unique names: in seed 2, Inception-v3 A loses 9 of those 13 edges. Part of the
  loss is unavoidable, because without stacks two `add` or `cat` nodes can be told apart only
  by their order. But that does not explain the Inception losses. Naming the variable on each
  edge (E2) makes the model follow the code locally. It still uses the listing order to know
  which line a node stands for. The pipeline works because mixed-v2 lists nodes in code
  order; this is a limit to state, not something the frozen setup depends on fixing.
- On torchvision 21, E4 is not significantly better than mixed-v2 alone (5–1). The case for
  it is the mechanism (the decomposition, E2, and the lossless composition), and the holdout.

## Freeze (15 Nov)

The final setup is **mixed-v2 + the 1.5B wirer**: two 1.5B adapters, no 14B. On torchvision 21
the 14B wirer was ahead by 13 to 11, but that is a 3–1 split; the two-adapter pipeline is
the "a small model can do this" result and runs in one session.

`data/generated/FROZEN.json` has a dated amendment, written before any reply on the holdout
exists. It fixes the pipeline (zero_shot, then `run_two_step --brief --step1-from`, caps
8000), the three adapters by sha256, the baselines (mixed-v2 alone; the 26 Oct submission's
adapter; optionally the 14B zero-shot in a separate session), and the comparisons (right
graphs, exact McNemar, n = 17, blocks and whole networks separately).

`notebooks/kaggle_holdout.ipynb` (`make_kaggle_notebook.py --holdout`) runs it once: GPU 0
mixed-v2 then the wirer, GPU 1 the submitted adapter. It was tested locally only with
throwaway LoRAs on a tiny random model, so no candidate model has produced a holdout reply.
The adapters to attach are the three zips in `dist/`.

Torchvision 21, before and after (development data; the generator was patched on these
after Session 1, so only 0 → 7 is a clean comparison):

| | Right of 21 |
|---|---|
| 26 Oct submission, fine-tuned 1.5B | 0 |
| 26 Oct submission, prompted 14B zero-shot | 4 |
| Session 1 mixed (before the generator saw these errors) | 7 |
| final: mixed-v2 + 1.5B wirer | 11 |

## Holdout result (scored once, 15 Nov): 0 of 17 for every model

`kaggle_results/graphart_holdout`: all three runs finished, 17 of 17 replies each, none
failed; the wirer joined 16 of 17. Scored with the unchanged scorer and classifier.

| Torchvision holdout (17) | Right graphs | Node lists right | Parsed | Valid | Node F1 | Edge F1 |
|---|---|---|---|---|---|---|
| 26 Oct submission | **0** | 0 | 0.77 | 0.12 | 0.19 | 0.07 |
| mixed-v2 alone | **0** | 0 | 0.94 | 0.35 | 0.34 | 0.15 |
| **final pipeline** (mixed-v2 + wirer) | **0** | 0 | 0.94 | 0.59 | 0.34 | 0.16 |

The declared comparisons have no discordant samples (0–0, p = 1). Wilson 95% for 0/17 is
0–18%. Blocks (14) and networks (3) separately: no right graph in either. On blocks, node F1
is 0.22 → 0.35 and edge F1 is 0.09 → 0.16 (submitted → final); on networks, node F1 is
0.06 → 0.26. The partial scores rose, but no graph is whole.

**The gains on the development set did not carry over to unseen torchvision families.**
mixed-v2 lists the right nodes for none of the 17, so the wirer, however good, had nothing
to wire. In `docs` terms, the node lister was the limit on torchvision 21 (15/21) and it is
the whole failure here. What the replies show:
- **Library layers whose code is not in the prompt.** 13 of the 17 holdout programs import
  layers from torchvision (`Conv2dNormActivation`, `SqueezeExcitation`, `StochasticDepth`,
  `Permute`, …), against 5 of the 21 development samples. fx expands them into
  Conv2d → BatchNorm2d → activation; the models write one box per layer
  (`block.0.0 Conv2dNormActivation`). Getting these right needs torchvision's internals,
  which the prompt never shows.
- **Configuration-dependent branches.** In `mobilenetv3-hs-down` the model adds a
  squeeze-excitation layer and a residual `add`, though with that configuration (no SE,
  stride 2) neither runs. This is E2's `branch_on` failure (0/4 for both adapters) on real
  code.
- **Constructs never trained on**: 3D convolutions and their functional form (`conv3d` with
  weights as `get_attr`), InstanceNorm, `interpolate`, `getattr` on a module.
- Even at the coarser level (each composite layer kept as one node and re-traced), no reply
  is right. For 6 programs the classifier's coarse re-trace raised an AttributeError, so they
  stay `wrong_ops`; the protocol fixes the classifier, so this is reported, not fixed.

What this means for the claims: the composition (E1, E4), the decomposition and E2 are
findings about **how** these models fail and what fixes part of it; they hold on the data
they were measured on. The torchvision 21 result (0 → 11) **does not generalise**: it was
partly fitted to those 21 (the caution written before the run), and real code that leans on
library layers and configuration is beyond what the generator teaches. The frozen holdout is
what showed this, which is what it was for.

How results are compared from here: exact McNemar on the discordant samples between two
runs (at n = 21 a 5–0 split is p = 0.06, 6–0 is p = 0.03; less is noise) and Wilson 95%
intervals (7/21: 17–55%; 9/21: 24–63%; 13/21: 41–79%).

Then **freeze** one setup and amend `data/generated/FROZEN.json`, which says the holdout is
scored with the `zero_shot` prompt — a pipeline needs that changed, dated, before scoring.
Score the holdout once, together with pre-declared baselines (14B zero-shot, mixed-v2 one
step). A caution to write down first: mixed-v2's node lister comes from a generator patched
on these 21 samples, so expect the holdout nearer Session 1's level than mixed-v2's.
Then the remaining closing sets (large_test, edits, RQ6) on the frozen setup, and write up.

## Hierarchical schema: evaluation criteria (written 22 Nov, before the results were read)

Setup: the hier adapter (fresh LoRA on Qwen2.5-Coder 1.5B, tree + wiring tasks, one epoch,
best val loss 0.0080), `pipeline/run_hier.py` with `--tree-tokens 8192 --wire-tokens 2048`,
scored by `pipeline/score_hier.py`. Dev = `hier_dev` (51: the old 21 torchvision blocks + 30
whole networks); `hier_gen_test` = 75 held-out generator programs. The criteria were proposed
in the session on 22 Nov before the eval zip was opened; the Kaggle run itself had already
finished when they were written here, but no result had been read.

| # | Question | Measure | Criterion |
|---|---|---|---|
| C1 (main) | Whole networks right? | `right` (flat graph after inlining == fx, node labels) on the 30 dev networks | ≥ 6/30 (Wilson 95% at 6/30: 10–38%, above the flat schema's 0); strong ≥ 15/30 |
| C2 | No worse on the old 21 blocks? | paired with the old final pipeline (mixed-v2 nodes + 1.5B wirer) on the same 21 | strict `right` ≥ 6/21 and `flat_topo` ≥ 11/21; exact McNemar on discordant pairs reported. Caveat: the old generator was patched on these 21 |
| C3 | Pipeline sound? | `right` on gen_test | ≥ 68/75 (90%); below that, fix training/assembly before reading C1/C2 |
| C4 | Where does it fail? (descriptive) | first failing part; own-tree vs oracle-tree wiring on dev | wiring right on all variants for ≥ 80% of dev programs ⇒ the tree is the bottleneck |

Decision rule: the hier holdout and timm test are scored once regardless of these results,
after amendment 2 of `FROZEN_v2.json` declares the model, runner, reducer, caps and their
criteria. Dev results decide what is claimed, not whether the frozen run happens. No tuning
loop on dev.

## Hierarchical schema: dev result (22 Nov)

`kaggle_results/graphart_hier_eval`: three runs, all programs answered (dev 51, oracle-tree
dev 51, gen_test 75); 8 dev tree replies were unreadable (counted as `parse`). Scorer fix
made after the run, before reading the criteria: `tree_exact` compared layer parameters as
ordered text, and dev truth stores them sorted, so `tree` (with params) read 0/51. It now
compares them order-free (`graphart/hier_score.py`, `modules`). `right`, wiring and
variants do not depend on it and did not change.

| Run | n | parse | tree (no params) | tree | variants | all wiring | flat topo | **right** | Wilson 95% |
|---|---|---|---|---|---|---|---|---|---|
| dev, blocks (old 21) | 21 | 21 | 13 | 10 | 18 | 15 | 11 | **9** | 24–63% |
| dev, networks | 30 | 20 | 3 | 0 | 4 | 7 | 3 | **0** | 0–11% |
| dev, oracle tree, blocks | 21 | — | — | — | — | 16 | 16 | **16** | 55–89% |
| dev, oracle tree, networks | 30 | — | — | — | — | 10 | 10 | **10** | 19–51% |
| gen_test, blocks | 24 | 24 | 23 | 23 | 23 | 22 | 23 | **23** | 80–99% |
| gen_test, networks | 51 | 51 | 39 | 35 | 32 | 35 | 32 | **28** | 41–68% |

Variant graphs right: dev 63/131 (own tree), 96/131 (oracle tree); gen_test 240/262.

| Criterion | Result | Verdict |
|---|---|---|
| C1 networks ≥ 6/30 | 0/30 (0–11%) | **fail** |
| C2 strict ≥ 6/21 and topo ≥ 11/21 | strict 9/21 (old 6; new-only 5, old-only 2, p = 0.45); topo 11/21 (old lenient 11; 6–6, p = 1) | **pass** (not worse; no significant gain) |
| C3 gen_test ≥ 68/75 | 51/75 (57–77%); blocks 23/24, networks 28/51 | **fail** |
| C4 wiring-alone ≥ 80% of dev | 26/51 programs (51%) all wiring right with the oracle tree | **below 80%**: wiring is a bottleneck too |

Own tree → oracle tree on dev: 9 → 26 right (17–0 discordant, p = 2e-5); blocks 9 → 16
(7–0, p = 0.016); networks 0 → 10 (10–0, p = 0.002).

C3 check (is the pipeline broken?): no program fails at assembly in any run, and the truth
scores 75/75 and 51/51 through the same runner and scorer. The 24 gen_test misses are tree
errors (13), variant partition (6), params (4) and wiring (1). Examples: channel and group
arithmetic in stage widths (`out=72, g=9` for `out=64, g=4`); the downsampling skip written
as AvgPool + conv when `down_type` gives a strided conv; staged-ViT stages 1–2 given one
variant though the strided conv between stages makes two. gen_test networks right by flat
size: ≤100 nodes 10/20, 101–200 11/20, >200 7/11. By family (networks): mbconv 0/4, regnet
1/7, convnext 1/4, dense 1/3, vit 9/15, mixer 5/5, vgg 4/4, shuffle 4/5.

Dev failures, by first failing part: blocks right 9, tree 8, params 3, wiring 1; networks
tree 17, parse 10, params 3. Of the 10 network parse failures, 8 replies are PyTorch
`print(model)`-style listings (`weight: torch.Size([64])`, `drop_rate: 0`,
`layer_scale: Parameter containing:`), 2 (regnet) break on node names; 3 replies run on
past 3× the truth's length (mobilenet_v3_small, deeplabv3_mobilenet_v3_large, s3d).
Oracle-tree wiring failures (35 variants), recurring kinds: configuration-dependent
residuals (`InvertedResidual`, `_InvertedResidual`, `MBConv`, `FusedMBConv`: 3 vs 4 nodes,
the residual add added or dropped), whole-network forwards of GoogLeNet / Inception3
(22 vs 31, 24 vs 40 nodes), `ASPPPooling`, ViT `EncoderBlock`, the 3D-conv subclasses
(`Conv3DSimple`, `Conv3DNoTemporal`: 4 vs 2), ShuffleNet `InvertedResidual` (right size,
wrong edges).

## Changed criterion (22 Nov, after the dev results were read)

The user's decision: the expanded fx graph carries no weight; only the hierarchical drawing
counts. New headline `hier_right`: tree right with layer parameters, the same variant
partition, and every visible variant's wiring graph isomorphic to the truth — no inlining.
`flat exact` is kept as a column. Because the change came after the dev results were seen,
the dev numbers below are descriptive, not a test of C1–C4; the criterion is a test only on
the frozen sets, once declared in amendment 2. Scorer: `hier_right` in
`graphart/hier_score.py`, headline of `pipeline/score_hier.py`.

| Run | n | tree (no params) | tree | variants | variant graphs right | all wiring | **hier right** | Wilson 95% | tree F1 | flat exact |
|---|---|---|---|---|---|---|---|---|---|---|
| dev, blocks | 21 | 13 | 10 | 18 | 26/32 (81%) | 15 | **9** | 24–63% | 0.91 | 9 |
| dev, networks | 30 | 3 | 0 | 4 | 37/99 (37%) | 7 | **0** | 0–11% | 0.41 | 0 |
| dev oracle tree, blocks | 21 | given | given | given | 27/32 (84%) | 16 | **16** | 55–89% | — | 16 |
| dev oracle tree, networks | 30 | given | given | given | 69/99 (70%) | 10 | **10** | 19–51% | — | 10 |
| gen_test, blocks | 24 | 23 | 23 | 23 | 50/52 (96%) | 22 | **21** | 69–96% | 0.98 | 23 |
| gen_test, networks | 51 | 39 | 35 | 32 | 190/210 (90%) | 35 | **24** | 34–60% | 0.95 | 28 |

On dev the two headlines agree on every program. On gen_test 6 programs are flat-right but
not hier-right: in 4 the truth wires the consumer of an `nn.Identity` placeholder (a
`drop_path` with rate 0) past it, leaving the Identity node with no consumer, while the model
wires `add <- drop_path1` as the code reads; in 2 the model splits one variant into two
(`MixerBlock`, `CNBlock`) with each graph right. Both are truth-convention questions for the
amendment, not changed here.

## Why the trees fail (22 Nov, dev and gen_test, descriptive)

Module-level errors in the parsed network trees (paths + class, then layer parameters on
the leaves whose path and class match):

| Networks | parsed | truth modules | missing | extra | wrong class | leaf params wrong |
|---|---|---|---|---|---|---|
| gen_test | 51/51 | 5,361 | 109 (2.0%) | 91 | 31 | 171/2,794 (6.1%) |
| dev (torchvision) | 20/30 | 2,847 | 583 (20%) | 1,993 | 68 | 416/1,120 (37%) |

Dev examples, by kind of error:
- loop and repeat counts: resnet50 / resnext50 get 4 blocks in `layer4` (truth 3), mnasnet
  and shufflenet stages off by one or two, efficientnet_v2_s 142 of 272 modules missing
  (stage repeats from the config table times the depth multiplier);
- computed widths: wide_resnet50_2 72/109 leaf params wrong (`width_per_group`), mnasnet
  78/92 (`_round_to_multiple_of(c * alpha)`), shufflenet 97/102 (`oup // 2` branches),
  mobilenet_v2 11/14;
- conditions: resnet18 adds a `downsample` to `layer1.0` (`stride != 1 or inplanes != ...`
  is false there); googlenet drops the aux heads;
- calls into another builder: deeplabv3_resnet50 72 of 183 modules (the backbone built by
  `resnet50(...)` inside `IntermediateLayerGetter`);
- runaway output: mobilenet_v3_small 858 and deeplabv3_mobilenet_v3_large 862 predicted
  modules (truth 73, 119).
- alexnet and vgg11 (no config arithmetic): structure exact, 1 leaf param wrong each.

Sizes are not the difference: dev networks have median 9.5k chars of reduced code (train v3
networks 4.3k, train timm networks 20.8k) and 84 tree lines (train 70 and 100).

Old vs new in-distribution test: the flat schema's 40/40 was on `test 40` (tiers 1–4:
median 12 flat nodes, max 24, one class, 25% with a loop in the code). hier gen_test has
median 88 flat nodes, 4 classes, 81% with a loop. gen_test blocks, the closer match, are
21/24.

## Two runs set up 22 Nov (criteria written before either ran)

Both on the 51 dev programs only; headline `hier_right` (amendment 2), comparisons paired on
the same programs with exact McNemar on discordant programs (at n = 30: 6–0 is p = 0.03).
Baseline = the hier 1.5B adapter: networks tree (no params) 3/30, tree 0/30, hier_right 0/30;
all 51: hier_right 9/51; oracle-tree wiring all right 26/51, variant graphs 96/131.

**P1, prompted Qwen2.5-Coder 14B** (`notebooks/kaggle_hier_14b.ipynb`): Ollama q4_K_M, two
worked examples from the train split (`prompts/hier_shots2.json`: the smallest v3 block with
≥ 2 variants and the smallest v3 network with a collapsed repeat and ≥ 3 variants, after a
legend of how each layer's parameters are written, from `graphart/extract.py`), tree cap
10,240 tokens, wiring 2,048. Two runs: whole answer, and wiring alone (`--oracle-tree`).
- P1a, trees: does size fix the constructor arithmetic? Better if the 14B's tree (with
  params) is right on more dev networks than the 1.5B's, discordant ≥ 6–0.
- P1b, wiring alone: 14B vs 1.5B on all-wiring-right programs (oracle tree), McNemar.
- P1c, composition, computed from the runs: 14B tree with the better oracle-tree wirer.
  hier_right of a composition = that tree right (with params, same partition) and that
  wirer's oracle-tree wiring all right (exact up to tag names, since the wiring prompt is
  then the truth's).

**P2, per-instance trees** (`graphart/hier_inst.py`, `notebooks/kaggle_hier_inst_train.ipynb`):
a fresh 1.5B LoRA writes one class instance's children given the class's code and its
constructor arguments (the root: the whole program); code joins the answers. Training data
from the same 1,810 training programs; every train/val program's true answers re-assemble to
its stored tree exactly (checked by `scripts/build_hier_inst.py`). Tonight: training only.
Its scoring notebook comes next, with this criterion fixed now:
- P2a, trees: per-instance tree (with params) right on more dev networks than the
  whole-tree 1.5B (0/30), discordant ≥ 6–0. Reported with tree (no params), tree F1, and
  hier_right with the hier adapter wiring the per-instance tree.

## P1 result (23 Nov): the prompted 14B does not fix the trees, and wires worse

`kaggle_results/graphart_hier_14b`: all 51 dev programs answered in both runs. 21 tree
replies were unreadable to the runner (19 of them write list parameters with commas,
`AdaptiveAvgPool2d(out=[6, 6])`), so those 21 got no wiring calls. Scorer fix found here:
`score_answer` let a parameter `ParseError` escape instead of scoring `parse` (now caught;
earlier runs never reached it because the runner failed first).

Strict = the declared scorer. Lenient (post hoc, descriptive, applied to both models alike)
forgives notation only: list params with commas, `g=1` written out, function vs tensor-method
spelling (`.cat()` / `cat()`), argument order of add / mul; the 14B's unreadable trees are
re-read from their raw replies.

| Dev (51) | tree (no params) net / blk | tree net / blk | variant graphs right | all wiring | **hier_right** |
|---|---|---|---|---|---|
| 1.5B hier adapter, own tree | 3/30 · 13/21 | 0/30 · 10/21 | 63/131 | 22 | **9** |
| 14B 2-shot, own tree, strict | 0/30 · 15/21 | 0/30 · 11/21 | 7/131 | 1 | **0** |
| 14B 2-shot, own tree, lenient | 3/30 · 17/21 | 2/30 · 12/21 | 22/131 | 5 | **3** |
| 1.5B, oracle tree | — | — | 96/131 | 26 | **26** |
| 14B, oracle tree, strict | — | — | 10/131 | 0 | **0** |
| 14B, oracle tree, lenient | — | — | 50/131 | 13 | **13** |

(The 1.5B's numbers are the same strict and lenient.)

| Criterion | Result | Verdict |
|---|---|---|
| P1a 14B trees right on more networks, ≥ 6–0 | strict 0–0; lenient 2–0 (p = 0.5) | **not met** |
| P1b wiring alone, 14B vs 1.5B | strict 0 vs 26 programs (0–26, p < 1e-7); lenient 13 vs 26 (0–13, p = 0.0002) | **1.5B better** |
| P1c composition 14B tree + 1.5B wiring | strict 9/51 (networks 0/30); lenient 12/51 (networks 2/30) | same as the 1.5B alone (strict) |

Size alone does not run the constructors: the 14B's network trees are no more often right
than the 1.5B's, and as a prompted wirer it is far behind the tuned 1.5B even after notation
is forgiven.

## P2 training (23 Nov)

`kaggle_results/graphart_hier_inst_train`: one full epoch, 769 steps of 16 on both T4s,
91.6 min; val loss 0.0303 / 0.0246 / 0.0203 / 0.0213 / 0.0202 / 0.0197 / **0.0196** (step
700, kept) / 0.0196. 12,304 train / 666 val examples, none over the length limit. Not scored
yet: P2a needs the per-instance runner and its scoring notebook.

## P2 scoring set up (23 Nov)

`pipeline/run_hier_inst.py`: per-instance calls (inst adapter), joined by
`hier_inst.assemble`; wiring per distinct block instance, deepest first (hier adapter); tags
by matching graphs per class; never-run blocks (`#0`) where the nearest block ancestor's
graph does not reach them. Checked with true answers on every call: 99/99 val programs
`hier_right`. A bare-named class from the code (no `__init__` of its own, e.g.
`LayerNorm2d(nn.LayerNorm)`) is treated as a childless block. Notebook
`notebooks/kaggle_hier_inst_eval.ipynb` (dev, then gen_test, each split over both T4s;
inputs: both adapters). Criterion P2a as written on 22 Nov.

## Lighter schemas, scored on existing replies (23 Nov, descriptive, after results were seen)

What the hier adapter's existing answers would score if less were asked (truth for each
level derived from the same fx-grounded truth):
- S3 current: tree with params + variant partition + every wiring graph (`hier_right`);
- S2 no layer parameters;
- S1 skeleton: tree classes and nesting, runs of identical numbered children counted once
  (no repeat counts, no parameters), plus per class the set of graphs = the set of true
  variant graphs (no instance-to-variant assignment);
- S0 class graphs only: per class in the code, the set of graphs = the true variants' (S0
  any: each graph written matches some true variant). No tree at all.

| | S3 | S2 | S1 tree | S1 | S0 | S0 any |
|---|---|---|---|---|---|---|
| dev blocks (21), own pipeline | 9 | 11 | 13 | 11 | 16 | 16 |
| dev networks (30), own pipeline | **0** | 3 | 8 | 5 | **9** | **13** |
| dev networks (30), true tree given | 10 | 10 | 30 | 10 | 10 | 16 |
| gen_test blocks (24) | 21 | 21 | 24 | 22 | 22 | 24 |
| gen_test networks (51) | 24 | 28 | 44 | 30 | 35 | 48 |

The replies were produced for S3 (the tree is written, then used to prompt the wiring); for
S1/S0 the tree acts only as a scratchpad. Not a test: any schema adopted from this must be
declared (FROZEN_v2 amendment 3) before the frozen sets are scored.

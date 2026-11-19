# Plan: the hierarchical schema (from 19 Nov 2026)

*Working plan, not report text. Dates assume the ~1 Dec submission (unconfirmed).*

## Why

The flat fx graph grows with the unrolled network: real torchvision networks are 155–433
nodes (median 188), our tier-6 programs are capped at 120, and the dev 21 are blocks of
median 19. On the frozen holdout every model scored 0/17. The project brief asks for
"layers, neurons within layers, and their connectivity, **hierarchy** and other topological
properties" from "the forward() method and neural network class code" — so we describe
the network the way its code is written: a module tree, plus one small graph per class
`forward()`. fx stays the answer key: inlining the graphs into the tree must give back
the flat fx graph exactly.

Measured on 16 full torchvision networks (children traced as opaque nodes, identical
blocks counted once): ResNet-50 177 flat nodes → 7 graphs, 61 nodes, largest 13;
Inception-v3 316 → 7 graphs, largest 24; DenseNet-121 433 → 8 graphs, largest 27. No
wiring graph above 27 nodes.

## The schema (v1 spec, to be fixed in phase 1)

Two sections in one answer, or two answers in the pipeline.

**A. Module tree** (layers, neurons, hierarchy — read off `__init__`)
- One line per module, indented by nesting, named by attribute (`layer1`, `0`, `conv1`).
- Built-in `torch.nn` layers carry their parameters as today (`Conv2d(16→32,k3,s2)`,
  `BatchNorm2d(32)`, `Linear(32→10)`): this is "neurons within layers".
- A class **defined in the shown source** carries a variant tag: `BasicBlock#2`.
- A class **not in the source** (imported, e.g. `Conv2dNormActivation`) is an opaque leaf:
  class name only. This is correct by definition — the code does not show its insides.
- Consecutive identical siblings collapse: `0–2: 3× BasicBlock#1`.
- Containers: `nn.Sequential` (and subclasses defined in the source with no own
  `forward`) is a chain and gets no graph; `nn.ModuleList`/`ModuleDict` are transparent
  (the parent's graph refers to `layers.0` directly).

**B. Wiring graphs** (connectivity — one per variant, read off `forward()`)
- `graph BasicBlock#2`, then numbered nodes: `input`, direct children by attribute name
  (a child used twice appears twice), functions/methods (`add`, `cat`, `flatten`,
  `.view`), `output`. No parameters, no shapes.
- Edges as `← src, src` with argument slots as in the current schema (order matters for
  `cat`, `sub`).
- Variant = (class, wiring structure). Variants are numbered by first appearance in a
  pre-order walk of the tree; the scorer matches variants up to renumbering.
- No variable tags on edges (decided 19 Nov). Known cost: E2 found the tagged format
  followed code edits far better (unedited graph 14/69 vs 39/69), so watch for
  wiring-by-adjacency errors (a node fed from the one listed before it).

**Derived by code, not written by the model:** the flat graph (inline), shapes (from fx,
for drawing only), depth, merges, skip connections, fan-in, block counts.

## Pipeline

- **Fine-tuned:** one fresh LoRA on base Qwen2.5-Coder-1.5B-Instruct, trained on two
  tasks told apart by the prompt: (1) whole program → tree; (2) one class's code + its
  child names → that class's wiring graph(s). Inference: one tree call, one wiring call
  per class defined in the source, then code assembles and validates. Not started from
  the old adapters (they learned the flat format).
- **Prompted large model:** Qwen2.5-Coder 14B (Ollama, Kaggle) on the same two prompts,
  few-shot from train. Free. Covers "whether large models can reliably reason about
  topology".
- Only if time allows: a 3B LoRA (shorter sequences now make it feasible), or a second
  adapter so tree and wiring are separate models.

## Data

- **Training:** `generate_large.py` programs, changed to look like whole real networks:
  no 120-node cap; 3–4 levels of nesting; stages of 2–6 blocks; 2–3 variants of a block
  in one network (downsample on/off, stride, flags); imported layers with no source
  (opaque leaves); `_make_layer`, config tables, injected `norm_layer`/`block`
  arguments. Small share (~10–20%) of tiers 1–4 as insurance (finding 8).
- **Dev (new, replaces the 21):** whole torchvision networks plus blocks from a fixed list
  of families, ~40–60 samples, sample = the model file's classes + one construction line.
  The 17 old holdout families may go here (we have read their failure analysis).
- **Holdout (new, frozen before any generator change or training):** families never
  looked at — candidates Swin, MaxViT, MViT, detection/segmentation heads (FCN, LRASPP,
  RetinaNet/SSD heads), and/or timm models (real code by other authors; needs
  `pip install timm`, free). sha256 + protocol in a new `data/generated/FROZEN_v2.json`.
- Builder rejects training programs whose variant graphs are isomorphic to a whole
  dev/holdout sample's, as now.

## What changes, file by file

| Area | Files | Change |
|---|---|---|
| Ground truth | new `graphart/hier.py` (uses `extract.py`) | tree, variants, wiring graphs, `inline()`; lossless check vs flat fx |
| Text format | `graphart/schema.py` (or `hier.py`) | writer + tolerant parser for tree and graphs |
| Validator | `graphart/validate.py` | children/variants resolve, DAG, arity |
| Scorer | `graphart/metrics.py`, `pipeline/score.py`, `scripts/classify_errors.py` | tree F1/exact, per-variant isomorphism, program right = all parts right, flat graph after inline (comparable with old numbers) |
| Generator | `graphart/generate_large.py`, `scripts/build_large_dataset.py` | as above; new output files `hier_*.jsonl` |
| Eval sets | `scripts/build_torchvision_set.py` (new mode) | new dev + holdout, `FROZEN_v2.json` |
| Prompts | `prompts/hier_tree*.txt`, `prompts/hier_wire*.txt` | brief + few-shot (train split only) |
| Training | `scripts/train_lora.py` (`--hier`), `scripts/make_kaggle_notebook.py` (`--hier-train`, `--hier-14b`) | two-task formatting; notebooks |
| Inference | new `pipeline/run_hier.py` | tree call → wiring calls → assemble → validate |
| Visuals | `graphart/render.py`, `interface/app.py`, generator viewer | nested drawing, blocks expandable to flat |
| Write-up | `docs/experiments.md`, README, report, deck | flat-schema results become the first chapter; new section after |

The edits (E2) and RQ6 sets are not rebuilt unless time is left.

## Phases

| # | When | Work | Done when |
|---|---|---|---|
| 1 | 20–21 Nov | `hier.py`: tree, variants, wiring, `inline()`, text writer/parser | inline == flat fx on all generator programs and the 16 full networks (failures listed, e.g. ViT's 12 blocks) |
| 2 | 21 Nov | New dev + holdout sets; **freeze holdout** (`FROZEN_v2.json`) | holdout hashed before phase 4 starts |
| 3 | 22 Nov | Validator, scorer, classifier; nested renderer | ground truth scores 1.0; hand-made broken answers score lower in the expected part |
| 4 | 22–23 Nov | Generator + dataset builder; check in the viewer against dev stats | tree size, nesting, variants per program and graph sizes cover the dev range |
| 5 | 24 Nov | Prompts, `train_lora --hier`, `run_hier.py`, notebooks; local smoke test on a tiny model; **write success criteria** | criteria written in `experiments.md` before any Kaggle run |
| 6 | 25–26 Nov | Kaggle (user runs): 1.5B training + dev eval (~2 h); 14B prompted on dev | results in `kaggle_results/` |
| 7 | 27 Nov | Freeze the setup; score the holdout once | one scored run per declared model |
| 8 | 28 Nov–1 Dec | Tables, figures, deck, README; user writes the prose | — |

## Phase 1 result (19 Nov)

`graphart/hier.py` + `scripts/check_hier.py`. Inline (directly, and after text → parse)
equals the flat fx graph for **all 837** programs: 800 generator, 21 dev, 16 full
torchvision networks (ViT, DenseNet and the `Conv2dNormActivation` families included).
Answer sizes in Qwen tokens, median / max:

| Set | Flat graph | New answer | of which tree | Graphs | Largest graph (nodes) |
|---|---|---|---|---|---|
| generator (800) | 1272 / 5286 | 520 / 2233 | 358 / 1955 | 2 / 5 | 11 / 25 |
| dev 21 | 604 / 2956 | 333 / 1190 | 150 / 1077 | 1 / 3 | 9 / 17 |
| full networks (16) | 7993 / 27119 | 1240 / 9205 | 1114 / 6638 | 3 / 7 | 8 / 27 |

The tree is now most of the answer. A compact parameter format (`Conv2d(64>128,k1)`) saves
only 1–20% (DenseNet 6638 → 5426), so it is not worth a second notation. The long trees
are DenseNet-121 and Inception-v3 (siblings differ in width, or are not numbered), so the
fallback for those is splitting the tree call per top-level child (decide in phase 4–5).

## Risks

- **Tree length.** Siblings with different widths don't collapse (EfficientNet-B0: 337
  tree lines, DenseNet-121: 433). Measure tokens in phase 1; if too long, put leaf
  parameters into a compact form or split the tree call per top-level stage.
- **Variant choice is config reasoning** — the `branch_on` failure (0/4 in E2). Expected
  to be the main error; the scorer reports it separately.
- **Opaque classes with a custom `forward` defined in the source but built by imported
  code**, and fx quirks (ViT's `MultiheadAttention`, `nn.Conv3d` subclasses): handled by
  phase 1's failure list, not by hand-patching samples.
- **Time.** If phase 6 slips, drop the 3B and second-adapter options first, then the
  nested renderer polish; never skip the holdout freeze or the written criteria.

## Decisions (user, 19 Nov)

1. Variable-tagged edges in wiring graphs: **no** — plain `← src` edges.
2. Holdout source: **both** unseen torchvision families and timm.
3. Tiers 1–4 share in training: **~15%**.
4. One two-task adapter vs two adapters: **one**, chosen for time (one training run).
   E1/E4 did not test this: they showed that composing two *separate* models (mixed-v2
   lists, a wirer wires) loses nothing. Whether one model does both tasks as well is
   untested; if it underperforms and time allows, train the second adapter.

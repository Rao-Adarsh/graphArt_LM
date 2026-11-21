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
- **Real code in training: timm** (decided 19 Nov). timm models are real code by other
  authors, CNN-heavy like torchvision. Needs `pip install timm` (free). timm models of the
  **holdout's architectures are excluded** from training (Swin, MaxViT, MViT and any other
  holdout family), so the holdout stays unseen at the architecture level, not only the
  code level. Generator programs remain the bulk: all of torchvision is 108 distinct
  block wirings (121 constructors, 38 files), 800 generator programs are 648 and fresh
  ones keep adding ~0.4 each.
- **Dev (new, replaces the 21):** whole torchvision networks plus blocks from the families
  already looked at (the 21's, the old holdout's, and the 16 full networks in the viewer),
  ~40–60 samples, sample = the model file's classes + one construction line.
- **Holdout (new, frozen before any generator change or training):** torchvision families
  never looked at — proposed Swin (v1/v2), MaxViT, MViT (v1/v2), Swin3D, FCN and LRASPP
  heads. sha256 + protocol in a new `data/generated/FROZEN_v2.json`. Hugging Face vision
  models (69 in `transformers`) are a possible second holdout source if they trace.
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

## Phase 2 result (19 Nov): eval sets built, test sets frozen

`scripts/build_hier_eval.py` → `data/generated/hier_{dev,holdout,timm_test}.jsonl` and
`FROZEN_v2.json` (sha256 of holdout and timm test; a rebuild with `--check` reproduces both).

| Set | Programs | What | Flat nodes median / max | Code chars median / max |
|---|---|---|---|---|
| dev | 51 | the 21 dev blocks + 30 whole torchvision networks of seen families | 71 / 548 | 5,441 / 18,599 |
| holdout (frozen) | 15 | Swin, Swin v2, MaxViT, MViT, Swin3D, FCN, LRASPP — every constructor | 475 / 1,902 | 19,729 / 26,377 |
| timm test (frozen) | 6 | DLA, CSPNet, GhostNet, two sizes each (no backup needed) | 565 / 905 | 27,135 / 36,696 |

All 72 are lossless. A network's code is assembled from the real files: every class the
model is built from in `torchvision.models` / `timm.models`, what those reference in the
same files, and the builder; docstrings dropped; `torchvision.ops` / `timm.layers` classes
are opaque boxes. timm training excludes the files `swin_transformer*`, `maxxvit`,
`mvitv2`, `dla`, `cspnet`, `ghostnet` (recorded in FROZEN_v2.json).

Two extractor fixes came out of building the sets, both general (re-checked on all 837
earlier programs, still lossless): buffers are traced as named reads (`@relative_position_index`)
instead of fx's anonymous `_tensor_constantN`, and the source assembler follows code
identifiers only (not words in strings, not infrastructure files).

**New risk — prompt length.** Real networks' code is long: holdout ~5–7k tokens, timm
~7–10k (chars/3.8). Together with long trees this exceeds what the 1.5B trained on before
(~6k tokens total on a T4). The per-class wiring task is unaffected (one class at a time);
the tree task is the problem. To decide in phase 4–5: give the tree task only `__init__`
methods and builders (drop `forward` bodies), and/or split the tree call per top-level child.

**Code reducer (20 Nov), `graphart/reduce.py`.** A static rewrite applied to every program
before the model reads it — in training, evaluation and general use: drops docstrings,
comments, type annotations (not in dataclasses), methods and functions nothing refers to,
and weight-init / API-logging calls; re-printed with `ast.unparse`. It never runs the code
or decides a branch. Checked: on 981 executable programs (generator, dev 21, tiers 1–4
sample) the traced truth of the reduced code is identical to the original's.

| Set | Code tokens saved | Tree-task tokens (code + tree), max before → after |
|---|---|---|
| generator | 3% (already lean) | — |
| dev networks | 31% | 10,824 → 9,912 (dev, all) |
| timm training files | 40% (max 26,052 → 10,899 code) | — |
| holdout | — | 9,988 → 8,381 |
| timm test | — | 16,320 → 12,072 |

The reducer must be declared as a pipeline step in the FROZEN_v2 amendment before scoring.
**Adopted (20 Nov):** every prompt is built from `reduce_source(code)` — trainer, runner
and interface (phase 5). The stored datasets keep the original code, so frozen hashes are
unchanged.

**Generator vs real code under the new schema (20 Nov, viewer), median / max:**

| | Reduced code tokens | Tree lines | Block graphs | Largest block | Nesting | Classes |
|---|---|---|---|---|---|---|
| generator tier 6 | 702 / 975 | 54 / 138 | 3 / 5 | 12 / 25 | 4 / 5 | 3 / 4 |
| torchvision networks (16) | 1,379 / 4,366 | 78 / 433 | 3 / 7 | 8 / 27 | 4 / 6 | 2.5 / 8 |
| timm (25, training-type) | 3,796 / 7,406 | 99 / 540 | 6 / 13 | 29 / 119 | 6 / 10 | 6 / 11 |

The generator's programs are 2–5× shorter, shallower and use fewer classes and block types
than real networks — the targets for the generator rework (phase 4).

`graphart/hier.py` is hashed in FROZEN_v2.json; any later change must still pass
`build_hier_eval.py --check`, or be recorded as a dated amendment.

## Phase 3/5: scorer, runner, evaluation notebook (21 Nov)

| Piece | File | What it does |
|---|---|---|
| Scorer | `graphart/hier_score.py` | per program: `parse`; tree as (path, description) multisets — `tree_exact`, `tree_struct` (no layer params), `tree_f1`; `variants` (same partition of block instances, up to renumbering); `wiring_right/total` per truth variant (graph isomorphism: node kind + target, edges with slot + selector); `flat_exact` / `flat_topo` = `inline(answer, ref=truth)` vs `inline(truth)` with `node_label` / `topology_label` — the headline `right`, comparable with the flat-schema numbers; `category` = first failing part (parse, tree, params, variants, wiring, assembly) |
| Scoring CLI | `pipeline/score_hier.py` | table per run, by kind and source, Wilson 95% CI on `right`; writes `results/<run>.score.json`; frozen splits need `--frozen` |
| Runner | `pipeline/run_hier.py` | tree call on reduced code → one wiring call per variant tag of the model's own tree (pre-order, `#0` = built but never run, skipped; no call for a class not in the code) → TREE + GRAPH answer; `--oracle-tree` = wiring alone on the truth's tree (no tree call); replies cached per call, results saved after every program; frozen splits need `--frozen`; caps `--tree-tokens 8192 --wire-tokens 2048` |
| HF backend | `pipeline/backends.py` | the GQA-expansion patch on GPU (prompts up to ~16k tokens) |
| Eval notebook | `make_kaggle_notebook.py --hier-eval` → `notebooks/kaggle_hier_eval.ipynb` | adapter as an uploaded input; GPU 0: `hier_dev`, then `hier_dev --oracle-tree`; GPU 1: `hier_gen_test`; time guard; scores in-notebook; `graphart_hier_eval.zip`. No frozen set in the bundle |

Checks: the true answer scores `right` on all 126 dev + gen_test programs; broken answers fail in
the part broken (a layer parameter → `params` 126/126, a swapped edge → `wiring` 104/105 (the
other swap was symmetric), a renamed tag → `variants` or `wiring`, garbage → `parse`). The truth
put through the runner's cache scores 75/75 on gen_test and 49/51 on dev, both modes; the 2
dev misses are the assembler defect below. Untuned qwen2.5-coder:1.5b (Ollama, 2 blocks):
replies are box-drawing trees, recorded as unreadable and scored `parse`.

**Assembler defect found (21 Nov, not fixed):** `build_hier_eval.assemble` follows references
only to functions and `nn.Module` classes, and never includes module-level assignments. Plain
classes (config dataclasses) and constants are missing from real-code samples. Static check
(names used, never defined; symtable): dev 30/51 programs, timm training 63/177, v3 0/1,350,
tiers 1–4 0/283. Many are harmless (`ResNet18_Weights`), some carry structure: VGG `cfgs`,
`MBConvConfig` (the only reference to `MBConv` — so the reducer drops the class and
`dev-efficientnet_b0` / `_v2_s` cannot be answered), `InvertedResidualConfig`, RegNet
`BlockParams`, timm `model_cfgs` (21). The frozen holdout and timm test were built by the
same assembler; a fix changes their hashes (a dated amendment before any scoring).

**Fixed and re-frozen (21 Nov, user's go-ahead) — FROZEN_v2.json amendment 1.** `assemble`
now follows plain classes (not Enums: weight enums stay out) and module-level constants of a
shown file, and aliases. Dev: undefined names left = weight enums only; 16 programs' code grew
(VGG +300 tokens, EfficientNet/RegNet +650–960); truth unchanged (51/51); every variant's
class now in the reduced code; truth through the runner 51/51. `build_hier_eval.py
--refreeze` rebuilt all three (counts only printed): 51 / 15 / 6 programs, all lossless;
holdout a3688e55c989 → 3f7b41236abf, timm_test 2520ea9cea2b → 68b1d4b60318; `--check`
reproduces both. hier.py unchanged since the first freeze. **Not rebuilt:** the timm training
rows (63/177 affected) — the 21 Nov adapter was trained on them; rebuild before any retraining.

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
2. timm is **training data**; the holdout is unseen torchvision families (changed from
   "both", 19 Nov).
3. Tiers 1–4 share in training: **~15%**.
4. One two-task adapter vs two adapters: **one**, chosen for time (one training run).
   E1/E4 did not test this: they showed that composing two *separate* models (mixed-v2
   lists, a wirer wires) loses nothing. Whether one model does both tasks as well is
   untested; if it underperforms and time allows, train the second adapter.

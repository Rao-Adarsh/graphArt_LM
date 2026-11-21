# Generator rework plan (generator v3) — proposal, 20 Nov 2026

*Not started. For the user's go-ahead. Part of phase 4 of `docs/hierarchical_plan.md`.*

## Principles

1. **Aim at real code in aggregate, never at single samples.** Targets come from the
   statistics of two pools only: the 30 dev networks and timm training-type models (the
   viewer's numbers below). Generator v2 was patched on the dev 21's individual errors
   and did not generalise (holdout 0/17); v3 does not look at per-sample errors at all.
2. **The frozen tests stay unseen.** No family, idiom or operation is added *because* a
   holdout or timm-test architecture has it. Excluded outright: shifted-window attention
   (Swin, MaxViT), pooling attention (MViT), segmentation decoders on a backbone (FCN,
   LRASPP), DLA aggregation trees, CSP stages, Ghost modules. What those networks share
   with others (e.g. attention, depthwise convs) is taught through the generic families.
3. **Acceptance criteria are written before the data is generated** (below), and checked
   on the aggregate pools, not the tests.
4. **New file, old one kept:** `graphart/generate_v3.py`; `generate_large.py` stays as it
   is so earlier results remain reproducible.

## Where generator v2 stands (viewer, 20 Nov)

Under the new schema, median / max:

| | Reduced code tokens | Tree lines | Block graphs | Largest block | Nesting | Classes |
|---|---|---|---|---|---|---|
| generator tier 6 | 702 / 975 | 54 / 138 | 3 / 5 | 12 / 25 | 4 / 5 | 3 / 4 |
| torchvision networks | 1,379 / 4,366 | 78 / 433 | 3 / 7 | 8 / 27 | 4 / 6 | 2.5 / 8 |
| timm (training-type) | 3,796 / 7,406 | 99 / 540 | 6 / 13 | 29 / 119 | 6 / 10 | 6 / 11 |

Operations real networks use (share of torchvision + timm networks / of generator programs):
`nn.Identity` 59%/0%, `.permute()` 44%/0%, `nn.Flatten` 41%/0%, `.reshape()` 39%/0%,
`F.layer_norm` 29%/0%, `matmul` 24%/0%, `.softmax()` 24%/0%, `.unbind()` 22%/0%,
`F.scaled_dot_product_attention` 17%/0%, `get_attr` (parameters read in forward) 63%/8%,
`mul` 54%/5%, `getattr` 54%/9%, `torch._assert` 41%/9%, `nn.GELU` 34%/10%, `.mean()` 37%/10%,
`nn.Dropout` 76%/38%, `nn.AdaptiveAvgPool2d` 66%/28%.

Idioms (real networks / generator tier 6 / dev 21): `**kwargs` 100/9/38%, `Optional`/`None`
defaults 100/27/67%, top-level helper functions 100/27/29%, loops in `__init__` 98/52/14%,
`functools.partial` 59/0/5%, `assert`/`torch._assert` 63/15/5%, `OrderedDict`/`add_module`
29/0/0%, `nn.ModuleList` 32/16/0%, injected layer classes (`Callable[...]`) 27/8/67%,
layers imported without source 15/2/10%.

## What changes

**A. Whole networks first.** Every test set is now mostly whole networks, so ~70% of
programs are networks (stem → 3–5 stages of 2–6 blocks → head), ~30% single blocks.

**B. Size and depth targets** (medians, with tails up to the real maxima): reduced code
1.5–3k tokens (max ~8k), tree 60–200 lines, 3–8 block graphs, largest block up to ~40
nodes, nesting 4–7, 3–8 classes. Answers capped at ~7k tokens, tree-task examples at the
length the memory probe allows (~12k expected).

**C. Families.** Existing seven (ResNet, Inception, inverted residual, Fire, dense,
ShuffleNet, transformer encoder) are kept and grown to network size. New, generic ones:

| Family | What it teaches |
|---|---|
| ConvNeXt-style block | depthwise 7×7, LayerNorm on NHWC via `permute` or a LayerNorm2d class, MLP, layer scale (`gamma * x`, get_attr), drop path |
| EfficientNet-style MBConv stages | expand → depthwise → squeeze-excite (`mean`, `mul`) → project, stochastic depth, stage config tables, `_make_divisible` |
| RegNet-style | grouped convs, SE, width/depth computed by a helper function |
| VGG-style plain | config list (`[64, "M", 128, …]`) turned into layers by a loop |
| ViT-style | patch embedding conv → `flatten(2).transpose(1, 2)`, class token + position embedding (get_attr, `expand`, `cat`), attention written out (`qkv.reshape.permute.unbind`, `matmul`, `softmax`) or via `F.scaled_dot_product_attention` |
| MLP-Mixer-style | token mixing through `transpose`, channel MLPs |

**D. Idioms raised to real shares:** `**kwargs`, `Optional[...] = None` then defaults,
`partial(nn.BatchNorm2d, eps=...)`, injected `norm_layer` / `act_layer` / `block` arguments,
`OrderedDict` / `add_module(f"denselayer{i}")` (named, non-numbered children),
`nn.ModuleList` / `nn.ModuleDict` loops, `forward_features` / `forward_head` split,
`torch._assert` shape checks, `nn.Identity()` placeholders (`... if p > 0 else nn.Identity()`),
`nn.Flatten`, `self.training` / `torch.jit.is_scripting()` branches, helper functions at
top level.

**E. Imported layers with no source (~25% of programs):** `from torchvision.ops import
Conv2dNormActivation, SqueezeExcitation, StochasticDepth, MLP, Permute` and timm.layers
equivalents. They become opaque boxes in the truth — the 13/17 failure of the old holdout.

**F. Block variants:** in ~60% of networks one class appears in 2–3 variants (first block
of a stage downsamples; SE switched by config; projection vs identity shortcut), so the
tree's `#k` tags and the config reasoning behind them are trained.

**G. Config-driven branches:** constructor flags that switch submodules on and off, True
and False about equally often — the `branch_on` skill (0/4 in E2).

## Dataset

~1,500 v3 programs (70% networks) + timm training models (~210: 71 traceable files × ~3
sizes) + ~15% tiers 1–4. Every program passes the lossless check; duplicates removed by
answer (tree + isomorphic block graphs); anything isomorphic to a dev, holdout or timm-test
program rejected; length caps from the memory probe.

## Acceptance criteria (fixed before generating)

Measured on v3 vs the pooled dev networks + timm training-type models:
1. Medians of reduced code tokens, tree lines, block graphs, largest block, nesting and
   classes each fall inside the real pool's interquartile range.
2. Every operation used by ≥ 15% of real networks appears in ≥ 10% of v3 programs (except
   those only the excluded families use).
3. Every idiom above reaches at least half its real share.
4. 100% lossless; 0 programs isomorphic to a frozen test program.

If a criterion fails, the generator is changed and the whole check re-run — still without
looking at the tests.

## Effort

About 1–1.5 days of implementation (families and idioms, then the builder), ~1 h to build
the dataset locally. No GPU needed.

## Acceptance check, round 1 (20 Nov night) — `scripts/check_v3_acceptance.py`

Built: `graphart/generate_v3.py` (11 families: ResNet, MBConv, Fire, dense, ShuffleNet,
Inception, ConvNeXt, RegNet, ViT, VGG, MLP-Mixer; ViT drawn 4x and Mixer 2x as often) and
`scripts/build_hier_dataset.py` → `hier_train` 1,810 programs (1,350 v3, 177 timm from 90
files, 283 tiers 1–4), `hier_val` 99, `hier_gen_test` 75 (v3). All lossless; 125 v3
duplicates and 2 v3 programs matching an eval-set graph were dropped; 51 timm models did not
trace, 8 were not lossless. Training examples: 1,810 tree + 6,399 wiring, 9.0M tokens
(mean 1,104, max 15,619).

Real pool = 30 dev networks + 186 timm networks (timm dominates it).

| Criterion | Result |
|---|---|
| 1. size | **FAIL** on 2 of 6: reduced code 1,031 tokens vs real IQR 2,301–3,885; tree lines 50 vs 55–158. Pass: block graphs 4 (3–8), largest block 14 (11–28), nesting 4 (4–7), classes 4 (4–7) |
| 2. operations | PASS (counting only what the answer contains — the check first counted ops inside opaque layers, a measurement bug fixed before this round) |
| 3. idioms | **FAIL** on 3 of 23, all near misses: ModuleList 12% (needs 14%), OrderedDict/add_module 14% (needs 22%), `is None` checks 46% (needs 50%) |
| 4. lossless, unseen | PASS |

Against the 30 torchvision dev networks alone (round 0): code 934 vs IQR 1,211–1,766 was the
only size failure.

## Acceptance check, round 2 (21 Nov) — `results/raw/v3_acceptance_round2.txt`

Changes (aimed at the aggregate gaps above; no test program looked at):

| Change | Real-code pattern it copies |
|---|---|
| 45% of CNN networks mix two block families (stages from table A, then table B, `enumerate(cfg_b, start=len(cfg_a))`) | MobileNet / EfficientNetV2 / timm byobnet |
| ResNet: both block classes in the file, the network takes `block` (60%); downsample built by `downsample_conv` / `downsample_avg` chosen by `down_type` (50%) | torchvision `ResNet(block, layers)`, timm resnet |
| two-level builder `_net(variant, weights=None, progress=True)` with a `weights is not None` check (half the builders) | torchvision `_resnet(...)`, timm `_create_x` |
| `self.feature_info.append(dict(num_chs=..., module=...))`, `self.num_features` (50%) | timm |
| `if self.grad_checkpointing and not torch.jit.is_scripting(): checkpoint_sequential(...) else: ...` (40%) | timm |
| staged transformer (half the ViTs): global-attention blocks in stages of growing width, strided conv between stages, tokens ↔ NCHW by shape reads; no windows, patch merging or pooled attention | timm pvt / twins / nextvit-style stages |
| transformer blocks in `nn.ModuleList` looped in forward (45%); stems and heads as `nn.Sequential(OrderedDict([...]))`; ViT heads in an OrderedDict with an optional pre-logits layer (`representation_size is None`) | torchvision ViT, timm |
| `if x is None: x = default` instead of `x = x or default` (70%) | torchvision |

Rebuilt: `hier_train` 1,810 programs (v3 1,350, timm 177, tiers 1–4 283), val 99, gen_test 75;
108 v3 duplicates and 1 eval-set match dropped; 0 trace failures in a 180-program test across
all families. Training examples 1,810 tree + 6,480 wiring, mean 1,161 tokens (max 15,619):
9.6M tokens. Round 1 data kept in `results/raw/hier_build_cache/round1/`.

| Criterion | Round 1 | Round 2 |
|---|---|---|
| 1. size: reduced code tokens (real IQR 2,301–3,885; torchvision alone 1,211–1,766) | 1,031 FAIL | 1,155 FAIL |
| 1. size: tree lines (real IQR 55–158; torchvision alone 61–131) | 50 FAIL | 69 PASS |
| 1. size: block graphs / largest block / nesting / classes | PASS | PASS (4 / 14 / 4 / 4) |
| 2. operations | PASS | PASS |
| 3. idioms: ModuleList (real 28%) | 12% FAIL | 28% PASS |
| 3. idioms: OrderedDict / add_module (real 44%) | 14% FAIL | 32% PASS |
| 3. idioms: `is None` checks (real 100%) | 46% FAIL | 72% PASS |
| 4. lossless, unseen | PASS | PASS |

Where code length stands, by part (median characters of reduced code, networks): timm 11,518
(class `__init__`s 6,337, forwards 1,648, top-level functions 1,511, 5 classes, 3 functions);
v3 round 1 3,727 (2,069 / 896 / 213, 4 classes, 1 function). The rest of the gap is timm's
option plumbing (dilation, drop_block, `device`/`dtype` passing, attention-layer factories,
sibling classes chosen by strings), most of it calling `timm.layers` code that is not shown.

Round 2 accepted by the user (21 Nov).

## Training notebook (21 Nov) — `notebooks/kaggle_hier_train.ipynb`

`scripts/make_kaggle_notebook.py --hier-train`. Fresh LoRA (r 16) on Qwen2.5-Coder-1.5B-Instruct,
`train_lora.py --hier --max-len 20480 --drop-over-max --eval-steps 100`, one epoch, both T4s
(torchrun, 8 + 8 of each 16-example step), cap 2.5 h. Steps: data check → memory check on the
6 longest examples (fallback `--max-len 12288`) → 8-step smoke run → train → zip
(`graphart_hier_train.zip`: adapter, `train_summary.json`, logs). No scoring in it. The bundle
carries only code, prompts, `hier_train` and `hier_val` (id, source, family, kind, code, hier);
notebook 0.91 MB. Checked locally from the unpacked bundle: the data check (8,290 / 452
examples) and a 3-step CPU training run (model load, attention patch, loss, backward).

## Compressed schedule: training finished by 22 Nov afternoon (agreed 20 Nov)

Scope cut to what moves the measured gaps most; the rest is deferred.

| Keep | Defer |
|---|---|
| the 7 families grown to whole networks, with variants and config flags | RegNet-, VGG- and MLP-Mixer-style families |
| the idiom layer across all families (kwargs, partial, injected layers, OrderedDict, Identity, Flatten, asserts, forward_features/head) | a second acceptance iteration beyond one round of fixes |
| imported layers without source (torchvision.ops) | the hierarchical scorer and runner (built during the training run) |
| ViT-style (attention written out) and ConvNeXt/MBConv-style families — they carry most missing operations | |

| When | Who | What |
|---|---|---|
| 20 Nov | user | run `kaggle_memory_probe.ipynb` (15 min) |
| 20 Nov | Claude | v3 core: families at network size, idiom layer, opaque imports → **checkpoint 1**: 30-example viewer page |
| 21 Nov morning | Claude | the two new families, dataset builder, timm training data, acceptance check → **checkpoint 2**: criteria table |
| 21 Nov afternoon | Claude | prompts (tree + wiring), `train_lora.py --hier` (reducer applied), `--hier-train` notebook with memory check and time guard, local smoke test |
| 21 Nov evening/night | user | start the training run (unattended Kaggle commit, ≤ 2.5 h, both T4s) |
| 22 Nov morning | both | one retry if needed; Claude builds the scorer, runner and evaluation notebook meanwhile |
| 22 Nov afternoon | — | **training done**; evaluation follows in its own notebook |

Run size from the probe: if ≥ 12k tokens fit on one T4, train on both T4s in parallel with
~5–6M tokens (~1.5–2 h); if only shorter fits, cap examples at that length (the longest timm
examples drop out); if only the split model fits, ~3–4M tokens. Fallback if v3 slips: train
on the 800 v2 programs (already converted and lossless) + timm + whatever v3 is finished.

## Decisions for the user

1. Exclude segmentation decoders (FCN/LRASPP-like) from the generator so the holdout's two
   segmentation families stay unseen — at the cost that `F.interpolate` upsampling is never
   trained. **Default: exclude.**
2. 70% networks / 30% blocks. **Default: yes.**
3. ~1,500 v3 programs. **Default: yes** (≈ 5–6M tokens with timm; re-sized after the probe).

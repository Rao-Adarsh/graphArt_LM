# graphArt_LM — context for a new session

IE 643 course project, team Pensieve. **Question: can an LLM read PyTorch `nn.Module`
source and write the forward-pass graph that `torch.fx` would trace — without running
the code?** Ground truth is torch.fx + ShapeProp; scoring is labelled-graph isomorphism.

Read next, as needed: `docs/experiments.md` (every training/prompting variant, results,
the E1–E4 experiments, the freeze and the holdout result — the most current document),
`README.md` (dataset, evaluation, quickstart, interface), `docs/research_story.md` (how
the research questions came about), `docs/code_walkthrough.md`. `docs/handoff.md` is
historical (late Oct).

## Standing rules (the user's; do not relax them)

- **No money without an estimate and an explicit yes.** Free models and local Ollama are
  fine, but say so before running them. The gate is in code (`pipeline/cost.py`, `--yes`).
- **Never push to GitHub without asking.** Commit only when asked.
- **Kaggle GPU time is scarce**: one unattended run per notebook, capped at ~2–2.5 h,
  kept simple. **The user runs the notebooks**; results come back as zips in `kaggle_results/`.
- **Report rule:** AI-written prose or AI summaries of papers score zero. Supply tables,
  figures, numbers and outlines; the user writes the prose.
- `docs/Notes.md` is the user's file — never touch it.
- Never propose having the model call `torch.fx` itself (it is the answer key).
- Few-shot examples come from the **train** split only.
- **The torchvision holdout (`data/generated/torchvision_holdout.jsonl`, 17 samples) has
  been scored once (15 Nov) and is spent.** Never tune, re-prompt or re-run on it; never
  generate holdout replies locally. Protocol and its dated amendment:
  `data/generated/FROZEN.json`. The 21-sample `torchvision` set is development data.
- Refer to the user with they/them.

## Where things stand (19 Nov 2026)

| | Torchvision dev 21 | Holdout 17 |
|---|---|---|
| 26 Oct submission (fine-tuned 1.5B, one step) | 0 | 0 |
| Qwen2.5-Coder 14B zero-shot | 4 | not run |
| mixed-v2 (1.5B, one step, generator v2) | 7 | 0 |
| **final: mixed-v2's node list + 1.5B wirer** | **11** | **0** |
| mixed-v2's node list + 14B wirer (E1) | 13 | not run |

("right graph" = exact, notation, spelling or commutative; `scripts/classify_errors.py`.)
In-distribution (our generator's test 40) the adapters are 40/40.

What holds: the decomposition (node listing vs wiring fail separately), E2 (variable-tagged
two-step output follows code edits: unedited graph 14/69 vs 39/69, p≈4e-7), lossless
composition (E1, E4). What failed: E3 / order checks (wirers lean on node order), and the
holdout — 13/17 holdout programs import torchvision layers (`Conv2dNormActivation`,
`SqueezeExcitation`, …) whose source is not in the prompt; others hinge on configuration
flags or fx's handling of `nn.Conv3d` subclasses. The dev-set gain was partly fitted to
the 21 (the generator was patched on their errors after Session 1).

Remaining: commit the work since 10 Nov (asked for, not done), update report tables/figures,
the deck and the README for the final write-up. The submission date is unconfirmed
(assumed ~1 Dec).

## Environment

- Windows, no NVIDIA GPU. Use the venv: `.venv/Scripts/python.exe` (system Python has no
  torch). Shell: PowerShell or Git Bash.
- Ollama locally for small models; Kaggle (2× T4) for anything real.
- Adapters live outside git: `models/` (26 Oct adapter), `kaggle_results/<run>/lora/…`,
  upload zips in `dist/`. `results/raw/` (raw replies) is gitignored too.

## Layout and commands

- `graphart/` — domain: generator (`generate.py`, `generate_large.py`), fx extraction
  (`extract.py`), schema/text format (`schema.py`), metrics, validator, renderer
  (`render.py`), two-step format (`two_step.py`).
- `pipeline/` — `run_prompting.py` (one-step; `--nodes-from` for composition),
  `run_two_step.py` (`--brief`, `--step1-from`), `score.py`, `backends.py`, `cost.py`.
- `scripts/` — `train_lora.py` (`--two-step`, `--full-prompts`, `--wirer`),
  `make_kaggle_notebook.py` (one flag per notebook, e.g. `--e4-wirer`, `--holdout`; edit the
  script, never the generated .ipynb), `classify_errors.py`, `decompose.py`, builders.
- `results/<backend>__<model>__<prompt>[__<split>].jsonl` (+ `.meta.json`): scored with
  `python -m pipeline.score <file>`, `scripts/classify_errors.py <file>`,
  `scripts/decompose.py <files> --per-sample`.
- `interface/app.py` — Gradio viewer (`.claude/launch.json` "interface", port 7860):
  model graph vs fx graph for any saved answer, including the holdout.
- `report/` (LaTeX, last built 26 Oct), `docs/presentation/` (deck).

Comparisons use exact McNemar on discordant samples and Wilson 95% intervals; criteria are
written down before a run (see `docs/experiments.md`).

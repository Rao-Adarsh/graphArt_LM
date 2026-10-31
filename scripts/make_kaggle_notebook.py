"""Build the Kaggle notebook for Phase 1 prompting on a free GPU.

    python scripts/make_kaggle_notebook.py
    python scripts/make_kaggle_notebook.py --prompts cos --models qwen2.5-coder:14b

Writes two equivalent forms of the same code:

  notebooks/kaggle_ollama.ipynb    -- import it with File > Import Notebook
  notebooks/kaggle_bootstrap.txt   -- two one-line cells to paste instead, for when
                                      uploading a file is not an option (Kaggle's
                                      editor auto-indents typed multi-line code)

The laptop has no usable GPU, and OpenRouter's free tier allows 50 requests a
day. A Kaggle notebook gets two T4s for free, so it runs Ollama there and
drives it with the *same* run_prompting.py -- inference venue changes, scoring
path does not.

The notebook is self-contained: the runner, the prompts and the stratified
test samples travel inside it as a compressed bundle, so nothing has to be
pushed or uploaded first. Only inference happens on Kaggle. The replies come
back as results/raw/<run>/<id>.txt, exactly the layout the runner caches to,
so locally

    python -m pipeline.run_prompting --backend ollama --model <m> --prompt <p> --stratify 10 --cached-only
    python -m pipeline.score results/ollama__<m>__<p>.jsonl

rebuild and score them with no special-casing. Re-run this script whenever a
prompt changes, or the notebook will send the old one.
"""

import argparse
import base64
import hashlib
import io
import json
import lzma
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.run_prompting import load_split

OUT_DIR = ROOT / "notebooks"
STRATIFY = 10
CODE_FILES = ["pipeline/__init__.py", "pipeline/backends.py", "pipeline/cost.py",
              "pipeline/run_prompting.py"]
SAMPLE_FIELDS = ("id", "tier", "num_nodes", "code", "target")
# The closing evaluation sets (RQ4, RQ5), built by build_torchvision_set.py and
# build_edit_set.py; bundled whenever they exist. torchvision_holdout is frozen (see
# data/generated/FROZEN.json): score it once, on the final model, with an 8000 cap.
EVAL_SETS = ("torchvision", "edits", "rq6", "large_test", "torchvision_holdout")
# The library-style training set (build_large_dataset.py), bundled with train/val for
# fine-tuning whenever it exists; select it with TRAIN_ARGS --train-split/--val-split.
LARGE_TRAIN_SETS = ("large_train", "large_val")


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime = len(data), 0          # mtime 0: same inputs, same bundle
    tar.addfile(info, io.BytesIO(data))


def payload(finetune: bool = False) -> tuple[dict[str, bytes], int]:
    """Every file the runner needs on Kaggle, by repo-relative path, and the sample count.

    For fine-tuning it also carries the trainer and the full train and val splits
    (with their edges -- they are the training targets). The test samples stay
    stripped either way.
    """
    # The whole test split, in split order, so `--stratify 10` still picks exactly the
    # 40 every other run used -- plus the closing evaluation sets that exist. The
    # reference edges are dropped from all of them: nothing on Kaggle reads them
    # (edges_only is handed the nodes, not the edges), and leaving them out keeps the
    # answers off the box.
    files = {rel: (ROOT / rel).read_bytes() for rel in CODE_FILES}
    for prompt in sorted((ROOT / "prompts").glob("*.txt")):
        files[f"prompts/{prompt.name}"] = prompt.read_bytes()
    for name in ("test", *EVAL_SETS):
        if not (ROOT / "data" / "generated" / f"{name}.jsonl").exists():
            continue
        rows_ = load_split(name, None, None, None)
        files[f"data/generated/{name}.jsonl"] = "".join(
            json.dumps({**{k: r[k] for k in SAMPLE_FIELDS},
                        "target": r["target"].split("EDGES")[0].strip()}) + "\n" for r in rows_).encode("utf-8")
    rows = load_split("test", None, None, STRATIFY)
    if finetune:
        files["scripts/train_lora.py"] = (ROOT / "scripts" / "train_lora.py").read_bytes()
        for split in ("train", "val", *LARGE_TRAIN_SETS):
            if not (ROOT / "data" / "generated" / f"{split}.jsonl").exists():
                continue
            full = load_split(split, None, None, None)
            files[f"data/generated/{split}.jsonl"] = "".join(
                json.dumps({k: r[k] for k in SAMPLE_FIELDS}) + "\n" for r in full).encode("utf-8")
    return files, len(rows)


def _tar(files: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in files.items():
            _add(tar, name, data)
    return raw.getvalue()


def bundle(files: dict[str, bytes]) -> tuple[str, str]:
    """Base64 tar.xz of the files, for the .ipynb's unpack cell, and its sha256."""
    # xz, not gzip: the training splits are repetitive JSON, and gzip left the
    # notebook over Kaggle's 1 MB limit on kernel source.
    data = lzma.compress(_tar(files), preset=9 | lzma.PRESET_EXTREME)
    return base64.b64encode(data).decode("ascii"), hashlib.sha256(data).hexdigest()


def _chunks(b64: str, width: int = 100) -> str:
    return "\n".join(f'    "{b64[i:i + width]}"' for i in range(0, len(b64), width))


DEFAULT_MODELS = ["qwen2.5-coder:14b"]
# Highest value first, so a session that runs out of time loses the least.
DEFAULT_PROMPTS = ["zero_shot", "nodes_only", "edges_only", "cot", "cos", "few_shot"]


# cot and cos write their working before the graph, and torchvision's whole networks
# need ~3000-token answers; the runner's default cap (1500 tokens) would cut them off.
LONG_PROMPTS, LONG_SETS = ("cot", "cos"), ("torchvision", "torchvision_holdout")


def job(model: str, prompt: str, split: str = "test", stratify: int | None = STRATIFY,
        max_tokens: int | None = None) -> dict:
    long = prompt in LONG_PROMPTS or split in LONG_SETS
    return {"model": model, "prompt": prompt, "split": split, "stratify": stratify,
            "max_tokens": max_tokens or (4096 if long else None)}


def config(jobs: list[dict], hours: float = 11.0) -> str:
    """The settings cell, kept on one line so it can be typed into the editor as is."""
    return (f"JOBS = {jobs!r}; MODELS = list(dict.fromkeys(j['model'] for j in JOBS)); "
            f"TIME_BUDGET_HOURS = {hours}; SMOKE_ONLY = False; VALIDATE_ONLY = False")

CONFIG_NOTES = """
# JOBS         run in order, highest value first, so a session that runs out of time loses
#              the least. Each is one model x prompt x data set: split is test, torchvision
#              or edits; stratify 10 takes the same 40 test programs as every Phase 1 run
#              (None = the whole set); max_tokens None keeps the runner's 1500-token cap.
# MODELS       pulled in order. qwen2.5-coder:14b fits one T4; gemma4:31b needs both.
# TIME_BUDGET  stop starting new work with an hour of the 12 to spare.
# SMOKE_ONLY   True: install, download, try one sample, and stop.
# VALIDATE_ONLY True: only check the code arrived intact. Run it on a CPU session
#              (Accelerator: None), which costs no GPU quota.
"""


def code_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    """(heading, source) for every step after the settings. Plain Python only --
    no `!` shell lines -- so the same text also runs as one exec'd script."""
    return [
        ("1. Unpack the code, prompts and samples", f"""
import base64, hashlib, io, json, os, shutil, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

START = time.time()
WORK = Path(os.environ.get("GRAPHART_WORK", "/kaggle/working"))
REPO = WORK / "graphArt_LM"
BUNDLE_SHA256 = "{digest}"
BUNDLE = (
{_chunks(b64)}
)
data = base64.b64decode(BUNDLE)
assert hashlib.sha256(data).hexdigest() == BUNDLE_SHA256, "bundle damaged -- regenerate the notebook"
REPO.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tar:
    tar.extractall(REPO)
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print(sorted(str(p.relative_to(REPO)) for p in REPO.rglob("*") if p.is_file()))
if VALIDATE_ONLY:
    raise SystemExit("bundle intact -- VALIDATE_ONLY is set, so stopping here")
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
"""),
        ("2. Install and start Ollama", """
def ollama_up() -> bool:
    try:
        print(urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2).read().decode())
        return True
    except OSError:
        return False

if not shutil.which("ollama"):
    subprocess.run("apt-get -qq update > /dev/null && apt-get -qq install -y zstd pciutils lshw > /dev/null",
                   shell=True, check=True)
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

if not ollama_up():
    # Ollama splits a model over both T4s by itself when it does not fit on one.
    env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_KEEP_ALIVE="24h")
    subprocess.Popen(["ollama", "serve"], env=env,
                     stdout=open(WORK / "ollama.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(1)
        if ollama_up():
            break
    else:
        raise RuntimeError(f"ollama did not start -- see {WORK / 'ollama.log'}")

for model in MODELS:
    subprocess.run(["ollama", "pull", model], check=True)
subprocess.run(["ollama", "list"])
"""),
        ("3. Smoke test: one sample, before committing hours to it", """
# Shows the reply, how long it took and how long it is, so a model that thinks out loud
# or ignores the format is obvious before the full run.
from pipeline.backends import make_backend
from pipeline.run_prompting import load_prompt, load_split, _render

row = load_split("test", None, None, 10)[0]
system, template = load_prompt("zero_shot")
for model in MODELS:
    t = time.time()
    reply = make_backend("ollama", model)(system, _render(template, row))
    took = time.time() - t
    print(f"=== {model}: {took:.0f}s, {len(reply)} chars\\n{reply[:1500]}")
    # GPU hours are scarce: a model that does not answer in the format would burn the
    # whole session producing nothing scorable, so stop here instead.
    if "NODES" not in reply or "EDGES" not in reply:
        raise SystemExit(f"{model}: the smoke-test reply is not a graph -- stopping before the full run")
    # Longer answers (cot, cos, torchvision) take several times this; a lower bound.
    print(f"{model}: {took:.0f} s for one zero_shot test answer")
"""),
        ("4. The run", """
def remaining() -> float:
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def package() -> None:
    # Re-zipped after every prompt, so a session that dies late keeps what it finished.
    if (REPO / "results").exists():
        shutil.make_archive(str(WORK / "graphart_results"), "zip", REPO, "results")

for j in ([] if SMOKE_ONLY else JOBS):
    name = f"{j['model']} / {j['prompt']} / {j['split']}" + (f" ({j['stratify']} per tier)" if j["stratify"] else "")
    if remaining() < 600:
        print(f"time budget spent -- skipping {name}")
        continue
    cmd = [sys.executable, "-m", "pipeline.run_prompting", "--backend", "ollama",
           "--model", j["model"], "--prompt", j["prompt"], "--split", j["split"]]
    if j["stratify"]:
        cmd += ["--stratify", str(j["stratify"])]
    if j["max_tokens"]:
        cmd += ["--max-tokens", str(j["max_tokens"])]
    if j.get("limit"):                      # for trying a notebook on a few samples
        cmd += ["--limit", str(j["limit"])]
    print(f"\\n##### {name}   ({remaining() / 3600:.1f} h left)", flush=True)
    try:
        # Cached replies survive a timeout; a later session would only redo the rest.
        subprocess.run(cmd, check=False, timeout=remaining(), env=dict(os.environ, PYTHONUNBUFFERED="1"))
    except subprocess.TimeoutExpired:
        print("stopped by the time budget")
    package()

package()
print(f"\\ndone in {(time.time() - START) / 3600:.1f} h")
raw_dir = REPO / "results" / "raw"
for run in sorted(raw_dir.iterdir()) if raw_dir.exists() else []:
    print(f"{run.name:55s} {len(list(run.glob('*.txt'))):3d} replies")
for meta in sorted((REPO / "results").glob("*.meta.json")):
    m = json.loads(meta.read_text())
    print(f"{meta.stem:55s} n={m['n']} failed={m['failed']}")
if not SMOKE_ONLY:
    print(f"download {WORK / 'graphart_results.zip'} from the Output tab")
"""),
    ]


INTRO = """
# graphArt_LM: Phase 1 prompting on a free Kaggle GPU

Runs the six prompts over the same {n} stratified test samples as every other run, on a
local open model served by Ollama. Only inference happens here; scoring happens back on
the laptop with the unchanged `pipeline.score`.

**Before running**, in the right-hand panel under *Session options*:
- **Accelerator:** GPU T4 x2
- **Internet:** on (needed to install Ollama and download the model)

**To run unattended**, use *Save Version → Save & Run All (Commit)*. It runs in the
background with the browser closed, for up to 12 hours, and keeps everything written to
`/kaggle/working`.

**When it finishes**, download `graphart_results.zip` from the version's *Output* tab and
unzip it into the repo root. It contains `results/raw/ollama__<model>__<prompt>/`. Then
locally, for each model and prompt:

```
python -m pipeline.run_prompting --backend ollama --model <model> --prompt <prompt> --stratify 10 --cached-only
python -m pipeline.score results/ollama__<model>__<prompt>.jsonl
```

Generated by `scripts/make_kaggle_notebook.py`. Edit that script, not this file.
"""


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook(settings: str, cells: list[tuple[str, str]], n: int,
             intro: str = INTRO, notes: str = CONFIG_NOTES) -> dict:
    out = [_md(intro.format(n=n)), _code(notes.strip("\n") + "\n" + settings)]
    for heading, source in cells:
        out += [_md(f"## {heading}"), _code(source)]
    return {"cells": out, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


CHUNK = 3000          # characters per typed cell: short enough to retype one cheaply


def bootstrap(settings: str, files: dict[str, bytes], steps: list[tuple[str, str]]) -> list[str]:
    """The same notebook as one-line cells: settings, base64 chunks, and a loader.

    Every cell is a single line because the editor auto-indents typed newlines.
    The files and the steps after unpacking travel as one lzma-compressed tar, the
    smallest form found (~22 KB). Each chunk carries its own hash, so a chunk that
    was mistyped is named, not just detected.
    """
    prelude = ("import json, os, shutil, subprocess, sys, time, urllib.request\n"
               "from pathlib import Path\n")
    script = prelude + "\n".join(source.strip("\n") + "\n" for _, source in steps)
    blob = base64.b64encode(lzma.compress(_tar({**files, "kaggle_run.py": script.encode("utf-8")}),
                                          preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    chunks = [blob[i:i + CHUNK] for i in range(0, len(blob), CHUNK)]
    names = [f"B{i + 1}" for i in range(len(chunks))]
    hashes = [hashlib.sha256(c.encode("ascii")).hexdigest()[:12] for c in chunks]
    loader = (
        "import base64, hashlib, io, lzma, os, sys, tarfile, time; from pathlib import Path; "
        f"_C = [{', '.join(names)}]; _H = {hashes!r}; "
        "_bad = [f'B{i + 1}' for i, (c, h) in enumerate(zip(_C, _H)) "
        "if hashlib.sha256(c.strip().encode()).hexdigest()[:12] != h]; "
        "assert not _bad, f'retype {_bad} -- the rest arrived intact'; "
        "START = time.time(); WORK = Path(os.environ.get('GRAPHART_WORK', '/kaggle/working')); "
        "REPO = WORK / 'graphArt_LM'; REPO.mkdir(parents=True, exist_ok=True); "
        "tarfile.open(fileobj=io.BytesIO(lzma.decompress(base64.b64decode(''.join(c.strip() for c in _C))))).extractall(REPO); "
        "os.chdir(REPO); sys.path.insert(0, str(REPO)); "
        "print(f'all {len(_C)} chunks intact, unpacked to {REPO}'); "
        "None if VALIDATE_ONLY else exec(open(REPO / 'kaggle_run.py', encoding='utf-8').read())")
    return [settings] + [f'{n} = "{c}"' for n, c in zip(names, chunks)] + [loader]


# ---------------------------------------------------------------------------
# Phase 2: the fine-tuning notebook. Same bundle and unpack step; then train a
# LoRA adapter with scripts/train_lora.py and score it, and its untuned base,
# through the same runner as every prompted model.

# Versions the trainer was tested with locally. Kaggle's own torch is kept.
FINETUNE_PINS = {"transformers": "5.20.2", "peft": "0.21.2", "accelerate": "1.15.0"}

FINETUNE_CONFIG = ('BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; '
                   'ADAPTER = "qwen2.5-coder-1.5b-graphart-lora"; EPOCHS = 2.0; SMOKE_STEPS = 20; '
                   'TIME_BUDGET_HOURS = 11.0; TRAIN_ARGS = []; EVAL_ARGS = []; VALIDATE_ONLY = False')

FINETUNE_NOTES = """
# BASE_MODEL   the Hugging Face model to tune -- the same weights as Ollama's
#              qwen2.5-coder:1.5b, unquantised.
# ADAPTER      the adapter's name; its results land in results/raw/hf__<ADAPTER>__zero_shot/.
# EPOCHS       passes over the 1280 train samples. The val split picks the best checkpoint.
# SMOKE_STEPS  a short first run that must show the loss falling before the real one starts.
# TIME_BUDGET  the whole session; training stops early enough to leave an hour for scoring.
# TRAIN_ARGS / EVAL_ARGS  extra flags for train_lora.py / run_prompting.py (normally empty).
#              To add the library-style programs (tiers 5-6, answers up to ~6,900 tokens):
#              TRAIN_ARGS = ["--train-split", "train,large_train", "--val-split", "val,large_val",
#                            "--max-len", "8192"]
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

FINETUNE_INTRO = """
# graphArt_LM: Phase 2, LoRA fine-tuning on a free Kaggle GPU

Fine-tunes `Qwen2.5-Coder-1.5B-Instruct` on the 1280 training samples, then scores the
tuned adapter **and** the untuned base on the same {n} stratified test samples as every
Phase 1 run, with the `zero_shot` prompt. The test samples are never used in training.

**Before running**, in the right-hand panel under *Session options*:
- **Accelerator:** GPU T4 x2 (training uses one of them)
- **Internet:** on (to install the libraries and download the model)

Then *Save Version → Save & Run All (Commit)*. A 20-step smoke run comes first; if the loss
is not falling (or overflows in fp16, after one retry in fp32) the session stops there.

**When it finishes**, download `graphart_finetune.zip` from the *Output* tab and extract it
into a new folder; it holds `results/` (both runs) and the adapter under `lora/`.

Generated by `scripts/make_kaggle_notebook.py --finetune`. Edit that script, not this file.
"""


def finetune_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack = code_cells(b64, digest)[0]
    return [unpack, ("2. Install the pinned libraries", f"""
from importlib.metadata import PackageNotFoundError, version

PINS = {FINETUNE_PINS!r}

def pinned(package, wanted):
    try:
        return version(package) == wanted
    except PackageNotFoundError:
        return False

missing = [f"{{p}}=={{v}}" for p, v in PINS.items() if not pinned(p, v)]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

# peft probes two optional quantisation libraries when it builds LoRA layers, and
# raises -- rather than skipping them -- if they are installed but older than it
# supports. Kaggle ships torchao 0.10 (peft wants >= 0.16), which killed a run. We
# use neither, so remove them if present.
for pkg in ("torchao", "gptqmodel"):
    try:
        print(f"removing {{pkg}} {{version(pkg)}} (peft rejects old versions; not used here)", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", pkg], check=True)
    except PackageNotFoundError:
        pass

subprocess.run([sys.executable, "-c", "import torch, transformers, peft; "
                "print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), "
                "'| transformers', transformers.__version__, '| peft', peft.__version__)"])
"""), ("3. Download the base model", """
# Its own step with a time limit: a stalled download inside training would otherwise
# sit until the session cap. ~3 GB; a few minutes on Kaggle's network.
FETCH = ("from huggingface_hub import snapshot_download; "
         f"print('downloaded to', snapshot_download({BASE_MODEL!r}))")
for attempt in (1, 2):
    try:
        subprocess.run([sys.executable, "-c", FETCH], check=True, timeout=1800)
        break
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        print(f"download attempt {attempt} failed: {exc}", flush=True)
else:
    raise SystemExit("could not download the base model -- stopping before any training")
"""), ("4. Smoke run: is it learning?", """
# One GPU is plenty for a 1.5B with LoRA, and avoids DataParallel's quirks. Offline from
# here on: the model is on disk, and nothing later should wait on the network.
GPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")

def train(*extra):
    return subprocess.run([sys.executable, "scripts/train_lora.py", "--base", BASE_MODEL,
                           *TRAIN_ARGS, *extra], env=GPU_ENV).returncode

# Every train/val example must fit; checked in seconds, before the GPU does anything.
if train("--out", "lora/smoke", "--check-only") != 0:
    raise SystemExit("the data check failed -- see the traceback above")

# train_lora.py exits 3 when the loss is not falling and 4 when it is NaN/inf. Only
# those two can be an fp16 problem (Qwen2.5 can overflow on a T4), so only they get a
# retry in fp32 (~2x slower). Any other exit code is a crash: stop and read it.
RETRYABLE = (3, 4)
code = train("--out", "lora/smoke", "--smoke-steps", str(SMOKE_STEPS))
if code in RETRYABLE:
    print(f"smoke run failed in fp16 (exit {code}) -- retrying in fp32", flush=True)
    code = train("--out", "lora/smoke", "--smoke-steps", str(SMOKE_STEPS), "--fp32")
    if code == 0:
        TRAIN_ARGS = [*TRAIN_ARGS, "--fp32"]
if code == 3:
    raise SystemExit("the smoke run did not learn, in fp16 or fp32 -- stopping before the full run")
if code != 0:
    raise SystemExit(f"the smoke run crashed (exit {code}) -- see the traceback above")
"""), ("5. Train", """
import zipfile

def package():
    # The results and the adapter, without the optimizer-state checkpoints.
    with zipfile.ZipFile(WORK / "graphart_finetune.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root in (REPO / "results", REPO / "lora" / ADAPTER):
            for p in root.rglob("*") if root.exists() else []:
                if p.is_file() and "checkpoints" not in p.parts:
                    z.write(p, p.relative_to(REPO))

hours_left = TIME_BUDGET_HOURS - (time.time() - START) / 3600 - 1.0   # an hour kept for scoring
if train("--out", f"lora/{ADAPTER}", "--epochs", str(EPOCHS),
         "--time-budget-hours", f"{hours_left:.2f}") != 0:
    raise SystemExit("training failed -- see the log above")
print((REPO / "lora" / ADAPTER / "train_summary.json").read_text())
package()
"""), ("6. Score the tuned adapter and the untuned base", """
for model in [str(REPO / "lora" / ADAPTER), BASE_MODEL]:
    subprocess.run([sys.executable, "-m", "pipeline.run_prompting", "--backend", "hf",
                    "--model", model, "--prompt", "zero_shot", "--stratify", "10", *EVAL_ARGS],
                   env=GPU_ENV)
    package()
for meta in sorted((REPO / "results").glob("*.meta.json")):
    m = json.loads(meta.read_text())
    print(f"{meta.stem:60s} n={m['n']} failed={m['failed']}")
print(f"done in {(time.time() - START) / 3600:.1f} h -- download graphart_finetune.zip")
""")]


# ---------------------------------------------------------------------------
# The closing experiments (RQ4-RQ6), as two notebooks: prompted models on Ollama, and
# the tuned 1.5B plus a held-out-construct adapter on Hugging Face.

# ~1.5 h measured (14B at ~16 tok/s on a T4). Gemma 31B runs at ~5 tok/s, so the edit
# set alone would take it over three hours; it waits for a later session.
CLOSING_OLLAMA_JOBS = [
    job("qwen2.5-coder:14b", "zero_shot", "torchvision", None, 3200),   # longest real answer: 2956
    job("qwen2.5-coder:14b", "zero_shot", "edits", None),
]

# ~2.1 h measured (1.5B at ~24 tok/s, ~32 s per training step on a T4).
CLOSING_HF_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; FULL_ADAPTER = "qwen2.5-coder-1.5b-graphart-lora"; '
    'HELDOUT_ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-no-dead-branch"; HELDOUT_CONSTRUCT = "dead_branch"; '
    'FULL_EVALS = [("torchvision", 3200), ("edits", None)]; HELDOUT_EVALS = [("rq6", None)]; '
    'FULL_EVALS_IF_TIME = [("rq6", None)]; RETRAIN_FULL = False; EPOCHS = 2.0; HELDOUT_EPOCHS = 1.0; '
    'HELDOUT_TRAIN_ARGS = ["--eval-steps", "35"]; SMOKE_STEPS = 8; TIME_BUDGET_HOURS = 2.5; '
    'TRAIN_ARGS = []; EVAL_ARGS = []; VALIDATE_ONLY = False')

CLOSING_HF_NOTES = """
# FULL_ADAPTER     the Phase 2 adapter. Upload its folder as a Kaggle dataset (Add Input ->
#                  Upload). Without it, its evaluations are skipped -- unless RETRAIN_FULL,
#                  which adds ~1.6 h.
# HELDOUT_ADAPTER  a second adapter trained with every HELDOUT_CONSTRUCT program left out
#                  of train and val (RQ6), for HELDOUT_EPOCHS (one epoch: val loss had
#                  already reached 0.0003 after one in Phase 2).
# *_EVALS          (data set, reply cap) pairs, run in order: torchvision (RQ4), edits (RQ5),
#                  rq6 (the probe set). FULL_EVALS_IF_TIME runs last, only if time is left.
# TIME_BUDGET      the whole session, 2.5 h; work that would not fit is skipped, not cut short.
# VALIDATE_ONLY    True: only check the code arrived intact (works on a CPU session).
"""

CLOSING_HF_INTRO = """
# graphArt_LM: the closing experiments on Hugging Face (RQ4-RQ6)

About 2.1 hours, capped at 2.5.

1. Scores the Phase 2 adapter (Qwen2.5-Coder-1.5B + LoRA) on real torchvision code (RQ4)
   and on the controlled-edit set (RQ5).
2. Trains a second adapter, for one epoch, with every program containing a dead branch
   held out, and scores it on the RQ6 probe set (RQ6); then the Phase 2 adapter on the
   same probe, if time is left.

**Before running**: *GPU T4 x2*, *Internet on*. Optionally upload the folder
**Upload the folder `qwen2.5-coder-1.5b-graphart-lora`** (from the Phase 2 zip) as a dataset
under *Add Input -> Upload*. Without it, step 1 is skipped. Then *Save Version → Save & Run All (Commit)*.

**When it finishes**, download `graphart_closing_hf.zip` from the *Output* tab.

Generated by `scripts/make_kaggle_notebook.py --closing-hf`. Edit that script, not this file.
"""


def closing_hf_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. The Phase 2 adapter: uploaded, or trained here", """
import zipfile

GPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
INPUT = Path(os.environ.get("GRAPHART_INPUT", "/kaggle/input"))

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def package():
    # Results and both adapters, without optimizer-state checkpoints.
    with zipfile.ZipFile(WORK / "graphart_closing_hf.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root in (REPO / "results", REPO / "lora" / FULL_ADAPTER, REPO / "lora" / HELDOUT_ADAPTER):
            for p in root.rglob("*") if root.exists() else []:
                if p.is_file() and "checkpoints" not in p.parts:
                    z.write(p, p.relative_to(REPO))

def train(*extra):
    return subprocess.run([sys.executable, "scripts/train_lora.py", "--base", BASE_MODEL,
                           *TRAIN_ARGS, *extra], env=GPU_ENV).returncode

SMOKED = False
def smoke():
    # Before the first training only: data check, then a short run that must learn.
    global SMOKED, TRAIN_ARGS
    if SMOKED:
        return
    if train("--out", "lora/smoke", "--check-only") != 0:
        raise SystemExit("the data check failed -- see the traceback above")
    code = train("--out", "lora/smoke", "--smoke-steps", str(SMOKE_STEPS))
    if code in (3, 4):                      # not learning / NaN: maybe fp16, retry in fp32
        print(f"smoke run failed in fp16 (exit {code}) -- retrying in fp32", flush=True)
        code = train("--out", "lora/smoke", "--smoke-steps", str(SMOKE_STEPS), "--fp32")
        if code == 0:
            TRAIN_ARGS = [*TRAIN_ARGS, "--fp32"]
    if code != 0:
        raise SystemExit(f"the smoke run failed (exit {code}) -- see the log above")
    SMOKED = True

def uploaded_adapter():
    for cfg in INPUT.rglob("adapter_config.json") if INPUT.exists() else []:
        if json.loads(cfg.read_text()).get("base_model_name_or_path") == BASE_MODEL \\
                and (cfg.parent / "adapter_model.safetensors").exists():
            return cfg.parent
    return None

full = REPO / "lora" / FULL_ADAPTER
source = uploaded_adapter()
if source:
    shutil.copytree(source, full, dirs_exist_ok=True)
    print(f"using the uploaded adapter from {source}", flush=True)
elif RETRAIN_FULL:
    print("no adapter uploaded under /kaggle/input -- training it here (~1.6 h)", flush=True)
    smoke()
    if train("--out", f"lora/{FULL_ADAPTER}", "--epochs", str(EPOCHS),
             "--time-budget-hours", f"{max(remaining() / 3600 - 1, 0.5):.2f}") != 0:
        raise SystemExit("training the Phase 2 adapter failed -- see the log above")
else:
    print("NOTE: no Phase 2 adapter uploaded under /kaggle/input, and RETRAIN_FULL is False -- "
          "skipping its evaluations; RQ6 still runs", flush=True)
HAVE_FULL = (full / "adapter_config.json").exists()
package()
"""), ("5. Score the Phase 2 adapter (RQ4, RQ5, all 160 test programs)", """
def evaluate(adapter_dir, evals):
    for split, cap in evals:
        if remaining() < 900:
            print(f"time budget spent -- skipping {adapter_dir.name} on {split}")
            continue
        cmd = [sys.executable, "-m", "pipeline.run_prompting", "--backend", "hf", "--model", str(adapter_dir),
               "--prompt", "zero_shot", "--split", split, *(["--max-tokens", str(cap)] if cap else []), *EVAL_ARGS]
        print(f"\\n##### {adapter_dir.name} on {split}   ({remaining() / 3600:.1f} h left)", flush=True)
        try:
            subprocess.run(cmd, env=GPU_ENV, timeout=remaining())
        except subprocess.TimeoutExpired:
            print("stopped by the time budget")
        package()

if HAVE_FULL:
    evaluate(full, FULL_EVALS)
"""), ("6. Train the held-out adapter (RQ6)", f"""
heldout = REPO / "lora" / HELDOUT_ADAPTER
smoke()
# Keep ~25 minutes for scoring the probe set afterwards.
if train("--out", f"lora/{{HELDOUT_ADAPTER}}", "--epochs", str(HELDOUT_EPOCHS),
         "--exclude-construct", HELDOUT_CONSTRUCT, *HELDOUT_TRAIN_ARGS,
         "--time-budget-hours", f"{{max(remaining() / 3600 - 0.42, 0.25):.2f}}") != 0:
    raise SystemExit("training the held-out adapter failed -- the Phase 2 results above are saved")
print((heldout / "train_summary.json").read_text())
package()
"""), ("7. Score the held-out adapter, and list everything", """
evaluate(heldout, HELDOUT_EVALS)
if HAVE_FULL:
    evaluate(full, FULL_EVALS_IF_TIME)        # the same probe, for a side-by-side
for meta in sorted((REPO / "results").glob("*.meta.json")):
    m = json.loads(meta.read_text())
    print(f"{meta.stem:75s} n={m['n']} failed={m['failed']}")
print(f"done in {(time.time() - START) / 3600:.1f} h -- download graphart_closing_hf.zip")
""")]


# ---------------------------------------------------------------------------
# Fine-tuning on the first dataset plus the library-style one (tiers 1-6), then scoring
# the old test programs and real torchvision code. One session, capped at 2.5 h.

# Estimated from Phase 2 (97 min for 2 x 1280 examples of ~1400 tokens): one epoch of
# 1920 examples averaging ~1900 tokens is ~1.6 h. Scoring needs ~0.8 h.
FINETUNE_LARGE_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-mixed-v2"; '
    'EPOCHS = 1.0; TRAIN_ARGS = ["--train-split", "train,large_train", "--val-split", "val,large_val", '
    '"--max-len", "8192", "--eval-steps", "40"]; FALLBACK_MAX_LEN = 6144; MEMORY_CHECK = 6; SMOKE_STEPS = 8; '
    'EVALS = [("test", None), ("torchvision", 4500)]; EVAL_RESERVE_HOURS = 0.8; TIME_BUDGET_HOURS = 2.5; '
    'EVAL_ARGS = []; VALIDATE_ONLY = False')

FINETUNE_LARGE_NOTES = """
# BASE_MODEL   tuned from scratch (a fresh LoRA), so the result compares with Phase 2.
# ADAPTER      its name; results land in results/hf__<ADAPTER>__zero_shot*.jsonl.
# EPOCHS       passes over train + large_train (1920 examples). One fits the 2.5 h cap.
# TRAIN_ARGS   the splits, and --max-len 8192: the longest example is 6876 tokens.
# FALLBACK_MAX_LEN  if the longest examples do not fit in GPU memory, train with this
#              --max-len instead and leave out the few longer examples (5 at 6144).
# MEMORY_CHECK forward + backward on this many of the longest examples, before anything else.
# EVALS        (data set, reply cap) scored after training: the 40 stratified test programs
#              (did the old skill survive?) and the 21 torchvision samples (real code).
#              Cap 4500: at 3200 both whole networks were cut off mid-answer in Session 1
#              (the longest reference is 2956 tokens; replies run longer).
# EVAL_RESERVE_HOURS  kept back from training for EVALS.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

FINETUNE_LARGE_INTRO = """
# graphArt_LM: fine-tuning on the library-style programs too (Session 1)

Fine-tunes `Qwen2.5-Coder-1.5B-Instruct` with a fresh LoRA on the first dataset **plus**
the library-style programs from `generate_large.py` (1920 examples, answers up to ~6,900
tokens), for one epoch, then scores it on the same 40 test programs as every earlier run
and on the 21 real torchvision samples. About 2.3 hours, capped at 2.5.

Before training it checks that every example encodes, that the longest ones fit in GPU
memory (falling back to a shorter `--max-len` if not), and that a short run learns.

**Before running**: *GPU T4 x2* (one is used), *Internet on*. No inputs needed. Then
*Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_finetune_mixed.zip` from the *Output* tab: the
results and the adapter.

Generated by `scripts/make_kaggle_notebook.py --finetune-large`. Edit that script, not this file.
"""


def finetune_large_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. Check the data, then the GPU memory", """
import zipfile

GPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def train(*extra):
    return subprocess.run([sys.executable, "scripts/train_lora.py", "--base", BASE_MODEL,
                           *TRAIN_ARGS, *extra], env=GPU_ENV).returncode

def package():
    with zipfile.ZipFile(WORK / "graphart_finetune_mixed.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root in (REPO / "results", REPO / "lora" / ADAPTER):
            for p in root.rglob("*") if root.exists() else []:
                if p.is_file() and "checkpoints" not in p.parts:
                    z.write(p, p.relative_to(REPO))

def with_max_len(args, n):
    out = list(args)
    out[out.index("--max-len") + 1] = str(n)
    return out + ["--drop-over-max"]

if train("--out", "lora/check", "--check-only") != 0:
    raise SystemExit("the data check failed -- see the traceback above")
code = train("--out", "lora/check", "--memory-check", str(MEMORY_CHECK))
if code == 5:                                 # train_lora.py: out of memory
    print(f"the longest examples do not fit -- falling back to --max-len {FALLBACK_MAX_LEN}", flush=True)
    TRAIN_ARGS = with_max_len(TRAIN_ARGS, FALLBACK_MAX_LEN)
    code = train("--out", "lora/check", "--memory-check", str(MEMORY_CHECK))
if code != 0:
    raise SystemExit(f"the memory check failed (exit {code}) -- see the log above")
"""), ("5. Smoke run: is it learning?", """
code = train("--out", "lora/smoke", "--smoke-steps", str(SMOKE_STEPS))
if code in (3, 4):                            # not learning / NaN: maybe fp16, retry in fp32
    print(f"smoke run failed in fp16 (exit {code}) -- retrying in fp32", flush=True)
    code = train("--out", "lora/smoke", "--smoke-steps", str(SMOKE_STEPS), "--fp32")
    if code == 0:
        TRAIN_ARGS = [*TRAIN_ARGS, "--fp32"]
if code != 0:
    raise SystemExit(f"the smoke run failed (exit {code}) -- stopping before the full run")
"""), ("6. Train", """
hours = max(remaining() / 3600 - EVAL_RESERVE_HOURS, 0.25)
print(f"training for at most {hours:.2f} h", flush=True)
if train("--out", f"lora/{ADAPTER}", "--epochs", str(EPOCHS), "--time-budget-hours", f"{hours:.2f}") != 0:
    raise SystemExit("training failed -- see the log above")
print((REPO / "lora" / ADAPTER / "train_summary.json").read_text())
package()
"""), ("7. Score it", """
adapter = REPO / "lora" / ADAPTER
for split, cap in EVALS:
    if remaining() < 600:
        print(f"time budget spent -- skipping {split}")
        continue
    cmd = [sys.executable, "-m", "pipeline.run_prompting", "--backend", "hf", "--model", str(adapter),
           "--prompt", "zero_shot", "--split", split, *(["--max-tokens", str(cap)] if cap else []),
           *(["--stratify", "10"] if split == "test" else []), *EVAL_ARGS]
    print(f"\\n##### {split}   ({remaining() / 3600:.1f} h left)", flush=True)
    try:
        subprocess.run(cmd, env=GPU_ENV, timeout=remaining())
    except subprocess.TimeoutExpired:
        print("stopped by the time budget -- the finished replies are kept")
    package()
for meta in sorted((REPO / "results").glob("*.meta.json")):
    m = json.loads(meta.read_text())
    print(f"{meta.stem:75s} n={m['n']} failed={m['failed']}")
print(f"done in {(time.time() - START) / 3600:.1f} h -- download graphart_finetune_mixed.zip")
""")]


# ---------------------------------------------------------------------------
# Evaluation only: the uploaded Phase 2 adapter on any (prompt, data set) jobs, no training.

# CoS writes its simulation before the graph, roughly doubling the answer: the longest
# torchvision graph is 2956 tokens, so 6500 leaves room. Prompts run 2.7-4.6k tokens.
HF_EVAL_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; '
    'EVALS = [("cos", "torchvision", 6500)]; TIME_BUDGET_HOURS = 2.0; EVAL_ARGS = []; VALIDATE_ONLY = False')

HF_EVAL_NOTES = """
# BASE_MODEL   the base the uploaded adapter was trained on (read from its adapter_config.json).
# EVALS        (prompt, data set, reply cap) jobs, run in order with the uploaded adapter.
#              Data sets: test (the stratified 40), torchvision, edits, rq6, large_test (80
#              library-style programs; give it a cap of ~7000 for the whole networks).
# TIME_BUDGET  the whole session; a job that would not fit is skipped, one in progress keeps
#              every reply finished so far (replies are cached one by one).
# EVAL_ARGS    extra flags for run_prompting.py, e.g. ["--limit", "2"] for a quick try.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

HF_EVAL_INTRO = """
# graphArt_LM: the fine-tuned 1.5B on chosen prompts and data sets (evaluation only)

Runs the Phase 2 adapter (Qwen2.5-Coder-1.5B + LoRA) on the jobs in `EVALS` -- by default the
chain-of-simulation prompt (`cos`) on the 21 real torchvision samples. No training. About
20-60 minutes of one T4, capped at 2 hours.

**Before running**: *GPU T4 x2*, *Internet on*, and add the adapter folder
`qwen2.5-coder-1.5b-graphart-lora` as an input (*Add Input* -> your earlier upload, or
*Upload* the folder from `dist/qwen2.5-coder-1.5b-graphart-lora.zip`). Then
*Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_hf_eval.zip` from the *Output* tab.

Generated by `scripts/make_kaggle_notebook.py --hf-eval`. Edit that script, not this file.
"""


def hf_eval_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. Find the uploaded adapter", """
import zipfile

GPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
INPUT = Path(os.environ.get("GRAPHART_INPUT", "/kaggle/input"))

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def package():
    with zipfile.ZipFile(WORK / "graphart_hf_eval.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in (REPO / "results").rglob("*") if (REPO / "results").exists() else []:
            if p.is_file():
                z.write(p, p.relative_to(REPO))

adapter = None
for cfg in INPUT.rglob("adapter_config.json") if INPUT.exists() else []:
    if json.loads(cfg.read_text()).get("base_model_name_or_path") == BASE_MODEL \\
            and (cfg.parent / "adapter_model.safetensors").exists():
        adapter = REPO / "lora" / cfg.parent.name
        shutil.copytree(cfg.parent, adapter, dirs_exist_ok=True)
        break
if adapter is None:
    raise SystemExit("no adapter for this base under /kaggle/input -- add it as an input (see the top)")
print(f"using the adapter {adapter.name}", flush=True)
"""), ("5. Run the jobs", """
for prompt, split, cap in EVALS:
    if remaining() < 600:
        print(f"time budget spent -- skipping {prompt} on {split}")
        continue
    cmd = [sys.executable, "-m", "pipeline.run_prompting", "--backend", "hf", "--model", str(adapter),
           "--prompt", prompt, "--split", split, *(["--max-tokens", str(cap)] if cap else []),
           *(["--stratify", "10"] if split == "test" else []), *EVAL_ARGS]
    print(f"\\n##### {prompt} on {split}   ({remaining() / 3600:.1f} h left)", flush=True)
    try:
        subprocess.run(cmd, env=GPU_ENV, timeout=remaining())
    except subprocess.TimeoutExpired:
        print("stopped by the time budget -- the finished replies are kept")
    package()
for meta in sorted((REPO / "results").glob("*.meta.json")):
    m = json.loads(meta.read_text())
    print(f"{meta.stem:75s} n={m['n']} failed={m['failed']}")
print(f"done in {(time.time() - START) / 3600:.1f} h -- download graphart_hf_eval.zip")
""")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="Ollama model tags, one session's worth (default: %(default)s)")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                    help="prompts to run, in order (default: all six)")
    ap.add_argument("--finetune", action="store_true",
                    help="build notebooks/kaggle_finetune.ipynb (Phase 2) instead")
    ap.add_argument("--closing-hf", action="store_true",
                    help="build notebooks/kaggle_closing_hf.ipynb (RQ4-RQ6, tuned and held-out adapters)")
    ap.add_argument("--finetune-large", action="store_true",
                    help="build notebooks/kaggle_finetune_large.ipynb (train on tiers 1-6, score test + torchvision)")
    ap.add_argument("--hf-eval", action="store_true",
                    help="build notebooks/kaggle_hf_eval.ipynb (the uploaded adapter on chosen jobs, no training)")
    ap.add_argument("--closing-ollama", action="store_true",
                    help="build notebooks/kaggle_closing_ollama.ipynb (RQ4-RQ5, prompted 14B and Gemma 31B)")
    args = ap.parse_args()
    unknown = [p for p in args.prompts if not (ROOT / "prompts" / f"{p}.txt").exists()]
    if unknown:
        raise SystemExit(f"no such prompt(s): {unknown}")
    OUT_DIR.mkdir(exist_ok=True)

    if args.closing_hf:
        files, n = payload(finetune=True)
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_closing_hf.ipynb"
        nb = notebook(CLOSING_HF_CONFIG, closing_hf_cells(b64, digest), n,
                      intro=CLOSING_HF_INTRO, notes=CLOSING_HF_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.finetune_large:
        files, n = payload(finetune=True)
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_finetune_large.ipynb"
        nb = notebook(FINETUNE_LARGE_CONFIG, finetune_large_cells(b64, digest), n,
                      intro=FINETUNE_LARGE_INTRO, notes=FINETUNE_LARGE_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.hf_eval:
        files, n = payload()
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_hf_eval.ipynb"
        nb = notebook(HF_EVAL_CONFIG, hf_eval_cells(b64, digest), n, intro=HF_EVAL_INTRO, notes=HF_EVAL_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.closing_ollama:
        settings = config(CLOSING_OLLAMA_JOBS, hours=2.5)          # the user's cap
        files, n = payload()
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_closing_ollama.ipynb"
        nb_path.write_text(json.dumps(notebook(settings, code_cells(b64, digest), n), indent=1) + "\n",
                           encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} ({len(CLOSING_OLLAMA_JOBS)} jobs, "
              f"bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.finetune:
        files, n = payload(finetune=True)
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_finetune.ipynb"
        nb = notebook(FINETUNE_CONFIG, finetune_cells(b64, digest), n,
                      intro=FINETUNE_INTRO, notes=FINETUNE_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} ({n} test samples + train/val, "
              f"bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    settings = config([job(m, p) for m in args.models for p in args.prompts])
    files, n = payload()
    b64, digest = bundle(files)
    cells = code_cells(b64, digest)
    OUT_DIR.mkdir(exist_ok=True)
    nb_path = OUT_DIR / "kaggle_ollama.ipynb"
    nb_path.write_text(json.dumps(notebook(settings, cells, n), indent=1) + "\n", encoding="utf-8")
    boot = bootstrap(settings, files, cells[1:])
    boot_path = OUT_DIR / "kaggle_bootstrap.txt"
    boot_path.write_text("\n".join(boot) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {nb_path.relative_to(ROOT)} ({n} samples, bundle sha256 {digest[:12]}) and "
          f"{boot_path.relative_to(ROOT)} ({len(boot)} one-line cells, "
          f"{sum(map(len, boot[1:-1])) / 1024:.1f} KB of chunks)")


if __name__ == "__main__":
    main()

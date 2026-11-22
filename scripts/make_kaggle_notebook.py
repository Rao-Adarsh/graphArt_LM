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
              "pipeline/run_prompting.py",
              # the runner numbers the code for the stack and two-step prompts with these
              # (text only, no torch)
              "graphart/__init__.py", "graphart/schema.py", "graphart/two_step.py"]
SAMPLE_FIELDS = ("id", "tier", "num_nodes", "code", "target")
# The closing evaluation sets (RQ4, RQ5), built by build_torchvision_set.py and
# build_edit_set.py; bundled whenever they exist. torchvision_holdout is frozen (see
# data/generated/FROZEN.json): score it once, on the final model, with an 8000 cap.
EVAL_SETS = ("torchvision", "edits", "rq6", "large_test", "torchvision_holdout")
# The library-style training set (build_large_dataset.py), bundled with train/val for
# fine-tuning whenever it exists; select it with TRAIN_ARGS --train-split/--val-split.
LARGE_TRAIN_SETS = ("large_train", "large_val")
# The two-step format (graphart/two_step.py): its reader, its evaluator, and the training
# data scripts/build_two_step.py writes.
TWO_STEP_CODE = ("graphart/__init__.py", "graphart/schema.py", "graphart/two_step.py", "pipeline/run_two_step.py")
TWO_STEP_TRAIN_SETS = ("train", "val", "large_train", "large_val")


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime = len(data), 0          # mtime 0: same inputs, same bundle
    tar.addfile(info, io.BytesIO(data))


def payload(finetune: bool = False, two_step: bool = False) -> tuple[dict[str, bytes], int]:
    """Every file the runner needs on Kaggle, by repo-relative path, and the sample count.

    For fine-tuning it also carries the trainer and the full train and val splits
    (with their edges -- they are the training targets). The test samples stay
    stripped either way. `two_step` carries the two-step training data
    (data/generated/two_step/) and its reader and evaluator instead of the plain splits.
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
    if two_step:
        files["scripts/train_lora.py"] = (ROOT / "scripts" / "train_lora.py").read_bytes()
        for rel in TWO_STEP_CODE:
            files[rel] = (ROOT / rel).read_bytes()
        for split in TWO_STEP_TRAIN_SETS:
            files[f"data/generated/two_step/{split}.jsonl"] = (ROOT / "data" / "generated" / "two_step" /
                                                               f"{split}.jsonl").read_bytes()
        return files, len(rows)
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
        max_tokens: int | None = None, num_ctx: int | None = None, extra: list[str] | None = None) -> dict:
    long = prompt in LONG_PROMPTS or split in LONG_SETS
    out = {"model": model, "prompt": prompt, "split": split, "stratify": stratify,
           "max_tokens": max_tokens or (4096 if long else None), "num_ctx": num_ctx}
    return out | ({"extra": extra} if extra else {})     # more run_prompting flags


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
    if j.get("num_ctx"):
        cmd += ["--num-ctx", str(j["num_ctx"])]
    if j.get("limit"):                      # for trying a notebook on a few samples
        cmd += ["--limit", str(j["limit"])]
    cmd += j.get("extra", [])                # e.g. --nodes-from, --shuffle-nodes
    name += f" {' '.join(j.get('extra', []))}"
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

# Which half of the two-step task is hard on real code: the 14B handed the true nodes
# (edges_only), asked for the nodes alone (nodes_only), and with CoS, on torchvision 21.
# Cheapest first. At ~16 tok/s: ~10 + ~20 min, then cos ~1 h (resnet18's CoS answer is
# ~6k tokens). The cos prompt plus 2.2k of code plus that answer overflows Ollama's
# 8192-token default context, which would silently drop the start of the prompt.
PROBE_14B_JOBS = [
    job("qwen2.5-coder:14b", "edges_only", "torchvision", None, 1500),   # longest edge list: 762
    job("qwen2.5-coder:14b", "nodes_only", "torchvision", None, 3200),   # longest node list: 2194
    job("qwen2.5-coder:14b", "cos", "torchvision", None, 8000, num_ctx=16384),
]

# The second probe (prompts from scripts/build_probe_prompts.py). First, CoS wiring given
# the true nodes: does tracking variables help the 14B wire real code? Then three ways to
# better nodes without fine-tuning, the 14B having listed helper classes as single boxes:
# the @ source stack (zero-shot), two worked examples with helpers, and both. All on
# torchvision 21, ~1.5 h at ~16 tok/s. The few-shot prompts run 3k tokens before the
# code, so every job gets a 16k context.
PROBE_14B_NODES_JOBS = [
    job("qwen2.5-coder:14b", "edges_only_cos", "torchvision", None, 6000, num_ctx=16384),
    job("qwen2.5-coder:14b", "nodes_only_stack", "torchvision", None, 4000, num_ctx=16384),
    job("qwen2.5-coder:14b", "nodes_only_few_shot", "torchvision", None, 3200, num_ctx=16384),
    job("qwen2.5-coder:14b", "nodes_only_stack_few_shot", "torchvision", None, 4000, num_ctx=16384),
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


def finetune_large_cells(b64: str, digest: str, zip_name: str = "graphart_finetune_mixed.zip",
                         runner: str = "pipeline.run_prompting",
                         prompt_args: str = '"--prompt", "zero_shot"') -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. Check the data, then the GPU memory", """
import zipfile

GPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
ZIP_NAME = "__ZIP_NAME__"

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def train(*extra):
    return subprocess.run([sys.executable, "scripts/train_lora.py", "--base", BASE_MODEL,
                           *TRAIN_ARGS, *extra], env=GPU_ENV).returncode

def package():
    with zipfile.ZipFile(WORK / ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
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
""".replace("__ZIP_NAME__", zip_name)), ("5. Smoke run: is it learning?", """
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
    cmd = [sys.executable, "-m", "__RUNNER__", "--backend", "hf", "--model", str(adapter),
           __PROMPT_ARGS__, "--split", split, *(["--max-tokens", str(cap)] if cap else []),
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
print(f"done in {(time.time() - START) / 3600:.1f} h -- download {ZIP_NAME}")
""".replace("__RUNNER__", runner).replace("__PROMPT_ARGS__", prompt_args))]


# ---------------------------------------------------------------------------
# Fork B: the two-step format (graphart/two_step.py) -- nodes with their source lines,
# then edges with the variables they are read from -- on a fresh 1.5B LoRA.

# Both budget cuts: the brief prompts (no rules), every second tier 1-4 program, half the
# val programs. 2560 examples averaging 1378 tokens: 3.5M, a little under Session 2's
# 3.7M, which trained in 96 min. Scoring is two calls per program.
FINETUNE_TWO_STEP_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-two-step"; '
    'EPOCHS = 1.0; TRAIN_ARGS = ["--two-step", "--train-split", "train/2,large_train", '
    '"--val-split", "val/2,large_val/2", "--max-len", "8192", "--eval-steps", "40"]; FALLBACK_MAX_LEN = 6144; '
    'MEMORY_CHECK = 6; SMOKE_STEPS = 8; EVALS = [("test", 3000), ("torchvision", 4500)]; '
    'EVAL_RESERVE_HOURS = 0.8; TIME_BUDGET_HOURS = 2.5; EVAL_ARGS = []; VALIDATE_ONLY = False')

FINETUNE_TWO_STEP_NOTES = """
# BASE_MODEL   tuned from scratch (a fresh LoRA), like Sessions 1 and 2.
# ADAPTER      its name; results land in results/hf__<ADAPTER>__two_step_brief*.jsonl.
# TRAIN_ARGS   --two-step: two examples per program (nodes, then edges), with the brief
#              prompts. train/2 is every second tier 1-4 program; all of tiers 5-6; half
#              the val programs. The longest example is 6224 tokens.
# FALLBACK_MAX_LEN  if the longest examples do not fit in GPU memory, train with this
#              --max-len instead and leave out the few longer examples.
# MEMORY_CHECK forward + backward on this many of the longest examples, before anything else.
# EVALS        (data set, reply cap per call) scored after training, each program in two
#              calls: the 40 stratified test programs and the 21 torchvision samples.
# EVAL_RESERVE_HOURS  kept back from training for EVALS.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

FINETUNE_TWO_STEP_INTRO = """
# graphArt_LM: the two-step model (fork B)

Fine-tunes `Qwen2.5-Coder-1.5B-Instruct` with a fresh LoRA to answer in two steps: first
the nodes, each with the source lines that make it; then, given those nodes, the edges,
each input with the variable it is read from. The training answers were recorded from
the fx trace itself (`graphart/cos_trace.py`), not written by hand.

Then scores it on the same 40 test programs and the 21 torchvision samples, step 2
always given the model's own step-1 nodes. About 2.3 hours, capped at 2.5.

**Before running**: *GPU T4 x2* (one is used), *Internet on*. No inputs needed. Then
*Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_finetune_two_step.zip` from the *Output* tab:
the results and the adapter.

Generated by `scripts/make_kaggle_notebook.py --finetune-two-step`. Edit that script, not this file.
"""


# The same with the full-rules prompts: the brief-prompt adapter wrote a method name,
# `_forward`, into module paths on three Inception-v3 samples, and the rule that a target
# is the attribute path was one of those left out. The rules cost ~1200 tokens a
# program, so tiers 1-4 drop to every eighth program (40 a tier): 1600 examples
# averaging 2344 tokens, 3.75M -- the last run trained 3.62M in 96 min.
FINETUNE_TWO_STEP_FULL_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-two-step-full"; '
    'EPOCHS = 1.0; TRAIN_ARGS = ["--two-step", "--full-prompts", "--train-split", "train/8,large_train", '
    '"--val-split", "val/8,large_val/2", "--max-len", "8192", "--eval-steps", "40"]; FALLBACK_MAX_LEN = 6144; '
    'MEMORY_CHECK = 6; SMOKE_STEPS = 8; EVALS = [("test", 3000), ("torchvision", 4500)]; '
    'EVAL_RESERVE_HOURS = 0.8; TIME_BUDGET_HOURS = 2.5; EVAL_ARGS = []; VALIDATE_ONLY = False')

FINETUNE_TWO_STEP_FULL_NOTES = """
# BASE_MODEL   tuned from scratch (a fresh LoRA), like every earlier session.
# ADAPTER      its name; results land in results/hf__<ADAPTER>__two_step*.jsonl.
# TRAIN_ARGS   --two-step --full-prompts: two examples per program (nodes, then edges),
#              with the full-rules prompts. train/8 is every eighth tier 1-4 program (40 a
#              tier); all of tiers 5-6; val/8 and half of large_val. Longest example 7022.
# FALLBACK_MAX_LEN  if the longest examples do not fit in GPU memory, train with this
#              --max-len instead and leave out the few longer examples.
# MEMORY_CHECK forward + backward on this many of the longest examples, before anything else.
# EVALS        (data set, reply cap per call), each program in two calls with the same
#              full prompts: the 40 stratified test programs and the 21 torchvision samples.
# EVAL_RESERVE_HOURS  kept back from training for EVALS; training ~100 min, and the time
#              guard stops it early (keeping the best checkpoint) if it would eat into this.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

FINETUNE_TWO_STEP_FULL_INTRO = """
# graphArt_LM: the two-step model with the full-rules prompts

Fine-tunes `Qwen2.5-Coder-1.5B-Instruct` with a fresh LoRA on the two-step format (nodes
with their source lines, then edges with the variables they are read from), like the last
run, but with the full-rules prompts: the brief ones left out the rule that a target is the
attribute path, and the last adapter wrote `_forward` into module paths. To pay for the
rules, tiers 1-4 are cut to every eighth program.

Then scores it on the same 40 test programs and the 21 torchvision samples, step 2 always
given the model's own step-1 nodes. About 2.4 hours, capped at 2.5.

**Before running**: *GPU T4 x2* (one is used), *Internet on*. No inputs needed. Then
*Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_finetune_two_step_full.zip` from the *Output* tab:
the results and the adapter.

Generated by `scripts/make_kaggle_notebook.py --finetune-two-step-full`. Edit that script, not this file.
"""


# The same run again on both T4s, so it finishes its epoch: on one GPU it averaged 84 s a
# step and the time guard stopped it at step 64 of ~100. The two-GPU smoke test measured
# 37.3 s a step against 69.3 on one, so ~45 s a step here: ~75 min for the epoch. Training
# runs under torchrun (DistributedDataParallel, 8 of each step's 16 examples per GPU);
# the data and memory checks run in one process; the two scoring jobs run side by side,
# one per GPU. --max-len starts at 6144, where the last run's memory check landed.
FINETUNE_TWO_GPU_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-two-step-full-epoch"; '
    'EPOCHS = 1.0; GPUS = 2; EFFECTIVE_BATCH = 16; TRAIN_ARGS = ["--two-step", "--full-prompts", '
    '"--train-split", "train/8,large_train", "--val-split", "val/8,large_val/2", "--max-len", "6144", '
    '"--drop-over-max", "--eval-steps", "40"]; FALLBACK_MAX_LEN = 5120; MEMORY_CHECK = 6; SMOKE_STEPS = 8; '
    'EVALS = [("test", 3000), ("torchvision", 4500)]; EVAL_RESERVE_HOURS = 0.5; TIME_BUDGET_HOURS = 2.5; '
    'EVAL_ARGS = []; VALIDATE_ONLY = False')

FINETUNE_TWO_GPU_NOTES = """
# BASE_MODEL   tuned from scratch (a fresh LoRA), like every earlier session.
# ADAPTER      its name; results land in results/hf__<ADAPTER>__two_step*.jsonl.
# GPUS, EFFECTIVE_BATCH  training is split across both T4s; each step is still 16
#              examples (8 per GPU), as in every earlier run.
# TRAIN_ARGS   --two-step --full-prompts: two examples per program (nodes, then edges),
#              full-rules prompts. train/8 is every eighth tier 1-4 program (40 a tier);
#              all of tiers 5-6; val/8 and half of large_val. --max-len 6144 leaves out
#              the 10 longest tier-6 examples (of 640), whole -- 8192 did not fit last time.
# FALLBACK_MAX_LEN  if even 6144 does not fit in GPU memory, train with this instead.
# MEMORY_CHECK forward + backward on this many of the longest examples, before anything else.
# EVALS        (data set, reply cap per call), each program in two calls with the same full
#              prompts: the 40 stratified test programs and the 21 torchvision samples.
#              The two jobs run at the same time, one on each GPU.
# EVAL_RESERVE_HOURS  kept back from training for EVALS (they took ~15 min last time).
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

FINETUNE_TWO_GPU_INTRO = """
# graphArt_LM: the full-rules two-step model, a full epoch on two GPUs

The last full-rules run (9 of 21 torchvision graphs right) hit the time cap after about
65% of an epoch. This is the same run -- the same data, prompts, and settings, a fresh
LoRA on `Qwen2.5-Coder-1.5B-Instruct` -- trained on both T4s at once, so the epoch
finishes. Then it scores the 40 test programs and the 21 torchvision samples, step 2
given the model's own step-1 nodes. About 2 hours, capped at 2.5.

**Before running**: *GPU T4 x2* (both are used), *Internet on*. No inputs needed. Then
*Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_finetune_two_step_epoch.zip` from the *Output*
tab: the results, the adapter, and the scoring logs.

Generated by `scripts/make_kaggle_notebook.py --finetune-two-gpu`. Edit that script, not this file.
"""


def finetune_two_gpu_cells(b64: str, digest: str,
                           zip_name: str = "graphart_finetune_two_step_epoch.zip") -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. Two GPUs? Then check the data and the GPU memory", """
import re, zipfile

n = int(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                       capture_output=True, text=True).stdout.strip() or 0)
print(f"{n} GPU(s) visible", flush=True)
if n < GPUS:
    raise SystemExit(f"this session has {n} GPU(s), not {GPUS} -- set the accelerator to GPU T4 x2")

ONE_GPU = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
ALL_GPUS = dict(ONE_GPU, CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(GPUS)))
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(GPUS)]
ZIP_NAME = "__ZIP_NAME__"
LOGS = REPO / "logs"
LOGS.mkdir(exist_ok=True)

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def run(cmd, env):
    # Run, echo the output live, and return (exit code, output).
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return proc.wait(), "".join(lines)

def check(*extra):
    # One process on one GPU: the data check and the memory check.
    return subprocess.run([sys.executable, "scripts/train_lora.py", "--base", BASE_MODEL, *TRAIN_ARGS, *extra],
                          env=ONE_GPU).returncode

def train(*extra):
    # Both GPUs: each step's EFFECTIVE_BATCH examples split between them.
    return run([*TORCHRUN, "scripts/train_lora.py", "--base", BASE_MODEL, *TRAIN_ARGS,
                "--grad-accum", str(EFFECTIVE_BATCH // GPUS), *extra], ALL_GPUS)

def package():
    with zipfile.ZipFile(WORK / ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        for root in (REPO / "results", REPO / "lora" / ADAPTER, LOGS):
            for p in root.rglob("*") if root.exists() else []:
                if p.is_file() and "checkpoints" not in p.parts:
                    z.write(p, p.relative_to(REPO))

def with_max_len(args, n):
    out = list(args)
    out[out.index("--max-len") + 1] = str(n)
    return out

if check("--out", "lora/check", "--check-only") != 0:
    raise SystemExit("the data check failed -- see the traceback above")
code = check("--out", "lora/check", "--memory-check", str(MEMORY_CHECK))
if code == 5:                                 # train_lora.py: out of memory
    print(f"the longest examples do not fit -- falling back to --max-len {FALLBACK_MAX_LEN}", flush=True)
    TRAIN_ARGS = with_max_len(TRAIN_ARGS, FALLBACK_MAX_LEN)
    code = check("--out", "lora/check", "--memory-check", str(MEMORY_CHECK))
if code != 0:
    raise SystemExit(f"the memory check failed (exit {code}) -- see the log above")
""".replace("__ZIP_NAME__", zip_name)), ("5. Smoke run on both GPUs: is it learning?", """
# Under torchrun a failing process's own exit code is lost, so the cause is read from the
# log: "NOT LEARNING" or a non-finite loss can be fp16 trouble, worth one retry in fp32.
code, out = train("--out", "lora/smoke", "--smoke-steps", str(SMOKE_STEPS))
if code != 0 and ("NOT LEARNING" in out or "STOPPED: loss is" in out):
    print(f"smoke run failed in fp16 -- retrying in fp32", flush=True)
    code, out = train("--out", "lora/smoke", "--smoke-steps", str(SMOKE_STEPS), "--fp32")
    if code == 0:
        TRAIN_ARGS = [*TRAIN_ARGS, "--fp32"]
if code != 0:
    raise SystemExit(f"the smoke run failed (exit {code}) -- stopping before the full run")
"""), ("6. Train on both GPUs", """
hours = max(remaining() / 3600 - EVAL_RESERVE_HOURS, 0.25)
print(f"training for at most {hours:.2f} h", flush=True)
code, out = train("--out", f"lora/{ADAPTER}", "--epochs", str(EPOCHS), "--time-budget-hours", f"{hours:.2f}")
(LOGS / "train.log").write_text(out)
if code != 0:
    package()
    raise SystemExit("training failed -- see the log above")
print((REPO / "lora" / ADAPTER / "train_summary.json").read_text())
package()
"""), ("7. Score it: both jobs at once, one per GPU", """
adapter = REPO / "lora" / ADAPTER
jobs = []
# An EVALS entry is (data set, reply cap) or (data set, reply cap, extra flags, name).
for i, (split, cap, *more) in enumerate(EVALS):
    extra, name = (more[0], more[1]) if more else ([], split)
    cmd = [sys.executable, "-m", "pipeline.run_two_step", "--backend", "hf", "--model", str(adapter),
           "--split", split, "--max-tokens", str(cap), *(["--stratify", "10"] if split == "test" else []),
           *extra, *EVAL_ARGS]
    log = open(LOGS / f"score_{name}.log", "w")
    print(f"##### {name} on GPU {i % GPUS}   ({remaining() / 3600:.1f} h left)", flush=True)
    jobs.append((name, log, subprocess.Popen(cmd, env=dict(ONE_GPU, CUDA_VISIBLE_DEVICES=str(i % GPUS)),
                                             stdout=log, stderr=subprocess.STDOUT)))
for name, log, proc in jobs:
    try:
        proc.wait(timeout=max(remaining() - 180, 60))
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"{name}: stopped by the time budget -- the finished replies are kept", flush=True)
    log.close()
    print(f"\\n##### {name} (exit {proc.returncode})\\n" + (LOGS / f"score_{name}.log").read_text(), flush=True)
    package()
for meta in sorted((REPO / "results").glob("*.meta.json")):
    m = json.loads(meta.read_text())
    print(f"{meta.stem:80s} n={m['n']} failed={m['failed']} joined={m.get('joined')}")
print(f"done in {(time.time() - START) / 3600:.1f} h -- download {ZIP_NAME}")
""")]


# ---------------------------------------------------------------------------
# Two GPUs: a smoke test before any long run on both T4s. One GPU at 84 s a step could not
# finish an epoch of the full-prompt two-step data in the 2.5 h cap; DistributedDataParallel
# splits each step's batch across the two. This checks it works on Kaggle and what it buys.

TWO_GPU_SMOKE_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; '
    'TRAIN_ARGS = ["--two-step", "--full-prompts", "--train-split", "train/8,large_train", '
    '"--val-split", "val/8,large_val/2", "--max-len", "6144", "--drop-over-max"]; '
    'SMOKE_STEPS = 8; EFFECTIVE_BATCH = 16; GUARD_BUDGET_HOURS = 0.05; TIME_BUDGET_HOURS = 1.0; '
    'VALIDATE_ONLY = False')

TWO_GPU_SMOKE_NOTES = """
# BASE_MODEL   the model the real run will tune.
# TRAIN_ARGS   the full-prompt two-step data, as the real run uses it (--max-len 6144, where
#              the last run's memory check landed; the few longer examples are left out).
# SMOKE_STEPS  steps timed on one GPU, then on two -- the same effective batch both times.
# GUARD_BUDGET_HOURS  the time budget of the guard test (3 min): it must stop both GPUs at
#              the same step, save, and exit, not hang.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

TWO_GPU_SMOKE_INTRO = """
# graphArt_LM: does training on both T4s work? (smoke test, no real training)

Three checks, about 30 minutes, capped at 1 hour. Nothing is kept but the log:
1. Kaggle gave the session two GPUs.
2. The same few training steps on one GPU and then on two: the seconds per step of each,
   the time a full epoch would take on two, and that the loss still falls on two.
3. The time guard on two GPUs: a short run whose 3-minute budget must stop both at the
   same step, save the adapter, and exit (a hang is caught and reported).

The verdict is printed at the end, after `##### VERDICT`.

**Before running**: *GPU T4 x2* (both are used), *Internet on*. No inputs. Then
*Save Version -> Save & Run All (Commit)*.

Generated by `scripts/make_kaggle_notebook.py --two-gpu-smoke`. Edit that script, not this file.
"""


def two_gpu_smoke_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. Two GPUs?", """
import re

CHECKS = {}
n = int(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                       capture_output=True, text=True).stdout.strip() or 0)
print(f"{n} GPU(s) visible", flush=True)
CHECKS["two GPUs"] = n >= 2
if n < 2:
    raise SystemExit("this session has fewer than two GPUs -- set the accelerator to GPU T4 x2")

def run(cmd, env, timeout):
    # Run, echo the output live, and return (exit code, output, seconds); exit None on a hang.
    t, lines = time.time(), []
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
            if time.time() - t > timeout:
                raise subprocess.TimeoutExpired(cmd, timeout)
        code = proc.wait(timeout=max(timeout - (time.time() - t), 1))
    except subprocess.TimeoutExpired:
        proc.kill()
        code = None
    return code, "".join(lines), time.time() - t

def s_per_step(out):
    m = re.search(r"smoke: loss ([\d.]+) -> ([\d.]+) over (\d+) steps, ([\d.]+) s/step -> (\w+)", out)
    return (float(m.group(4)), float(m.group(1)), float(m.group(2)), m.group(5)) if m else None

ENV = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
TRAINER = ["scripts/train_lora.py", "--base", BASE_MODEL, *TRAIN_ARGS]
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2"]
"""), ("5. The same steps on one GPU, then on two", """
code1, out1, _ = run([sys.executable, *TRAINER, "--out", "lora/one", "--smoke-steps", str(SMOKE_STEPS),
                      "--grad-accum", str(EFFECTIVE_BATCH)], dict(ENV, CUDA_VISIBLE_DEVICES="0"), 1800)
one = s_per_step(out1)
code2, out2, _ = run([*TORCHRUN, *TRAINER, "--out", "lora/two", "--smoke-steps", str(SMOKE_STEPS),
                      "--grad-accum", str(EFFECTIVE_BATCH // 2)], dict(ENV, CUDA_VISIBLE_DEVICES="0,1"), 1800)
two = s_per_step(out2)
CHECKS["one GPU learns"] = code1 == 0 and bool(one) and one[3] == "OK"
CHECKS["two GPUs run and learn"] = code2 == 0 and bool(two) and two[3] == "OK"
examples = re.search(r"(\d+) train / \d+ val examples", out1 or out2)
EXAMPLES = int(examples.group(1)) if examples else None
print(f"one GPU: {one}  |  two GPUs: {two}  (s/step, first loss, last loss, verdict)", flush=True)
"""), ("6. The time guard on two GPUs", """
# 160 programs is 10 steps; a 3-minute budget stops it a few steps in. Both GPUs must stop
# at the same step, evaluate, save and exit. 20 minutes means it hung.
code3, out3, secs3 = run([*TORCHRUN, *TRAINER, "--out", "lora/guard", "--epochs", "1", "--limit-train", "160",
                          "--limit-val", "8", "--grad-accum", str(EFFECTIVE_BATCH // 2), "--eval-steps", "40",
                          "--time-budget-hours", str(GUARD_BUDGET_HOURS)],
                         dict(ENV, CUDA_VISIBLE_DEVICES="0,1"), 1200)
summary = REPO / "lora" / "guard" / "train_summary.json"
saved = json.loads(summary.read_text()) if summary.exists() else {}
stops = re.findall(r"time budget spent at step (\d+)", out3)
CHECKS["guard stops both GPUs together and saves"] = (code3 == 0 and saved.get("gpus") == 2
                                                      and len(set(stops)) == 1 and len(stops) == 2)
print(f"exit {code3} after {secs3:.0f} s; stop messages at steps {stops}; summary: "
      f"{ {k: saved.get(k) for k in ('gpus', 'steps', 'effective_batch', 'best_eval_loss')} }", flush=True)
"""), ("7. Verdict", """
print("\\n##### VERDICT")
for name, ok in CHECKS.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
if one and two:
    steps = -(-EXAMPLES // EFFECTIVE_BATCH) if EXAMPLES else None
    print(f"  one GPU {one[0]:.1f} s/step, two GPUs {two[0]:.1f} s/step: {one[0] / two[0]:.2f}x faster")
    if steps:
        print(f"  a full epoch ({EXAMPLES} examples, {steps} steps of {EFFECTIVE_BATCH}): ~{steps * one[0] / 60:.0f} min "
              f"on one GPU, ~{steps * two[0] / 60:.0f} min on two")
print(f"  done in {(time.time() - START) / 60:.0f} min")
(WORK / "two_gpu_smoke.json").write_text(json.dumps({"checks": CHECKS, "one_gpu": one, "two_gpus": two,
                                                      "examples": EXAMPLES}, indent=1))
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

# os.walk with followlinks: Kaggle may mount inputs through links, which Path.rglob does
# not descend into on Python 3.13.
found = [Path(d) / "adapter_config.json" for d, _, files in os.walk(INPUT, followlinks=True)
         if "adapter_config.json" in files] if INPUT.exists() else []
adapter = None
# ADAPTER, when the settings name one, must match the folder exactly: with two adapters of
# the same base attached, the first one found was once taken -- the wrong one.
WANTED = globals().get("ADAPTER")
for cfg in found:
    if json.loads(cfg.read_text()).get("base_model_name_or_path") == BASE_MODEL \\
            and (cfg.parent / "adapter_model.safetensors").exists() \\
            and (WANTED is None or cfg.parent.name == WANTED):
        adapter = REPO / "lora" / cfg.parent.name
        shutil.copytree(cfg.parent, adapter, dirs_exist_ok=True)
        break
if adapter is None:
    # Say what is there, so a wrong upload is visible in the log rather than guessed at.
    print(f"adapter_config.json files found: {[str(c) for c in found]}")
    for d, dirs, files in os.walk(INPUT, followlinks=True):
        if d.count(os.sep) - str(INPUT).count(os.sep) <= 3:
            print(f"  {d}: {sorted(dirs)[:10]} {sorted(files)[:10]}")
    raise SystemExit(f"no adapter {WANTED or ''} for this base under /kaggle/input -- add it as an input "
                     "(see the top)")
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


# ---------------------------------------------------------------------------
# E2 (docs/experiments.md): the controlled-edit set (RQ5) on a one-step and a two-step
# adapter, both right on all 40 test programs. Does writing each edge with the variable it
# is read from make answers follow edited code? Phase 2's one-step adapter answered 68%
# (47/69) of meaningful edits with the graph of the unedited program. One adapter per GPU,
# side by side: 193 programs at ~14 s each, ~45 min, ~1 h with the setup.
EDITS_TWO_ADAPTER_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; SPLIT = "edits"; '
    'JOBS = [("qwen2.5-coder-1.5b-graphart-lora-mixed-v2", "pipeline.run_prompting", ["--prompt", "zero_shot"]), '
    '("qwen2.5-coder-1.5b-graphart-lora-two-step", "pipeline.run_two_step", ["--brief", "--max-tokens", "3000"])]; '
    'TIME_BUDGET_HOURS = 1.5; EVAL_ARGS = []; VALIDATE_ONLY = False')

EDITS_TWO_ADAPTER_NOTES = """
# BASE_MODEL   the base both adapters were tuned from.
# SPLIT        the 193 controlled edits of the test programs (124 neutral, 69 meaningful).
# JOBS         (adapter folder name, runner, its flags), one per GPU, run side by side, each
#              with the settings its test-split run used: mixed-v2 one step (zero_shot,
#              default cap), and the brief two-step adapter (both steps, 3000 per call).
# EVAL_ARGS    extra flags for both runners, e.g. ["--limit", "5"] to try the notebook.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

EDITS_TWO_ADAPTER_INTRO = """
# graphArt_LM: controlled edits, one-step vs two-step (evaluation only)

Runs two fine-tuned 1.5B adapters on the 193 controlled edits of the test programs, one
on each T4 at the same time: `mixed-v2` (whole graph in one reply) and the brief
`two-step` adapter (nodes, then edges with the variable each input is read from). No
training. About 1 hour, capped at 1.5.

**Before running**: *GPU T4 x2* (both are used), *Internet on*, and **both adapters
attached under Input** (*Add Input* -> your datasets): `qwen2.5-coder-1.5b-graphart-lora-mixed-v2`
and `qwen2.5-coder-1.5b-graphart-lora-two-step` (the zips in `dist/`, uploaded as datasets).
Check the Input panel lists both before saving. Then *Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_edits_eval.zip` from the *Output* tab.

Generated by `scripts/make_kaggle_notebook.py --edits-two-adapter`. Edit that script, not this file.
"""


def edits_two_adapter_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. Two GPUs, and both uploaded adapters", """
import zipfile

n = int(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                       capture_output=True, text=True).stdout.strip() or 0)
print(f"{n} GPU(s) visible", flush=True)
if n < len(JOBS):
    raise SystemExit(f"{len(JOBS)} jobs need {len(JOBS)} GPUs, this session has {n} -- set GPU T4 x2")
INPUT = Path(os.environ.get("GRAPHART_INPUT", "/kaggle/input"))
LOGS = REPO / "logs"
LOGS.mkdir(exist_ok=True)

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def package():
    with zipfile.ZipFile(WORK / "graphart_edits_eval.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root in (REPO / "results", LOGS):
            for p in root.rglob("*") if root.exists() else []:
                if p.is_file():
                    z.write(p, p.relative_to(REPO))

# Each adapter by its exact folder name (os.walk with followlinks: Kaggle may mount inputs
# through links, which Path.rglob does not follow on Python 3.13).
found = {Path(d).name: Path(d) for d, _, files in os.walk(INPUT, followlinks=True)
         if "adapter_config.json" in files and "adapter_model.safetensors" in files} if INPUT.exists() else {}
missing = [name for name, _, _ in JOBS if name not in found]
if missing:
    print(f"adapters found: {sorted(found)}")
    for d, dirs, files in os.walk(INPUT, followlinks=True):
        if d.count(os.sep) - str(INPUT).count(os.sep) <= 3:
            print(f"  {d}: {sorted(dirs)[:10]} {sorted(files)[:10]}")
    raise SystemExit(f"missing adapter(s) {missing} under /kaggle/input -- attach them as inputs (see the top)")
for name, _, _ in JOBS:
    cfg = json.loads((found[name] / "adapter_config.json").read_text())
    if cfg.get("base_model_name_or_path") != BASE_MODEL:
        raise SystemExit(f"{name} was tuned from {cfg.get('base_model_name_or_path')}, not {BASE_MODEL}")
    shutil.copytree(found[name], REPO / "lora" / name, dirs_exist_ok=True)
    print(f"using {name}", flush=True)
"""), ("5. Both adapters at once, one per GPU", """
ENV = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
running = []
for gpu, (name, runner, flags) in enumerate(JOBS):
    cmd = [sys.executable, "-m", runner, "--backend", "hf", "--model", str(REPO / "lora" / name),
           "--split", SPLIT, *flags, *EVAL_ARGS]
    log = open(LOGS / f"{name}.log", "w")
    print(f"##### {name} on GPU {gpu}: {' '.join(cmd[2:])}", flush=True)
    running.append((name, log, subprocess.Popen(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                                                stdout=log, stderr=subprocess.STDOUT)))
for name, log, proc in running:
    try:
        proc.wait(timeout=max(remaining() - 180, 60))
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"{name}: stopped by the time budget -- the finished replies are cached, not written", flush=True)
    log.close()
    text = (LOGS / f"{name}.log").read_text()
    print(f"\\n##### {name} (exit {proc.returncode}), last lines:\\n" + "\\n".join(text.splitlines()[-15:]), flush=True)
    package()
for meta in sorted((REPO / "results").glob("*.meta.json")):
    m = json.loads(meta.read_text())
    print(f"{meta.stem:80s} n={m['n']} failed={m['failed']}")
print(f"done in {(time.time() - START) / 3600:.1f} h -- download graphart_edits_eval.zip")
""")]


# ---------------------------------------------------------------------------
# The frozen holdout, scored once (data/generated/FROZEN.json, amendment of 2026-11-15).
# GPU 0: mixed-v2 answers, then the wirer is handed mixed-v2's nodes -- the final pipeline.
# GPU 1: the adapter of the 26 Oct submission. mixed-v2 alone is a baseline for free.

HOLDOUT_MIXED_V2 = "results/hf__qwen2.5-coder-1.5b-graphart-lora-mixed-v2__zero_shot__torchvision_holdout.jsonl"
HOLDOUT_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; SPLIT = "torchvision_holdout"; LANES = ['
    '[("qwen2.5-coder-1.5b-graphart-lora-mixed-v2", "pipeline.run_prompting", ["--prompt", "zero_shot", "--max-tokens", "8000"]), '
    '("qwen2.5-coder-1.5b-graphart-lora-wirer", "pipeline.run_two_step", ["--brief", "--max-tokens", "8000", '
    f'"--step1-from", "{HOLDOUT_MIXED_V2}"])], '
    '[("qwen2.5-coder-1.5b-graphart-lora", "pipeline.run_prompting", ["--prompt", "zero_shot", "--max-tokens", "8000"])]]; '
    'TIME_BUDGET_HOURS = 2.0; EVAL_ARGS = []; VALIDATE_ONLY = False')

HOLDOUT_NOTES = """
# BASE_MODEL   the base all three adapters were tuned from.
# SPLIT        the 17 frozen holdout samples (data/generated/FROZEN.json). Scored ONCE.
# LANES        one list per GPU, run side by side; the steps of a lane run in order.
#              GPU 0, the final pipeline: mixed-v2 (zero_shot), then the wirer handed
#              mixed-v2's node lists. GPU 1: the adapter of the 26 Oct submission.
#              Reply caps 8000, as the protocol fixes (the longest reference is 7220).
# EVAL_ARGS    leave empty: the protocol scores all 17.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

HOLDOUT_INTRO = """
# graphArt_LM: the frozen torchvision holdout, scored once

17 real torchvision blocks and networks that nothing was tuned on (frozen 31 Oct). Three
fine-tuned 1.5B adapters, no training:
- **GPU 0, the final pipeline**: `mixed-v2` writes the graph; its node list goes to the
  `wirer`, which writes the edges.
- **GPU 1, the previous submission**: the adapter of 26 Oct.

About 1 hour, capped at 2. **Run it once.** If it fails, do not re-run: send the log.

**Before running**: *GPU T4 x2*, *Internet on*, and **three adapters attached under Input**
(*Add Input* -> your datasets), uploaded from the zips in `dist/`:
`qwen2.5-coder-1.5b-graphart-lora-mixed-v2`, `qwen2.5-coder-1.5b-graphart-lora-wirer` and
`qwen2.5-coder-1.5b-graphart-lora`. Check the Input panel lists all three. Then
*Save Version -> Save & Run All (Commit)*, and download `graphart_holdout.zip` from *Output*.

Generated by `scripts/make_kaggle_notebook.py --holdout`. Edit that script, not this file.
"""


def holdout_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. Two GPUs, and the three uploaded adapters", """
import threading, zipfile

n = int(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                       capture_output=True, text=True).stdout.strip() or 0)
print(f"{n} GPU(s) visible", flush=True)
if n < len(LANES):
    raise SystemExit(f"{len(LANES)} lanes need {len(LANES)} GPUs, this session has {n} -- set GPU T4 x2")
INPUT = Path(os.environ.get("GRAPHART_INPUT", "/kaggle/input"))
LOGS = REPO / "logs"
LOGS.mkdir(exist_ok=True)
ADAPTERS = list(dict.fromkeys(name for lane in LANES for name, _, _ in lane))

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def package():
    with zipfile.ZipFile(WORK / "graphart_holdout.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root in (REPO / "results", LOGS):
            for p in root.rglob("*") if root.exists() else []:
                if p.is_file():
                    z.write(p, p.relative_to(REPO))

# Each adapter by its exact folder name (os.walk with followlinks: Kaggle may mount inputs
# through links, which Path.rglob does not follow on Python 3.13).
found = {Path(d).name: Path(d) for d, _, files in os.walk(INPUT, followlinks=True)
         if "adapter_config.json" in files and "adapter_model.safetensors" in files} if INPUT.exists() else {}
missing = [name for name in ADAPTERS if name not in found]
if missing:
    print(f"adapters found: {sorted(found)}")
    for d, dirs, files in os.walk(INPUT, followlinks=True):
        if d.count(os.sep) - str(INPUT).count(os.sep) <= 3:
            print(f"  {d}: {sorted(dirs)[:10]} {sorted(files)[:10]}")
    raise SystemExit(f"missing adapter(s) {missing} under /kaggle/input -- attach them as inputs (see the top). "
                     "Nothing has been scored, so it is safe to fix and run again.")
for name in ADAPTERS:
    cfg = json.loads((found[name] / "adapter_config.json").read_text())
    if cfg.get("base_model_name_or_path") != BASE_MODEL:
        raise SystemExit(f"{name} was tuned from {cfg.get('base_model_name_or_path')}, not {BASE_MODEL}")
    shutil.copytree(found[name], REPO / "lora" / name, dirs_exist_ok=True)
    print(f"using {name}", flush=True)
"""), ("5. Score the holdout: one lane per GPU", """
ENV = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
codes = {}

def run_lane(gpu, lane):
    for name, runner, flags in lane:
        flags = [str(REPO / f) if f.startswith("results/") else f for f in flags]
        given = [f for f in flags if f.endswith(".jsonl")]
        if given and not Path(given[0]).exists():
            codes[name] = "skipped: no node lists to hand it"
            continue
        if remaining() < 300:
            codes[name] = "skipped: out of time"
            continue
        cmd = [sys.executable, "-m", runner, "--backend", "hf", "--model", str(REPO / "lora" / name),
               "--split", SPLIT, *flags, *EVAL_ARGS]
        print(f"##### {name} on GPU {gpu}: {' '.join(cmd[2:])}", flush=True)
        with open(LOGS / f"{name}.log", "w") as log:
            proc = subprocess.Popen(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdout=log, stderr=subprocess.STDOUT)
            try:
                codes[name] = proc.wait(timeout=max(remaining() - 180, 60))
            except subprocess.TimeoutExpired:
                proc.kill()
                codes[name] = "stopped by the time budget"
        print(f"##### {name}: {codes[name]}", flush=True)

threads = [threading.Thread(target=run_lane, args=(gpu, lane)) for gpu, lane in enumerate(LANES)]
for t in threads:
    t.start()
for t in threads:
    t.join()
for name in ADAPTERS:
    text = (LOGS / f"{name}.log").read_text() if (LOGS / f"{name}.log").exists() else ""
    print(f"\\n##### {name} ({codes.get(name)}), last lines:\\n" + "\\n".join(text.splitlines()[-8:]), flush=True)
package()
for meta in sorted((REPO / "results").glob("*.meta.json")):
    m = json.loads(meta.read_text())
    print(f"{meta.stem:80s} n={m['n']} failed={m['failed']}")
print(f"done in {(time.time() - START) / 3600:.1f} h -- download graphart_holdout.zip")
""")]


# ---------------------------------------------------------------------------
# Evaluation only, the uploaded (brief-prompt) two-step adapter: step 2 handed mixed-v2's
# node lists instead of its own step 1 -- how many graphs step 2 delivers from good nodes,
# and so how much a better step 1 is worth.

# Bundled under data/, not results/: bundle files carry mtime 0, which zipfile refuses
# when the results are packaged -- and the input is not a result of this run anyway.
MIXED_V2_TV = "results/hf__qwen2.5-coder-1.5b-graphart-lora-mixed-v2__zero_shot__torchvision.jsonl"
MIXED_V2_TV_BUNDLED = "data/step1_given/hf__qwen2.5-coder-1.5b-graphart-lora-mixed-v2__zero_shot__torchvision.jsonl"

# E1 and E3 (docs/experiments.md, "Where this leaves the plan"). E1: the 14B's CoS wiring
# handed mixed-v2's node lists, which are right on 15 of 21 -- do the best node lister
# and the best wirer compose? (ceiling 13: node-right there AND wired right on true nodes).
# E3: CoS wiring on the true nodes listed in a random valid topological order, so parallel
# branches interleave -- does it track variables, or lean on the listing order? The
# reordered lists are made here (shuffled_node_files), so no reference edges go to
# Kaggle; only the 11 samples whose order can change, two seeds. ~16 tok/s: E1 ~45 min,
# each E3 seed ~30 min.
SHUFFLE_SEEDS = (1, 2)
SHUFFLED_BUNDLED = "data/step1_given/reference__shuffled{seed}__torchvision.jsonl"
PROBE_14B_COMPOSE_JOBS = [
    job("qwen2.5-coder:14b", "edges_only_cos", "torchvision", None, 6000, num_ctx=16384,
        extra=["--nodes-from", MIXED_V2_TV_BUNDLED]),
    *(job("qwen2.5-coder:14b", "edges_only_cos", "torchvision", None, 6000, num_ctx=16384,
          extra=["--nodes-from", SHUFFLED_BUNDLED.format(seed=seed)]) for seed in SHUFFLE_SEEDS),
]


def shuffled_node_files(split: str = "torchvision") -> dict[str, bytes]:
    """The reference node lists in a random valid topological order (one file per seed,
    as `--nodes-from` reads them: id and reply), for the samples whose order changes."""
    from pipeline.run_prompting import node_block, shuffled
    rows = load_split(split, None, None, None)
    out = {}
    for seed in SHUFFLE_SEEDS:
        lines = [json.dumps({"id": r["id"], "reply": node_block(g)}) for r in rows
                 if (g := shuffled(r["graph"], f"{seed}:{r['id']}"))["nodes"] != r["graph"]["nodes"]]
        out[SHUFFLED_BUNDLED.format(seed=seed)] = "".join(line + "\n" for line in lines).encode("utf-8")
    return out

PROBE_14B_COMPOSE_INTRO = """
# graphArt_LM: composing node lists and CoS wiring (Qwen 14B, torchvision 21)

Three jobs, inference only, on Ollama:
1. **E1**: the 14B wires the graph with the CoS prompt, handed the node lists that the
   fine-tuned 1.5B (mixed-v2) wrote, instead of the true ones.
2. **E3**: the same prompt on the true nodes, listed in a random valid order so parallel
   branches interleave (two seeds; only the 11 samples whose order can change).

About 1.75 hours, capped at 2.5. Scoring happens on the laptop.

**Before running**: *GPU T4 x2*, *Internet on*. No inputs. Then
*Save Version -> Save & Run All (Commit)*. When it finishes, download
`graphart_results.zip` from the *Output* tab.

Generated by `scripts/make_kaggle_notebook.py --probe-14b-compose`. Edit that script, not this file.
"""

# ---------------------------------------------------------------------------
# E4 (docs/experiments.md): a 1.5B wirer -- step 2 alone, trained on stack-free reference
# node lists -- then handed mixed-v2's node lists. E1 got 13/21 doing this with the 14B;
# can a 1.5B do it, and does a variable-tagged wirer, unlike the 14B (E3), survive a
# reordered node list?

MIXED_V2_TEST = "results/hf__qwen2.5-coder-1.5b-graphart-lora-mixed-v2__zero_shot.jsonl"
MIXED_V2_TEST_BUNDLED = "data/step1_given/hf__qwen2.5-coder-1.5b-graphart-lora-mixed-v2__zero_shot.jsonl"
TRUE_NODES_BUNDLED = "data/step1_given/reference__truenodes__torchvision.jsonl"


def true_node_file(split: str = "torchvision") -> dict[str, bytes]:
    """The reference node lists in the traced order, no edges, as `--step1-from` reads them."""
    from pipeline.run_prompting import node_block
    rows = load_split(split, None, None, None)
    return {TRUE_NODES_BUNDLED: "".join(json.dumps({"id": r["id"], "reply": node_block(r["graph"])}) + "\n"
                                        for r in rows).encode("utf-8")}


def _wirer_eval(split: str, nodes: str, name: str) -> str:
    return f'("{split}", 3000, ["--brief", "--step1-from", "{nodes}"], "{name}")'


E4_WIRER_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-wirer"; '
    'EPOCHS = 1.0; GPUS = 2; EFFECTIVE_BATCH = 16; TRAIN_ARGS = ["--wirer", "--train-split", "train/2,large_train", '
    '"--val-split", "val/8,large_val/2", "--max-len", "4096", "--drop-over-max", "--eval-steps", "20"]; '
    'FALLBACK_MAX_LEN = 3072; MEMORY_CHECK = 6; SMOKE_STEPS = 8; EVALS = ['
    + ", ".join([_wirer_eval("torchvision", MIXED_V2_TV_BUNDLED, "tv_mixed-v2_nodes"),
                 _wirer_eval("test", MIXED_V2_TEST_BUNDLED, "test_mixed-v2_nodes"),
                 _wirer_eval("torchvision", TRUE_NODES_BUNDLED, "tv_true_nodes"),
                 *(_wirer_eval("torchvision", SHUFFLED_BUNDLED.format(seed=s), f"tv_shuffled{s}")
                   for s in SHUFFLE_SEEDS)])
    + ']; EVAL_RESERVE_HOURS = 0.5; TIME_BUDGET_HOURS = 2.0; EVAL_ARGS = []; VALIDATE_ONLY = False')

E4_WIRER_NOTES = """
# BASE_MODEL   tuned from scratch (a fresh LoRA), like every earlier session.
# ADAPTER      its name; results land in results/hf__<ADAPTER>__two_step_brief_nodes_from_*.jsonl.
# TRAIN_ARGS   --wirer: step 2 only, one example per program -- the brief edges prompt given
#              the reference nodes WITHOUT stacks, as a one-step model lists them. The same
#              programs as the brief two-step adapter: every 2nd tier 1-4 program, all of 5-6.
#              ~1,150 tokens an example (max ~4,000): about 80 steps, ~35 min on two T4s.
# EVALS        (data set, reply cap, flags, name), each step 2 only, handed a node file:
#              mixed-v2's replies on torchvision (E4) and on the 40 test programs; the true
#              nodes; the true nodes in two random valid orders (as in E3, 11-12 samples).
#              Five jobs, alternating between the two GPUs.
# TIME_BUDGET  the whole session; training stops early to leave EVAL_RESERVE_HOURS for EVALS.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

E4_WIRER_INTRO = """
# graphArt_LM: E4, a 1.5B wirer for mixed-v2's node lists

Trains a fresh LoRA on step 2 alone -- wiring a given node list, each input tagged with the
variable it is read from -- on node lists without stacks, then hands it the node lists the
one-step mixed-v2 model wrote. The 14B did this at 13/21 (E1); the question is whether the
1.5B can, so that the whole pipeline is two small adapters.

Also scored: the true node lists, and the true nodes in a random valid order (does it track
variables, or lean on the order, as the 14B did?).

About 1.3 hours on both T4s, capped at 2. Scoring happens on the laptop.

**Before running**: *GPU T4 x2*, *Internet on*. **No inputs.** Then
*Save Version -> Save & Run All (Commit)*. When it finishes, download
`graphart_e4_wirer.zip` from the *Output* tab.

Generated by `scripts/make_kaggle_notebook.py --e4-wirer`. Edit that script, not this file.
"""

# Step 2 alone is ~600 tokens a program at ~24 tok/s: ~10 min, ~25 with the setup.
TWO_STEP_EVAL_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-two-step"; '
    f'JOBS = [(["--brief", "--step1-from", "{MIXED_V2_TV_BUNDLED}"], "torchvision", 3000)]; '
    'TIME_BUDGET_HOURS = 1.0; EVAL_ARGS = []; VALIDATE_ONLY = False')

TWO_STEP_EVAL_NOTES = """
# BASE_MODEL   the base the uploaded adapter was trained on (read from its adapter_config.json).
# JOBS         (run_two_step.py flags, data set, reply cap per call): step 2 given mixed-v2's
#              node lists, with the brief prompt the adapter was trained on.
# TIME_BUDGET  the whole session; a job that would not fit is skipped, one in progress keeps
#              every reply finished so far (replies are cached one by one).
# EVAL_ARGS    extra flags for run_two_step.py, e.g. ["--limit", "2"] for a quick try.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

TWO_STEP_EVAL_INTRO = """
# graphArt_LM: the two-step 1.5B's step 2, given mixed-v2's nodes (evaluation only)

Runs step 2 of the two-step adapter (`qwen2.5-coder-1.5b-graphart-lora-two-step`) on the
21 torchvision samples, handed the one-step mixed-v2 model's node lists instead of its own
step 1. No training. About 25 minutes of one T4, capped at 1 hour.

**Before running**: *GPU T4 x2*, *Internet on*, and add the adapter folder
`qwen2.5-coder-1.5b-graphart-lora-two-step` as an input (*Add Input* -> *Upload* ->
`dist/qwen2.5-coder-1.5b-graphart-lora-two-step.zip`). Then
*Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_hf_eval.zip` from the *Output* tab.

Generated by `scripts/make_kaggle_notebook.py --two-step-eval`. Edit that script, not this file.
"""


def two_step_eval_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    return hf_eval_cells(b64, digest)[:4] + [("5. Run the jobs", """
for flags, split, cap in JOBS:
    if remaining() < 600:
        print(f"time budget spent -- skipping {flags} on {split}")
        continue
    cmd = [sys.executable, "-m", "pipeline.run_two_step", "--backend", "hf", "--model", str(adapter),
           *flags, "--split", split, "--max-tokens", str(cap),
           *(["--stratify", "10"] if split == "test" else []), *EVAL_ARGS]
    print(f"\\n##### {flags} on {split}   ({remaining() / 3600:.1f} h left)", flush=True)
    try:
        subprocess.run(cmd, env=GPU_ENV, timeout=remaining())
    except subprocess.TimeoutExpired:
        print("stopped by the time budget -- the finished replies are kept")
    package()
for meta in sorted((REPO / "results").glob("*.meta.json")):
    m = json.loads(meta.read_text())
    print(f"{meta.stem:85s} n={m['n']} failed={m['failed']} joined={m.get('joined')}")
print(f"done in {(time.time() - START) / 3600:.1f} h -- download graphart_hf_eval.zip")
""")]


# ---------------------------------------------------------------------------
# Memory probe (20 Nov, hierarchical schema): can the free T4s train examples of 8-20k tokens?
# One forward + backward per length through train_lora.py's own model setup and loss, on one
# GPU and with the model split over both. No training, nothing kept but the numbers.

MEMORY_PROBE_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; LENGTHS = [2048, 4096, 6144, 8192, 12288, 16384, 20480]; '
    'MODES = ["one"]; VARIANTS = ["default", "nogqa", "efficient"]; VALIDATE_ONLY = False')

MEMORY_PROBE_NOTES = """
# BASE_MODEL   the model training will tune.
# LENGTHS      sequence lengths to try, in tokens (prompt + answer). With the code reducer the
#              longest test input is ~12k; timm training examples reach ~12k after reduction.
# MODES        "one": the model on one T4 (as training runs now); "split": its layers spread
#              over both T4s in one process (twice the memory, slower).
# VARIANTS     attention set-ups (the first probe ran out of memory at 8k on both modes):
#              "default" as training runs now; "nogqa" expands Qwen's shared key/value heads
#              first so PyTorch's memory-efficient kernel may run; "efficient" also forbids any
#              other kernel, so it errors out if that kernel cannot run on a T4.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

MEMORY_PROBE_INTRO = """
# graphArt_LM: memory probe -- how long a training example fits on the free T4s?

One training step (forward + backward, exactly as `train_lora.py` does it: fp16, gradient
checkpointing, LoRA, the chunked answer-only loss) at 2k to 20k tokens on one T4, under three
attention set-ups. The first probe ran out of memory at 8k even split over both T4s, which
points at attention running on PyTorch's plain kernel (memory grows with length squared);
this one finds the ceiling of the current set-up and tests the fix. Prints the peak memory and seconds per step of
each, and a verdict: the longest length that stays under 85% of a GPU, and what a full
training run would take at that speed. About 15 minutes, nothing is trained or kept.

**Before running**: *GPU T4 x2*, *Internet on* (to install the libraries and download the
model). No inputs. Then *Save Version -> Save & Run All (Commit)*.

**When it finishes**: the verdict is at the end of the log, after `##### VERDICT`; the numbers
are in `memory_probe.json` in the *Output* tab.

Generated by `scripts/make_kaggle_notebook.py --memory-probe`. Edit that script, not this file.
"""


def memory_probe_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. One training step at each length", """
cmd = [sys.executable, "scripts/memory_probe.py", "--base", BASE_MODEL, "--lengths", *map(str, LENGTHS),
       "--modes", *MODES, "--variants", *VARIANTS, "--out", str(WORK / "memory_probe.json")]
code = subprocess.run(cmd, env=dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")).returncode
print(f"probe exit {code}; done in {(time.time() - START) / 60:.0f} min")
""")]


# ---------------------------------------------------------------------------
# The hierarchical schema (docs/hierarchical_plan.md): one fresh LoRA on two tasks -- the
# module tree of a program, and the wiring graph of one block class variant -- trained on
# both T4s. Training only: scoring is a notebook of its own (and the frozen sets are never
# in this bundle). The memory probe (21 Nov) fit 20,480 tokens at 63% of a T4 once Qwen's
# grouped-query attention is expanded before the kernel (train_lora.attention_without_gqa).

HIER_CODE = ("graphart/hier.py", "graphart/hier_tasks.py", "graphart/reduce.py", "graphart/extract.py",
             "scripts/train_lora.py", *TWO_STEP_CODE)
HIER_TRAIN_SETS = ("hier_train", "hier_val")
HIER_FIELDS = ("id", "source", "family", "kind", "code", "hier")


def hier_payload() -> dict[str, bytes]:
    """Code, prompts and the hierarchical train / val splits -- nothing else: no test, dev,
    holdout or timm-test program travels with a training run."""
    files = {rel: (ROOT / rel).read_bytes() for rel in (*CODE_FILES, *HIER_CODE)}
    for prompt in sorted((ROOT / "prompts").glob("*.txt")):
        files[f"prompts/{prompt.name}"] = prompt.read_bytes()
    for split in HIER_TRAIN_SETS:
        rows = [json.loads(l) for l in (ROOT / "data" / "generated" / f"{split}.jsonl").open(encoding="utf-8")]
        files[f"data/generated/{split}.jsonl"] = "".join(
            json.dumps({k: r[k] for k in HIER_FIELDS}, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    return files


HIER_TRAIN_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-hier"; '
    'EPOCHS = 1.0; GPUS = 2; EFFECTIVE_BATCH = 16; TRAIN_ARGS = ["--hier", "--max-len", "20480", '
    '"--drop-over-max", "--eval-steps", "100"]; FALLBACK_MAX_LEN = 12288; MEMORY_CHECK = 6; SMOKE_STEPS = 8; '
    'EVAL_RESERVE_HOURS = 0.1; TIME_BUDGET_HOURS = 2.5; VALIDATE_ONLY = False')

HIER_TRAIN_NOTES = """
# BASE_MODEL   tuned from scratch (a fresh LoRA), not from any earlier adapter.
# ADAPTER      its name; the adapter lands in lora/<ADAPTER>/ inside the zip.
# EPOCHS       one pass: 1,810 tree + 6,480 wiring examples (9.6M tokens), ~520 steps of 16.
# GPUS, EFFECTIVE_BATCH  both T4s under torchrun, 8 of each step's 16 examples per GPU.
# TRAIN_ARGS   --hier: per program one tree example (hier_tree prompt) and one wiring
#              example per block class variant (hier_wire prompt), on reduced code.
#              --max-len 20480: the longest example is 15,619 tokens, so nothing is left
#              out. Val loss every 100 steps picks the checkpoint.
# FALLBACK_MAX_LEN  if the memory check fails at 20480, train with this instead; the few
#              longer examples (long timm models) are left out, and the log names them.
# MEMORY_CHECK forward + backward on this many of the longest examples, before anything else.
# EVAL_RESERVE_HOURS  kept back at the end for saving and zipping (no scoring here).
# TIME_BUDGET_HOURS  the whole session; the time guard stops training early enough to save
#              the best checkpoint so far.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

HIER_TRAIN_INTRO = """
# graphArt_LM: the hierarchical model (module tree + per-class wiring), two GPUs

Fine-tunes a fresh LoRA on `Qwen2.5-Coder-1.5B-Instruct` for the hierarchical schema: given
reduced PyTorch code, write the **module tree** (tree prompt), and given one block class,
write its **wiring graph** (wiring prompt). One adapter, both tasks. Training data:
generator v3 (1,350 programs), real timm models (177) and small tiers 1-4 programs (283).

Steps: data check -> memory check on the longest examples (falls back to a shorter
`--max-len` if needed) -> 8-step smoke run on both GPUs (must show the loss falling) ->
one epoch on both GPUs. Expected ~1.5 h in all, capped at 2.5 h. No scoring in this
notebook; that is a separate run.

**Before running**: *GPU T4 x2* (both are used), *Internet on*. No inputs needed. Then
*Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_hier_train.zip` from the *Output* tab: the adapter
(`lora/qwen2.5-coder-1.5b-graphart-lora-hier/`, with `train_summary.json`) and the logs.
If the time guard stopped training early, the best checkpoint so far is what was saved.

Generated by `scripts/make_kaggle_notebook.py --hier-train`. Edit that script, not this file.
"""


def hier_train_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    # unpack, install, download, data + memory checks, smoke run, training: the two-GPU
    # notebook's steps 1-6, without its scoring step
    cells = finetune_two_gpu_cells(b64, digest, zip_name="graphart_hier_train.zip")[:6]
    return cells + [("7. Done", """
summary = json.loads((REPO / "lora" / ADAPTER / "train_summary.json").read_text())
print(f"steps {summary['steps']}, {summary['minutes']} min, best val loss {summary['best_eval_loss']}")
for h in summary["eval_history"]:
    print(f"  step {h['step']:4d}  val loss {h['eval_loss']:.4f}")
package()
print(f"done in {(time.time() - START) / 3600:.1f} h -- download {ZIP_NAME}")
""")]


# ---------------------------------------------------------------------------
# Scoring the hierarchical adapter (pipeline/run_hier.py, pipeline/score_hier.py): the dev set
# (51 torchvision programs) and the generator's held-out test (75), plus wiring alone on dev
# (--oracle-tree). Never the frozen sets: they are not in this bundle.

HIER_EVAL_SETS = ("hier_dev", "hier_gen_test")


def hier_eval_payload() -> dict[str, bytes]:
    """All of graphart/ and pipeline/ (small text files; the scorer pulls in several), the
    prompts, and the two development sets with their truth (the runner's --oracle-tree and
    the scorer read it). No frozen set travels."""
    files = {}
    for pkg in ("graphart", "pipeline"):
        for f in sorted((ROOT / pkg).glob("*.py")):
            files[f"{pkg}/{f.name}"] = f.read_bytes()
    for prompt in sorted((ROOT / "prompts").glob("*.txt")):
        files[f"prompts/{prompt.name}"] = prompt.read_bytes()
    for split in HIER_EVAL_SETS:
        files[f"data/generated/{split}.jsonl"] = (ROOT / "data" / "generated" / f"{split}.jsonl").read_bytes()
    return files


HIER_EVAL_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-hier"; '
    'GPUS = 2; JOBS = [[("hier_dev", []), ("hier_dev", ["--oracle-tree"])], [("hier_gen_test", [])]]; '
    'RUN_ARGS = ["--tree-tokens", "8192", "--wire-tokens", "2048"]; TIME_BUDGET_HOURS = 2.5; '
    'PACKAGE_RESERVE_MIN = 6; VALIDATE_ONLY = False')

HIER_EVAL_NOTES = """
# BASE_MODEL   the base the adapter was trained on (its adapter_config.json names it too).
# ADAPTER      the adapter folder to look for under /kaggle/input (from graphart_hier_train.zip:
#              lora/qwen2.5-coder-1.5b-graphart-lora-hier/, uploaded as a dataset). If no folder
#              of that name is found, any single adapter under /kaggle/input is used.
# JOBS         one list per GPU, run in order on that GPU; the GPUs run at the same time.
#              (split, extra flags): hier_dev = the 51 torchvision dev programs, hier_gen_test =
#              75 held-out generator v3 programs, --oracle-tree = wiring alone (truth's tree).
# RUN_ARGS     output caps: the longest true tree is ~6,600 tokens, a wiring graph a few hundred.
# TIME_BUDGET_HOURS  the whole session; jobs still running at the end are stopped, and every
#              program finished so far is kept and scored.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

HIER_EVAL_INTRO = """
# graphArt_LM: scoring the hierarchical model (dev + generator test)

Runs the trained hierarchical adapter through `pipeline/run_hier.py` -- per program one tree
call, then one wiring call per block variant **of the model's own tree**, assembled by code --
and scores it with `pipeline/score_hier.py`: tree, variants, wiring, and the headline, the
flat graph after expansion against torch.fx. Three runs, two GPUs side by side:
- GPU 0: the 51 torchvision **dev** programs, then wiring alone on dev (`--oracle-tree`);
- GPU 1: the 75 held-out **generator v3** programs.
The frozen test sets are not in this notebook. About 1.5 h, capped at 2.5.

**Before running**: *Add Input* -> your dataset holding the adapter folder
`qwen2.5-coder-1.5b-graphart-lora-hier` (from `graphart_hier_train.zip`). *GPU T4 x2*,
*Internet on*. Then *Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_hier_eval.zip` from the *Output* tab: the results
(`results/*.jsonl`, `.meta.json`, `.score.json`), the raw replies, and the logs.

Generated by `scripts/make_kaggle_notebook.py --hier-eval`. Edit that script, not this file.
"""


def hier_eval_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. Find the adapter and the GPUs", """
import zipfile

INPUT = Path(os.environ.get("GRAPHART_INPUT", "/kaggle/input"))
found = [c.parent for c in INPUT.rglob("adapter_config.json")] if INPUT.exists() else []
named = [p for p in found if p.name == ADAPTER]
if len(named) >= 1:
    adapter_src = named[0]
elif len(found) == 1:
    adapter_src = found[0]
else:
    raise SystemExit(f"no adapter folder {ADAPTER!r} under {INPUT} (found: {[str(p) for p in found]}) -- "
                     "add the dataset with the adapter as an input (see the top)")
# copied under its own name, so the results are named after it
adapter = REPO / "lora" / ADAPTER
if not adapter.exists():
    shutil.copytree(adapter_src, adapter)
print("adapter:", adapter_src, "->", adapter, flush=True)
summary = adapter / "train_summary.json"
if summary.exists():
    s = json.loads(summary.read_text())
    print(f"trained {s.get('steps')} steps, best val loss {s.get('best_eval_loss')}", flush=True)

n = int(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                       capture_output=True, text=True).stdout.strip() or 0)
print(f"{n} GPU(s) visible", flush=True)
if n < 1:
    raise SystemExit("no GPU -- set the accelerator to GPU T4 x2")
if n < GPUS:
    print(f"only {n} GPU(s): running every job on GPU 0, one after another", flush=True)
    JOBS = [[job for jobs in JOBS for job in jobs]]
LOGS = REPO / "logs"
LOGS.mkdir(exist_ok=True)
ZIP_NAME = "graphart_hier_eval.zip"

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def package():
    with zipfile.ZipFile(WORK / ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        for root in (REPO / "results", LOGS):
            for p in root.rglob("*") if root.exists() else []:
                if p.is_file():
                    z.write(p, p.relative_to(REPO))
"""), ("5. Run: one job list per GPU, side by side", """
# Each GPU runs its list in one shell process; the lists run at the same time.
ENV = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
procs = []
for gpu, jobs in enumerate(JOBS):
    script = []
    for split, extra in jobs:
        name = split + "".join(x.replace("--", "_") for x in extra)
        cmd = [sys.executable, "-m", "pipeline.run_hier", "--backend", "hf", "--model", str(adapter),
               "--split", split, *RUN_ARGS, *extra]
        script.append((name, cmd))
    log = open(LOGS / f"gpu{gpu}.log", "w")
    runner = ("import subprocess, sys\\n"
              f"for name, cmd in {script!r}:\\n"
              "    print(f'##### {name}', flush=True)\\n"
              "    subprocess.run(cmd)\\n")
    print(f"GPU {gpu}: {[n for n, _ in script]}", flush=True)
    procs.append((gpu, log, subprocess.Popen([sys.executable, "-c", runner], stdout=log, stderr=subprocess.STDOUT,
                                             env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)))))
last = time.time()
while any(p.poll() is None for _, _, p in procs):
    if remaining() < PACKAGE_RESERVE_MIN * 60:
        for gpu, _, p in procs:
            if p.poll() is None:
                p.kill()
                print(f"GPU {gpu}: stopped by the time budget -- finished programs are kept", flush=True)
        break
    time.sleep(30)
    if time.time() - last > 600:            # a progress line every 10 minutes
        last = time.time()
        for gpu, _, _ in procs:
            lines = (LOGS / f"gpu{gpu}.log").read_text(errors="replace").splitlines()
            done = [l for l in lines if l.strip().startswith("[")]
            print(f"  GPU {gpu}: {done[-1].strip() if done else 'starting'}   ({remaining() / 3600:.1f} h left)",
                  flush=True)
for gpu, log, p in procs:
    log.close()
    print(f"\\n##### GPU {gpu} (exit {p.returncode})\\n" +
          "\\n".join((LOGS / f"gpu{gpu}.log").read_text(errors="replace").splitlines()[-25:]), flush=True)
package()
"""), ("6. Score", """
runs = sorted(str(p.relative_to(REPO)) for p in (REPO / "results").glob("hf__*__hier*.jsonl"))
out = subprocess.run([sys.executable, "-m", "pipeline.score_hier", *runs], capture_output=True, text=True)
print(out.stdout[-20000:], out.stderr[-3000:], flush=True)
(LOGS / "scores.txt").write_text(out.stdout)
package()
print(f"done in {(time.time() - START) / 3600:.1f} h -- download {ZIP_NAME}")
""")]


# ---------------------------------------------------------------------------
# The prompted 14B on the hierarchical schema (22 Nov): Qwen2.5-Coder 14B through Ollama, two
# worked examples from the train split, on the 51 dev programs -- the whole answer (tree +
# wiring) and wiring alone given the true tree. One Ollama server per T4. Never the frozen sets.

HIER_14B_CONFIG = (
    'MODEL = "qwen2.5-coder:14b"; SHOTS = 2; NUM_CTX = 20480; FALLBACK_NUM_CTX = 16384; '
    'RUN_ARGS = ["--tree-tokens", "10240", "--wire-tokens", "2048"]; TIME_BUDGET_HOURS = 2.5; '
    'PACKAGE_RESERVE_MIN = 6; VALIDATE_ONLY = False')

HIER_14B_NOTES = """
# MODEL        Ollama tag; q4_K_M, ~9 GB, fits one T4 with its context.
# SHOTS        worked examples before each prompt, from the TRAIN split only, fixed by a rule
#              (pipeline/run_hier.py shots(); saved in prompts/hier_shots2.json).
# NUM_CTX      context per call: the longest dev prompt is ~6k tokens with the examples, and
#              the longest tree answer ~9k. If the model does not fit wholly on one GPU at
#              this size (checked in step 3), FALLBACK_NUM_CTX is used instead.
# RUN_ARGS     output caps: tree 10,240 tokens (densenet121's tree is ~9k), wiring 2,048.
# TIME_BUDGET_HOURS  the whole session; jobs still running at the end are stopped, and every
#              program finished so far is kept and scored.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

HIER_14B_INTRO = """
# graphArt_LM: prompted Qwen2.5-Coder 14B on the hierarchical schema (dev)

No training. Qwen2.5-Coder 14B through Ollama, with two worked examples from the train split,
runs `pipeline/run_hier.py` on the 51 torchvision **dev** programs: per program one tree call,
then one wiring call per block variant of its own tree. Two jobs, one Ollama server per T4:
- the whole answer (tree + wiring), split over both GPUs;
- wiring alone, given the true tree (`--oracle-tree`), on GPU 1.
Scored with `pipeline/score_hier.py` (headline: `hier_right`). The frozen test sets are not
in this notebook. About 1.5-2 h, capped at 2.5.

**Before running**: *GPU T4 x2*, *Internet on* (installs Ollama, downloads the model, ~9 GB).
No inputs. Then *Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_hier_14b.zip` from the *Output* tab: results,
scores, raw replies, logs.

Generated by `scripts/make_kaggle_notebook.py --hier-14b`. Edit that script, not this file.
"""


def hier_14b_payload() -> dict[str, bytes]:
    files = hier_eval_payload()
    files["prompts/hier_shots2.json"] = (ROOT / "prompts" / "hier_shots2.json").read_bytes()
    return files


def hier_14b_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack = code_cells(b64, digest)[0]
    return [unpack, ("2. Install Ollama, one server per GPU, pull the model", """
import zipfile

def up(port) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/version", timeout=2).read()
        return True
    except OSError:
        return False

n_gpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 0)
print(f"{n_gpu} GPU(s)", flush=True)
if n_gpu < 1:
    raise SystemExit("no GPU -- set the accelerator to GPU T4 x2")
PORTS = [11434, 11435][:max(1, min(2, n_gpu))]
if not shutil.which("ollama"):
    subprocess.run("apt-get -qq update > /dev/null && apt-get -qq install -y zstd pciutils lshw > /dev/null",
                   shell=True, check=True)
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
LOGS = REPO / "logs"
LOGS.mkdir(exist_ok=True)
for gpu, port in enumerate(PORTS):
    if up(port):
        continue
    env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_KEEP_ALIVE="24h", OLLAMA_NUM_PARALLEL="1",
               OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0", CUDA_VISIBLE_DEVICES=str(gpu))
    subprocess.Popen(["ollama", "serve"], env=env, stdout=open(LOGS / f"ollama{gpu}.log", "w"),
                     stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(1)
        if up(port):
            break
    else:
        raise RuntimeError(f"ollama on GPU {gpu} did not start -- see logs/ollama{gpu}.log")
for attempt in (1, 2, 3):
    r = subprocess.run(["ollama", "pull", MODEL], env=dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{PORTS[0]}"))
    if r.returncode == 0:
        break
else:
    raise SystemExit(f"could not pull {MODEL}")
print("servers:", PORTS, flush=True)
"""), ("3. Warm up each server; does the model fit on one GPU?", """
def generate(port, prompt, num_ctx, n=16):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0, "num_predict": n, "num_ctx": num_ctx}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())

def on_gpu(port):
    ps = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ps", timeout=10).read())
    m = [x for x in ps.get("models", []) if x.get("name", "").startswith(MODEL.split(":")[0])]
    return bool(m) and m[0].get("size_vram", 0) >= m[0].get("size", 1), m

for port in PORTS:
    t = time.time()
    r = generate(port, "def add(a, b):", NUM_CTX)
    ok, m = on_gpu(port)
    print(f"port {port}: {time.time() - t:.0f}s, reply {r.get('response', '')[:40]!r}, "
          f"all on GPU: {ok}", flush=True)
    if not ok and NUM_CTX != FALLBACK_NUM_CTX:
        print(f"  partly on CPU at num_ctx {NUM_CTX}: using {FALLBACK_NUM_CTX}", flush=True)
        NUM_CTX = FALLBACK_NUM_CTX
        generate(port, "def add(a, b):", NUM_CTX)
        print("  now all on GPU:", on_gpu(port)[0], flush=True)
# speed, on a real-size prompt: tokens per second of generation
r = generate(PORTS[0], "Write a long Python class for a ResNet.", NUM_CTX, n=200)
print(f"generation {r['eval_count'] / (r['eval_duration'] / 1e9):.1f} tok/s, "
      f"prompt {r['prompt_eval_count'] / max(r['prompt_eval_duration'], 1) * 1e9:.0f} tok/s", flush=True)
subprocess.run("nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv", shell=True)
"""), ("4. Run: the whole answer split over the GPUs, wiring alone on the second", """
from graphart.hier import to_text
rows = [json.loads(l) for l in open(REPO / "data/generated/hier_dev.jsonl", encoding="utf-8")]
ntags = {r["id"]: sum(1 for v in r["hier"]["variants"].values() if not v["hidden"]) for r in rows}
work = {r["id"]: len(to_text(r["hier"])) + 600 * ntags[r["id"]] for r in rows}   # ~ output to write
# GPU 1 also runs wiring alone; balance the whole-answer programs around it, longest first
load = [0, sum(600 * n for n in ntags.values())] if len(PORTS) > 1 else [0]
part = {}
for rid in sorted(work, key=lambda k: -work[k]):
    g = min(range(len(load)), key=lambda i: load[i])
    part[rid] = g
    load[g] += work[rid]
COMMON = ["--backend", "ollama", "--model", MODEL, "--split", "hier_dev", "--shots", str(SHOTS),
          "--num-ctx", str(NUM_CTX), *RUN_ARGS]
JOBS = [[] for _ in PORTS]
for g in range(len(PORTS)):
    ids = ",".join(r["id"] for r in rows if part[r["id"]] == g)
    JOBS[g].append(("whole", COMMON + ["--ids", ids, "--part", "ab"[g]]))
JOBS[-1].insert(0, ("oracle_tree", COMMON + ["--oracle-tree"]))
print({g: [n for n, _ in j] for g, j in enumerate(JOBS)}, "| est. load", load, flush=True)

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

ZIP_NAME = "graphart_hier_14b.zip"

def package():
    with zipfile.ZipFile(WORK / ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        for root in (REPO / "results", LOGS):
            for p in root.rglob("*") if root.exists() else []:
                if p.is_file():
                    z.write(p, p.relative_to(REPO))

procs = []
for g, jobs in enumerate(JOBS):
    script = [(name, [sys.executable, "-m", "pipeline.run_hier", *args]) for name, args in jobs]
    runner = ("import subprocess, sys\\n"
              f"for name, cmd in {script!r}:\\n"
              "    print(f'##### {name}', flush=True)\\n"
              "    subprocess.run(cmd)\\n")
    log = open(LOGS / f"gpu{g}.log", "w")
    env = dict(os.environ, PYTHONUNBUFFERED="1", OLLAMA_URL=f"http://127.0.0.1:{PORTS[g]}/api/generate")
    procs.append((g, log, subprocess.Popen([sys.executable, "-c", runner], stdout=log,
                                           stderr=subprocess.STDOUT, env=env)))
last = time.time()
while any(p.poll() is None for _, _, p in procs):
    if remaining() < PACKAGE_RESERVE_MIN * 60:
        for g, _, p in procs:
            if p.poll() is None:
                p.kill()
                print(f"GPU {g}: stopped by the time budget -- finished programs are kept", flush=True)
        break
    time.sleep(30)
    if time.time() - last > 600:
        last = time.time()
        for g, _, _ in procs:
            done = [l for l in (LOGS / f"gpu{g}.log").read_text(errors="replace").splitlines()
                    if l.strip().startswith("[")]
            print(f"  GPU {g}: {done[-1].strip() if done else 'starting'}   ({remaining() / 3600:.1f} h left)",
                  flush=True)
for g, log, p in procs:
    log.close()
    print(f"\\n##### GPU {g} (exit {p.returncode})\\n" +
          "\\n".join((LOGS / f"gpu{g}.log").read_text(errors="replace").splitlines()[-20:]), flush=True)
package()
"""), ("5. Merge the two halves, score", """
res = REPO / "results"
for parts in {tuple(sorted(res.glob(p.name.split("__part")[0] + "__part*.jsonl")))
              for p in res.glob("*__part*.jsonl")}:
    base = parts[0].name.split("__part")[0]
    order = {r["id"]: i for i, r in enumerate(rows)}
    recs = sorted((json.loads(l) for p in parts for l in p.open(encoding="utf-8")), key=lambda r: order[r["id"]])
    (res / f"{base}.jsonl").write_text("".join(json.dumps(r) + "\\n" for r in recs), encoding="utf-8")
    metas = [json.loads(p.with_name(p.stem + ".meta.json").read_text()) for p in parts]
    meta = dict(metas[0], n=len(recs), of=len(rows), failed=sum(m["failed"] for m in metas),
                calls=sum(m["calls"] for m in metas), parts=[p.name for p in parts])
    (res / f"{base}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"merged {len(parts)} parts -> {base}.jsonl ({len(recs)} programs)", flush=True)
runs = sorted(str(p.relative_to(REPO)) for p in res.glob("ollama__*__hier*.jsonl") if "__part" not in p.name)
out = subprocess.run([sys.executable, "-m", "pipeline.score_hier", *runs], capture_output=True, text=True)
print(out.stdout[-20000:], out.stderr[-3000:], flush=True)
(LOGS / "scores.txt").write_text(out.stdout)
package()
print(f"done in {(time.time() - START) / 3600:.1f} h -- download {ZIP_NAME}")
""")]


# ---------------------------------------------------------------------------
# Per-instance trees (22 Nov, graphart/hier_inst.py): a fresh LoRA that writes ONE class
# instance's contents given its constructor arguments. Training only; the wiring stays with
# the hier adapter, and scoring is a separate notebook.

HIER_INST_SETS = ("hier_inst_train", "hier_inst_val")
HIER_INST_FIELDS = ("id", "program", "user", "answer")


def hier_inst_payload() -> dict[str, bytes]:
    """Code, prompts and the per-instance train / val examples (built locally by
    scripts/build_hier_inst.py from hier_train / hier_val) -- nothing else."""
    files = {rel: (ROOT / rel).read_bytes() for rel in (*CODE_FILES, *HIER_CODE, "graphart/hier_inst.py")}
    for prompt in sorted((ROOT / "prompts").glob("*.txt")):
        files[f"prompts/{prompt.name}"] = prompt.read_bytes()
    for split in HIER_INST_SETS:
        rows = [json.loads(l) for l in (ROOT / "data" / "generated" / f"{split}.jsonl").open(encoding="utf-8")]
        files[f"data/generated/{split}.jsonl"] = "".join(
            json.dumps({k: r[k] for k in HIER_INST_FIELDS}, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    return files


HIER_INST_TRAIN_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-hier-inst"; '
    'EPOCHS = 1.0; GPUS = 2; EFFECTIVE_BATCH = 16; TRAIN_ARGS = ["--hier-inst", "--hier-inst-max", "12", '
    '"--max-len", "16384", "--drop-over-max", "--eval-steps", "100"]; FALLBACK_MAX_LEN = 12288; MEMORY_CHECK = 6; SMOKE_STEPS = 8; '
    'EVAL_RESERVE_HOURS = 0.1; TIME_BUDGET_HOURS = 2.5; VALIDATE_ONLY = False')

HIER_INST_TRAIN_NOTES = """
# BASE_MODEL   tuned from scratch (a fresh LoRA), not from any earlier adapter.
# ADAPTER      its name; the adapter lands in lora/<ADAPTER>/ inside the zip.
# EPOCHS       one pass over the per-instance examples (__N_TRAIN__ train, __N_VAL__ val, ~8.6M
#              tokens, ~520 steps of 16).
# GPUS, EFFECTIVE_BATCH  both T4s under torchrun, 8 of each step's 16 examples per GPU.
# TRAIN_ARGS   --hier-inst: one example per distinct class instance of each training program
#              (hier_inst prompt): the root gets the whole reduced program, every other
#              instance its class's code and its constructor arguments; the answer is that
#              instance's children. --hier-inst-max 12: per program the root and 11 instances
#              spread over the rest (of 17,044 in all), to fit the session. Val loss every
#              100 steps picks the checkpoint.
#              --max-len 16384 covers every example (longest __MAX_TOK__ tokens).
# FALLBACK_MAX_LEN  if the memory check fails, train with this instead; longer examples are
#              left out, and the log names them.
# MEMORY_CHECK forward + backward on this many of the longest examples, before anything else.
# EVAL_RESERVE_HOURS  kept back at the end for saving and zipping (no scoring here).
# TIME_BUDGET_HOURS  the whole session; the time guard stops training early enough to save
#              the best checkpoint so far.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

HIER_INST_TRAIN_INTRO = """
# graphArt_LM: per-instance module trees, two GPUs

Fine-tunes a fresh LoRA on `Qwen2.5-Coder-1.5B-Instruct` for one step of the hierarchical
schema, split up: instead of the whole module tree at once, the model writes **one class
instance at a time** -- given the class's code and the arguments its constructor receives,
its child modules, with child instances of classes in the code written as
`name: Class(arg=value, ...)` (each gets its own call later). The root call gets the whole
program. Code joins the answers into the tree. Training data: the same 1,810 training
programs as the hierarchical adapter (generator v3, timm, tiers 1-4), one example per
distinct instance.

Steps: data check -> memory check on the longest examples -> 8-step smoke run on both GPUs
(must show the loss falling) -> one epoch on both GPUs. Expected ~__HOURS__ h, capped at 2.5 h.
No scoring in this notebook.

**Before running**: *GPU T4 x2* (both are used), *Internet on*. No inputs needed. Then
*Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_hier_inst_train.zip` from the *Output* tab: the
adapter (`lora/qwen2.5-coder-1.5b-graphart-lora-hier-inst/`, with `train_summary.json`) and
the logs. If the time guard stopped training early, the best checkpoint so far is saved.

Generated by `scripts/make_kaggle_notebook.py --hier-inst-train`. Edit that script, not this file.
"""


def hier_inst_train_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    cells = finetune_two_gpu_cells(b64, digest, zip_name="graphart_hier_inst_train.zip")[:6]
    return cells + [("7. Done", """
summary = json.loads((REPO / "lora" / ADAPTER / "train_summary.json").read_text())
print(f"steps {summary['steps']}, {summary['minutes']} min, best val loss {summary['best_eval_loss']}")
for h in summary["eval_history"]:
    print(f"  step {h['step']:4d}  val loss {h['eval_loss']:.4f}")
package()
print(f"done in {(time.time() - START) / 3600:.1f} h -- download {ZIP_NAME}")
""")]


# ---------------------------------------------------------------------------
# Scoring the per-instance trees (pipeline/run_hier_inst.py): the instance adapter builds the
# tree one class instance at a time, the hier adapter wires it. dev first, then gen_test, each
# split over both GPUs. Never the frozen sets.

HIER_INST_EVAL_CONFIG = (
    'BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"; TREE_ADAPTER ="qwen2.5-coder-1.5b-graphart-lora-hier-inst"; '
    'WIRE_ADAPTER = "qwen2.5-coder-1.5b-graphart-lora-hier"; SPLITS = ["hier_dev", "hier_gen_test"]; '
    'RUN_ARGS = ["--inst-tokens", "2048", "--wire-tokens", "2048", "--max-calls", "300"]; '
    'TIME_BUDGET_HOURS = 2.5; PACKAGE_RESERVE_MIN = 6; VALIDATE_ONLY = False')

HIER_INST_EVAL_NOTES = """
# BASE_MODEL   the base both adapters were tuned from (checked against their adapter_config.json).
# TREE_ADAPTER the per-instance adapter (graphart_hier_inst_train.zip: lora/<this name>/).
# WIRE_ADAPTER the hierarchical adapter that wires (graphart_hier_train.zip: lora/<this name>/).
#              Both are found by folder name anywhere under /kaggle/input.
# SPLITS       in order: the 51 dev programs (the criterion, P2a), then the 75 generator test
#              programs; each split is divided between the two GPUs.
# RUN_ARGS     output caps per call; at most 300 instance calls per program.
# TIME_BUDGET_HOURS  the whole session; jobs still running at the end are stopped, and every
#              program finished so far is kept and scored.
# VALIDATE_ONLY True: only check the code arrived intact (works on a CPU session).
"""

HIER_INST_EVAL_INTRO = """
# graphArt_LM: scoring the per-instance trees (dev + generator test)

`pipeline/run_hier_inst.py`: the **instance adapter** writes the module tree one class
instance at a time (the root from the whole program, every other instance from its class's
code and constructor arguments), code joins the answers; the **hierarchical adapter** then
wires every distinct block instance, and instances whose graphs match share a variant tag.
Scored by `pipeline/score_hier.py` (headline `hier_right`). Dev first (51), then the
generator test (75), each split over both T4s. About 2 h, capped at 2.5 h.

**Before running**: *Add Input* -> both training outputs: `graphart_hier_inst_train`
(folder `qwen2.5-coder-1.5b-graphart-lora-hier-inst`) and `graphreArt_heir_train` (folder
`qwen2.5-coder-1.5b-graphart-lora-hier`), as notebook outputs or uploaded datasets.
*GPU T4 x2*, *Internet on*. Then *Save Version -> Save & Run All (Commit)*.

**When it finishes**, download `graphart_hier_inst_eval.zip`: results, scores, raw replies, logs.

Generated by `scripts/make_kaggle_notebook.py --hier-inst-eval`. Edit that script, not this file.
"""


def hier_inst_eval_cells(b64: str, digest: str) -> list[tuple[str, str]]:
    unpack, install, download = finetune_cells(b64, digest)[:3]
    return [unpack, install, download, ("4. Find the two adapters and the GPUs", """
import re, zipfile

INPUT = Path(os.environ.get("GRAPHART_INPUT", "/kaggle/input"))
found = [c.parent for c in INPUT.rglob("adapter_config.json")] if INPUT.exists() else []
ADAPTERS = {}
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())     # a dataset mounts under its slug: dots become dashes
for name in (TREE_ADAPTER, WIRE_ADAPTER):
    hits = [p for p in found if "checkpoint" not in str(p)
            and norm(name) in {norm(part) for part in p.parts[-4:]}]
    if not hits:
        raise SystemExit(f"no adapter folder {name!r} under {INPUT} (found: {[str(p) for p in found]}) -- "
                         "add both training outputs as inputs (see the top)")
    base = json.loads((hits[0] / "adapter_config.json").read_text()).get("base_model_name_or_path")
    if base != BASE_MODEL:
        raise SystemExit(f"{name} was tuned from {base}, not {BASE_MODEL}")
    ADAPTERS[name] = REPO / "lora" / name
    if not ADAPTERS[name].exists():
        shutil.copytree(hits[0], ADAPTERS[name])
    print(name, "<-", hits[0], flush=True)

n = int(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                       capture_output=True, text=True).stdout.strip() or 0)
print(f"{n} GPU(s) visible", flush=True)
if n < 1:
    raise SystemExit("no GPU -- set the accelerator to GPU T4 x2")
GPUS = min(n, 2)
LOGS = REPO / "logs"
LOGS.mkdir(exist_ok=True)
ZIP_NAME = "graphart_hier_inst_eval.zip"

def remaining():
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def package():
    with zipfile.ZipFile(WORK / ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        for root in (REPO / "results", LOGS):
            for p in root.rglob("*") if root.exists() else []:
                if p.is_file():
                    z.write(p, p.relative_to(REPO))
"""), ("5. Run: each split divided between the GPUs, dev first", """
ENV = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
JOBS = [[] for _ in range(GPUS)]
for split in SPLITS:
    rows = [json.loads(l) for l in open(REPO / f"data/generated/{split}.jsonl", encoding="utf-8")]
    for g in range(GPUS):
        ids = ",".join(r["id"] for k, r in enumerate(rows) if k % GPUS == g)
        JOBS[g].append((f"{split} part {'ab'[g]}",
                        [sys.executable, "-m", "pipeline.run_hier_inst", "--tree-model", str(ADAPTERS[TREE_ADAPTER]),
                         "--wire-model", str(ADAPTERS[WIRE_ADAPTER]), "--split", split, "--ids", ids,
                         "--part", "ab"[g], *RUN_ARGS]))
procs = []
for g, jobs in enumerate(JOBS):
    runner = ("import subprocess, sys\\n"
              f"for name, cmd in {jobs!r}:\\n"
              "    print(f'##### {name}', flush=True)\\n"
              "    subprocess.run(cmd)\\n")
    log = open(LOGS / f"gpu{g}.log", "w")
    print(f"GPU {g}: {[name for name, _ in jobs]}", flush=True)
    procs.append((g, log, subprocess.Popen([sys.executable, "-c", runner], stdout=log, stderr=subprocess.STDOUT,
                                           env=dict(ENV, CUDA_VISIBLE_DEVICES=str(g)))))
last = time.time()
while any(p.poll() is None for _, _, p in procs):
    if remaining() < PACKAGE_RESERVE_MIN * 60:
        for g, _, p in procs:
            if p.poll() is None:
                p.kill()
                print(f"GPU {g}: stopped by the time budget -- finished programs are kept", flush=True)
        break
    time.sleep(30)
    if time.time() - last > 600:
        last = time.time()
        for g, _, _ in procs:
            done = [l for l in (LOGS / f"gpu{g}.log").read_text(errors="replace").splitlines()
                    if l.strip().startswith("[") or l.startswith("#####")]
            print(f"  GPU {g}: {done[-1].strip() if done else 'starting'}   ({remaining() / 3600:.1f} h left)",
                  flush=True)
for g, log, p in procs:
    log.close()
    print(f"\\n##### GPU {g} (exit {p.returncode})\\n" +
          "\\n".join((LOGS / f"gpu{g}.log").read_text(errors="replace").splitlines()[-20:]), flush=True)
package()
"""), ("6. Merge the halves, score", """
res = REPO / "results"
for parts in {tuple(sorted(res.glob(p.name.split("__part")[0] + "__part*.jsonl")))
              for p in res.glob("*__part*.jsonl")}:
    base = parts[0].name.split("__part")[0]
    split = base.split("__")[-1]
    order = {json.loads(l)["id"]: i for i, l in enumerate(open(REPO / f"data/generated/{split}.jsonl", encoding="utf-8"))}
    recs = sorted((json.loads(l) for p in parts for l in p.open(encoding="utf-8")), key=lambda r: order[r["id"]])
    (res / f"{base}.jsonl").write_text("".join(json.dumps(r) + "\\n" for r in recs), encoding="utf-8")
    metas = [json.loads(p.with_name(p.stem + ".meta.json").read_text()) for p in parts]
    meta = dict(metas[0], n=len(recs), of=len(order), failed=sum(m["failed"] for m in metas),
                parts=[p.name for p in parts])
    (res / f"{base}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"merged {len(parts)} parts -> {base}.jsonl ({len(recs)} of {len(order)} programs)", flush=True)
runs = sorted(str(p.relative_to(REPO)) for p in res.glob("hf__*__hier_inst__*.jsonl") if "__part" not in p.name)
out = subprocess.run([sys.executable, "-m", "pipeline.score_hier", *runs], capture_output=True, text=True)
print(out.stdout[-20000:], out.stderr[-3000:], flush=True)
(LOGS / "scores.txt").write_text(out.stdout)
package()
print(f"done in {(time.time() - START) / 3600:.1f} h -- download {ZIP_NAME}")
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
    ap.add_argument("--finetune-two-step", action="store_true",
                    help="build notebooks/kaggle_finetune_two_step.ipynb (fork B: the two-step 1.5B)")
    ap.add_argument("--finetune-two-gpu", action="store_true",
                    help="build notebooks/kaggle_finetune_two_gpu.ipynb (the full-rules two-step run, "
                         "a full epoch on both T4s)")
    ap.add_argument("--two-gpu-smoke", action="store_true",
                    help="build notebooks/kaggle_two_gpu_smoke.ipynb (does training on both T4s work?)")
    ap.add_argument("--finetune-two-step-full", action="store_true",
                    help="build notebooks/kaggle_finetune_two_step_full.ipynb (two-step, full-rules prompts)")
    ap.add_argument("--two-step-eval", action="store_true",
                    help="build notebooks/kaggle_two_step_eval.ipynb (the two-step adapter given mixed-v2's nodes, "
                         "and re-prompted with the full rules; no training)")
    ap.add_argument("--probe-14b-nodes", action="store_true",
                    help="build notebooks/kaggle_probe_14b_nodes.ipynb (14B: CoS edges given nodes, better nodes)")
    ap.add_argument("--edits-two-adapter", action="store_true",
                    help="build notebooks/kaggle_edits_two_adapter.ipynb (E2: the edit set on the mixed-v2 "
                         "and brief two-step adapters, one per GPU)")
    ap.add_argument("--probe-14b-compose", action="store_true",
                    help="build notebooks/kaggle_probe_14b_compose.ipynb (E1: 14B CoS wiring on mixed-v2's "
                         "nodes; E3: on the true nodes reordered)")
    ap.add_argument("--holdout", action="store_true",
                    help="build notebooks/kaggle_holdout.ipynb (the frozen holdout, scored once: the final "
                         "pipeline and the submitted adapter)")
    ap.add_argument("--e4-wirer", action="store_true",
                    help="build notebooks/kaggle_e4_wirer.ipynb (E4: train a 1.5B step-2 wirer on both T4s, "
                         "score it on mixed-v2's, true and reordered node lists)")
    ap.add_argument("--probe-14b", action="store_true",
                    help="build notebooks/kaggle_probe_14b.ipynb (14B edges_only, nodes_only, cos on torchvision)")
    ap.add_argument("--hier-eval", action="store_true",
                    help="build notebooks/kaggle_hier_eval.ipynb (score the hierarchical adapter on hier_dev "
                         "and hier_gen_test; the adapter is an uploaded input)")
    ap.add_argument("--hier-train", action="store_true",
                    help="build notebooks/kaggle_hier_train.ipynb (the hierarchical-schema adapter: tree + "
                         "wiring, one fresh LoRA, both T4s; training only)")
    ap.add_argument("--hier-inst-train", action="store_true",
                    help="build notebooks/kaggle_hier_inst_train.ipynb (per-instance trees: a fresh LoRA, "
                         "both T4s; training only)")
    ap.add_argument("--hier-inst-eval", action="store_true",
                    help="build notebooks/kaggle_hier_inst_eval.ipynb (per-instance trees + hier wiring on "
                         "hier_dev and hier_gen_test; both adapters are inputs)")
    ap.add_argument("--hier-14b", action="store_true",
                    help="build notebooks/kaggle_hier_14b.ipynb (prompted Qwen2.5-Coder 14B on hier_dev: whole "
                         "answer and wiring alone; Ollama, one server per T4)")
    ap.add_argument("--memory-probe", action="store_true",
                    help="build notebooks/kaggle_memory_probe.ipynb (how long a training example fits "
                         "on the free T4s; no training)")
    args = ap.parse_args()
    unknown = [p for p in args.prompts if not (ROOT / "prompts" / f"{p}.txt").exists()]
    if unknown:
        raise SystemExit(f"no such prompt(s): {unknown}")
    OUT_DIR.mkdir(exist_ok=True)

    if args.hier_train or args.hier_eval or args.hier_14b or args.hier_inst_train or args.hier_inst_eval:
        if args.hier_inst_eval:
            files = hier_eval_payload()
            b64, digest = bundle(files)
            nb_path = OUT_DIR / "kaggle_hier_inst_eval.ipynb"
            nb = notebook(HIER_INST_EVAL_CONFIG, hier_inst_eval_cells(b64, digest), 0,
                          intro=HIER_INST_EVAL_INTRO, notes=HIER_INST_EVAL_NOTES)
        elif args.hier_inst_train:
            files = hier_inst_payload()
            b64, digest = bundle(files)
            nb_path = OUT_DIR / "kaggle_hier_inst_train.ipynb"
            stats = json.loads((ROOT / "data" / "generated" / "hier_inst_stats.json").read_text())
            fill = lambda t: (t.replace("__N_TRAIN__", f"{stats['train']:,}").replace("__N_VAL__", f"{stats['val']:,}")
                              .replace("__MAX_TOK__", f"{stats['max_tokens']:,}").replace("__HOURS__", stats["hours"]))
            nb = notebook(HIER_INST_TRAIN_CONFIG, hier_inst_train_cells(b64, digest), 0,
                          intro=fill(HIER_INST_TRAIN_INTRO), notes=fill(HIER_INST_TRAIN_NOTES))
        elif args.hier_14b:
            files = hier_14b_payload()
            b64, digest = bundle(files)
            nb_path = OUT_DIR / "kaggle_hier_14b.ipynb"
            nb = notebook(HIER_14B_CONFIG, hier_14b_cells(b64, digest), 0,
                          intro=HIER_14B_INTRO, notes=HIER_14B_NOTES)
        elif args.hier_train:
            files = hier_payload()
            b64, digest = bundle(files)
            nb_path = OUT_DIR / "kaggle_hier_train.ipynb"
            nb = notebook(HIER_TRAIN_CONFIG, hier_train_cells(b64, digest), 0,
                          intro=HIER_TRAIN_INTRO, notes=HIER_TRAIN_NOTES)
        else:
            files = hier_eval_payload()
            b64, digest = bundle(files)
            nb_path = OUT_DIR / "kaggle_hier_eval.ipynb"
            nb = notebook(HIER_EVAL_CONFIG, hier_eval_cells(b64, digest), 0,
                          intro=HIER_EVAL_INTRO, notes=HIER_EVAL_NOTES)
        text = json.dumps(nb, indent=1) + "\n"
        nb_path.write_text(text, encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} ({len(text.encode()) / 1e6:.2f} MB; bundle {len(b64) / 1024:.0f} KB, "
              f"sha256 {digest[:12]})")
        if len(text.encode()) > 1_000_000:
            print("WARNING: over Kaggle's 1 MB limit on notebook source")
        return

    if args.memory_probe:
        # code only -- the probe uses random tokens, so no data set travels with it
        files, n = {}, 0
        for rel in (*CODE_FILES, "scripts/train_lora.py", "scripts/memory_probe.py", *TWO_STEP_CODE):
            files[rel] = (ROOT / rel).read_bytes()        # train_lora imports pipeline and graphart.two_step
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_memory_probe.ipynb"
        nb = notebook(MEMORY_PROBE_CONFIG, memory_probe_cells(b64, digest), n,
                      intro=MEMORY_PROBE_INTRO, notes=MEMORY_PROBE_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

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

    if args.finetune_two_step:
        files, n = payload(two_step=True)
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_finetune_two_step.ipynb"
        cells = finetune_large_cells(b64, digest, zip_name="graphart_finetune_two_step.zip",
                                     runner="pipeline.run_two_step", prompt_args='"--brief"')
        nb = notebook(FINETUNE_TWO_STEP_CONFIG, cells, n,
                      intro=FINETUNE_TWO_STEP_INTRO, notes=FINETUNE_TWO_STEP_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.finetune_two_gpu:
        files, n = payload(two_step=True)
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_finetune_two_gpu.ipynb"
        nb = notebook(FINETUNE_TWO_GPU_CONFIG, finetune_two_gpu_cells(b64, digest), n,
                      intro=FINETUNE_TWO_GPU_INTRO, notes=FINETUNE_TWO_GPU_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.two_gpu_smoke:
        files, n = payload(two_step=True)
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_two_gpu_smoke.ipynb"
        nb = notebook(TWO_GPU_SMOKE_CONFIG, two_gpu_smoke_cells(b64, digest), n,
                      intro=TWO_GPU_SMOKE_INTRO, notes=TWO_GPU_SMOKE_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.finetune_two_step_full:
        files, n = payload(two_step=True)
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_finetune_two_step_full.ipynb"
        cells = finetune_large_cells(b64, digest, zip_name="graphart_finetune_two_step_full.zip",
                                     runner="pipeline.run_two_step", prompt_args="*[]")
        nb = notebook(FINETUNE_TWO_STEP_FULL_CONFIG, cells, n,
                      intro=FINETUNE_TWO_STEP_FULL_INTRO, notes=FINETUNE_TWO_STEP_FULL_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.two_step_eval:
        files, n = payload()
        files["pipeline/run_two_step.py"] = (ROOT / "pipeline" / "run_two_step.py").read_bytes()
        files[MIXED_V2_TV_BUNDLED] = (ROOT / MIXED_V2_TV).read_bytes()      # the node lists step 2 is handed
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_two_step_eval.ipynb"
        nb = notebook(TWO_STEP_EVAL_CONFIG, two_step_eval_cells(b64, digest), n,
                      intro=TWO_STEP_EVAL_INTRO, notes=TWO_STEP_EVAL_NOTES)
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

    if args.edits_two_adapter:
        files, n = payload()
        for rel in TWO_STEP_CODE:                      # the two-step runner and its reader
            files[rel] = (ROOT / rel).read_bytes()
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_edits_two_adapter.ipynb"
        nb = notebook(EDITS_TWO_ADAPTER_CONFIG, edits_two_adapter_cells(b64, digest), n,
                      intro=EDITS_TWO_ADAPTER_INTRO, notes=EDITS_TWO_ADAPTER_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.holdout:
        files, n = payload()
        for rel in TWO_STEP_CODE:                      # the wirer's runner and its reader
            files[rel] = (ROOT / rel).read_bytes()
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_holdout.ipynb"
        nb = notebook(HOLDOUT_CONFIG, holdout_cells(b64, digest), n, intro=HOLDOUT_INTRO, notes=HOLDOUT_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.e4_wirer:
        files, n = payload(two_step=True)
        files[MIXED_V2_TV_BUNDLED] = (ROOT / MIXED_V2_TV).read_bytes()
        files[MIXED_V2_TEST_BUNDLED] = (ROOT / MIXED_V2_TEST).read_bytes()
        files.update(true_node_file())
        files.update(shuffled_node_files())
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_e4_wirer.ipynb"
        nb = notebook(E4_WIRER_CONFIG, finetune_two_gpu_cells(b64, digest, zip_name="graphart_e4_wirer.zip"), n,
                      intro=E4_WIRER_INTRO, notes=E4_WIRER_NOTES)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} (bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.probe_14b_compose:
        settings = config(PROBE_14B_COMPOSE_JOBS, hours=2.5)
        files, n = payload()
        files[MIXED_V2_TV_BUNDLED] = (ROOT / MIXED_V2_TV).read_bytes()      # E1's node lists
        files.update(shuffled_node_files())                                  # E3's
        b64, digest = bundle(files)
        nb_path = OUT_DIR / "kaggle_probe_14b_compose.ipynb"
        nb = notebook(settings, code_cells(b64, digest), n, intro=PROBE_14B_COMPOSE_INTRO)
        nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} ({len(PROBE_14B_COMPOSE_JOBS)} jobs, "
              f"bundle {len(b64) / 1024:.0f} KB, sha256 {digest[:12]})")
        return

    if args.closing_ollama or args.probe_14b or args.probe_14b_nodes:
        jobs = PROBE_14B_NODES_JOBS if args.probe_14b_nodes else PROBE_14B_JOBS if args.probe_14b else CLOSING_OLLAMA_JOBS
        settings = config(jobs, hours=2.5)          # the user's cap
        files, n = payload()
        b64, digest = bundle(files)
        nb_path = OUT_DIR / ("kaggle_probe_14b_nodes.ipynb" if args.probe_14b_nodes else
                             "kaggle_probe_14b.ipynb" if args.probe_14b else "kaggle_closing_ollama.ipynb")
        nb_path.write_text(json.dumps(notebook(settings, code_cells(b64, digest), n), indent=1) + "\n",
                           encoding="utf-8")
        print(f"wrote {nb_path.relative_to(ROOT)} ({len(jobs)} jobs, "
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

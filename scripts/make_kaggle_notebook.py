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
import gzip
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


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime = len(data), 0          # mtime 0: same inputs, same bundle
    tar.addfile(info, io.BytesIO(data))


def payload() -> tuple[dict[str, bytes], int]:
    """Every file the runner needs on Kaggle, by repo-relative path, and the sample count."""
    # Only the stratified samples, in split order: --stratify on this file then
    # picks exactly these, the same ones every other run used. The reference
    # edges are dropped -- nothing on Kaggle reads them (edges_only is handed the
    # nodes, not the edges), and leaving them out keeps the answers off the box.
    rows = load_split("test", None, None, STRATIFY)
    samples = "".join(json.dumps({**{k: r[k] for k in SAMPLE_FIELDS},
                                  "target": r["target"].split("EDGES")[0].strip()}) + "\n"
                      for r in rows)
    files = {rel: (ROOT / rel).read_bytes() for rel in CODE_FILES}
    for prompt in sorted((ROOT / "prompts").glob("*.txt")):
        files[f"prompts/{prompt.name}"] = prompt.read_bytes()
    files["data/generated/test.jsonl"] = samples.encode("utf-8")
    return files, len(rows)


def _tar(files: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in files.items():
            _add(tar, name, data)
    return raw.getvalue()


def bundle(files: dict[str, bytes]) -> tuple[str, str]:
    """Base64 tar.gz of the files, for the .ipynb's unpack cell, and its sha256."""
    # gzip stamps the current time into its header unless told otherwise.
    data = gzip.compress(_tar(files), compresslevel=9, mtime=0)
    return base64.b64encode(data).decode("ascii"), hashlib.sha256(data).hexdigest()


def _chunks(b64: str, width: int = 100) -> str:
    return "\n".join(f'    "{b64[i:i + width]}"' for i in range(0, len(b64), width))


DEFAULT_MODELS = ["qwen2.5-coder:14b"]
# Highest value first, so a session that runs out of time loses the least.
DEFAULT_PROMPTS = ["zero_shot", "nodes_only", "edges_only", "cot", "cos", "few_shot"]


def config(models: list[str], prompts: list[str]) -> str:
    """The settings cell, kept on one line so it can be typed into the editor as is."""
    return (f"MODELS = {json.dumps(models)}; PROMPTS = {json.dumps(prompts)}; "
            'MAX_TOKENS = {"cot": 4096, "cos": 4096}; TIME_BUDGET_HOURS = 11.0; SMOKE_ONLY = False; '
            "VALIDATE_ONLY = False")

CONFIG_NOTES = """
# MODELS       one per session is the safe default. qwen2.5-coder:14b fits one T4 and
#              should take ~3 h for all six prompts; "gemma4:31b" needs both T4s and
#              roughly 6-9 h, and a session is capped at 12.
# PROMPTS      highest value first, so a session that runs out of time loses the least.
# MAX_TOKENS   cot and cos write their working before the graph; the runner's default
#              cap (1500 tokens) would cut them off mid-thought.
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
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
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
    # cot and cos write roughly 4x as much, so a zero_shot sample is a lower bound.
    print(f"rough projection for {model}: {took * 40 * (len(PROMPTS) + 3 * sum(p in ('cot', 'cos') for p in PROMPTS)) / 3600:.1f} h")
"""),
        ("4. The run", """
def remaining() -> float:
    return TIME_BUDGET_HOURS * 3600 - (time.time() - START)

def package() -> None:
    # Re-zipped after every prompt, so a session that dies late keeps what it finished.
    if (REPO / "results").exists():
        shutil.make_archive(str(WORK / "graphart_results"), "zip", REPO, "results")

for model in ([] if SMOKE_ONLY else MODELS):
    for prompt in PROMPTS:
        if remaining() < 600:
            print(f"time budget spent -- skipping {model} / {prompt}")
            continue
        cmd = [sys.executable, "-m", "pipeline.run_prompting", "--backend", "ollama",
               "--model", model, "--prompt", prompt, "--stratify", "10"]
        if prompt in MAX_TOKENS:
            cmd += ["--max-tokens", str(MAX_TOKENS[prompt])]
        print(f"\\n##### {model} / {prompt}   ({remaining() / 3600:.1f} h left)", flush=True)
        try:
            # Cached replies survive a timeout; a later session would only redo the rest.
            subprocess.run(cmd, check=False, timeout=remaining())
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


def notebook(settings: str, cells: list[tuple[str, str]], n: int) -> dict:
    out = [_md(INTRO.format(n=n)), _code(CONFIG_NOTES.strip("\n") + "\n" + settings)]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="Ollama model tags, one session's worth (default: %(default)s)")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                    help="prompts to run, in order (default: all six)")
    args = ap.parse_args()
    unknown = [p for p in args.prompts if not (ROOT / "prompts" / f"{p}.txt").exists()]
    if unknown:
        raise SystemExit(f"no such prompt(s): {unknown}")

    settings = config(args.models, args.prompts)
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

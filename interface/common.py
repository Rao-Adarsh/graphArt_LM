"""What every page of the interface shares: paths, saved runs, the local model, drawing.

Saved answers and adapters live outside git (results/raw, kaggle_results/, models/). In a
git worktree (.claude/worktrees/<name>) they are usually in the main checkout, so every
lookup tries this checkout first and the main one second.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# the main checkout, when this is a worktree under .claude/worktrees/
MAIN = ROOT.parents[2] if ROOT.parent.name == "worktrees" and ROOT.parents[1].name == ".claude" else ROOT
ROOTS = [ROOT] if MAIN == ROOT else [ROOT, MAIN]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graphart.render import MISSING, SPURIOUS, compare_figure, draw_graph, figure_height

OUT = Path(tempfile.mkdtemp(prefix="pensieve_"))
DATA_DIR = ROOT / "data" / "generated"

# The frozen sets. The interface never opens them: the flat holdout is scored once and
# only its saved answers are shown; the hierarchical locked sets (D9) are not read at all.
FROZEN_FLAT = "torchvision_holdout"
FROZEN_HIER = ("hier_holdout", "hier_timm_test")


def find(rel: str) -> Path | None:
    """`rel` under this checkout or the main one, whichever has it."""
    for r in ROOTS:
        p = r / rel
        if p.exists():
            return p
    return None


def jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def data(split: str) -> list[dict]:
    if split in FROZEN_HIER:
        raise ValueError(f"{split} is a locked set and is never read by the interface")
    return jsonl(DATA_DIR / f"{split}.jsonl")


def figure_path(stem: str) -> str:
    return str(OUT / f"{stem}_{time.time_ns()}.png")


def draw(truth: dict | None, pred: dict | None, title_t: str = "", title_p: str = "") -> str | None:
    """Reference and model side by side with mismatches coloured; or one of them alone."""
    if not truth and not pred:
        return None
    path = figure_path("graph")
    if truth and pred and pred["nodes"]:
        compare_figure(truth, pred, path=path)
        return path
    graph, style, label = (pred, SPURIOUS, title_p or "model output") if (pred and pred["nodes"]) else \
                          (truth, MISSING, title_t or "reference only")
    fig, ax = plt.subplots(figsize=(7, figure_height(graph)))
    draw_graph(graph, ax, bad_style=style, title=label)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# the local model: one Qwen2.5-Coder 1.5B, with every adapter the demo uses on top
# --------------------------------------------------------------------------
BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTERS = {
    "mixed-v2": "kaggle_results/graphart_finetune_mixed_v1/lora/qwen2.5-coder-1.5b-graphart-lora-mixed-v2",
    "wirer": "kaggle_results/graphart_e4_wirer/lora/qwen2.5-coder-1.5b-graphart-lora-wirer",
    "hier": "kaggle_results/graphart_hier_train/lora/qwen2.5-coder-1.5b-graphart-lora-hier",
}


class LocalModel:
    """The base model loaded once (float32 on the CPU, ~6 GB), the LoRA adapters loaded
    next to it and switched per call. Greedy decoding, as in every scored run. One call at
    a time: a second request waits for the first."""

    def __init__(self):
        self.net = None
        self.tok = None
        self.lock = threading.Lock()

    def missing(self) -> list[str]:
        return [name for name, rel in ADAPTERS.items() if find(rel + "/adapter_config.json") is None]

    def _load(self):
        if self.net is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        paths = {n: find(rel) for n, rel in ADAPTERS.items()}
        if None in paths.values():
            raise RuntimeError("adapters not found: " + ", ".join(self.missing()))
        # adapters ship the tokenizer they were trained with (the base's, same chat template)
        self.tok = AutoTokenizer.from_pretrained(str(paths["mixed-v2"]))
        net = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
        names = list(paths)
        net = PeftModel.from_pretrained(net, str(paths[names[0]]), adapter_name=names[0])
        for n in names[1:]:
            net.load_adapter(str(paths[n]), adapter_name=n)
        self.net = net.eval()

    def stream(self, adapter: str, system: str, user: str, max_tokens: int):
        """Yields the reply so far, every few tokens; the last value is the whole reply."""
        from transformers import TextIteratorStreamer
        with self.lock:
            self._load()
            self.net.set_adapter(adapter)
            messages = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": user}]
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tok(text, return_tensors="pt")
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=False,
                          pad_token_id=self.tok.eos_token_id, streamer=streamer)
            worker = threading.Thread(target=self.net.generate, kwargs=kwargs, daemon=True)
            worker.start()
            out = ""
            for piece in streamer:
                out += piece
                yield out
            worker.join()
            yield out


MODEL = LocalModel()

CSS = """
.verdict-right {color: #1b7f4b; font-weight: 600}
.verdict-wrong {color: #b3261e; font-weight: 600}
footer {display: none !important}
"""

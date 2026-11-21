"""Can the free GPUs train our longest examples? One training step at each length, measured.

    python scripts/memory_probe.py                                  # on Kaggle (2x T4)
    python scripts/memory_probe.py --lengths 256 512 --modes one    # a local smoke test (CPU)

For each sequence length and each way of placing the model, runs forward + backward through
exactly what train_lora.py trains with -- the 1.5B in fp16, gradient checkpointing, LoRA rank
16 on every projection, and the same chunked answer-only loss (`lean_loss`) -- and records
the peak GPU memory and the seconds per step. The tokens are random: memory and time depend
on the length, not on what the tokens say. The answer (the part the loss is computed on)
is the last 40% of the sequence, capped at 7,000 tokens -- about the shape of a tree-task
example (reduced code + tree).

Modes:
  one     the whole model on one T4 (what training does today, one process per GPU)
  split   the model's layers spread over both T4s in one process (twice the memory, slower)

Attention variants (the first probe ran out of memory at 8k even split over two GPUs --
something that grows with the square of the length, i.e. attention on PyTorch's plain
"math" kernel). Qwen2.5 shares each key/value head between 6 query heads; with no padding
mask, transformers hands that to PyTorch as `enable_gqa=True`, which the memory-saving
kernel may not support on a T4, so it falls back to the math kernel:
  default     as train_lora.py runs now
  nogqa       the key/value heads are expanded first (transformers' own fallback), so the
              memory-efficient kernel is allowed
  efficient   nogqa, and only the memory-efficient kernel may run -- an error instead of a
              silent fallback, which tells us whether it is usable at all

A length "fits" when the peak stays under MEMORY_HEADROOM (85%) of every GPU used -- the
rule train_lora.py --memory-check applies, because the first real run that passed at 95%
died ten steps later. Results go to memory_probe.json; the verdict is printed last.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # as train_lora.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from train_lora import IGNORE, MEMORY_HEADROOM, lean_loss

ANSWER_SHARE, MAX_ANSWER = 0.4, 7000
TRAIN_TOKENS = 7.8e6           # the estimated size of one training run (docs/hierarchical_plan.md)


def load(base: str, mode: str, cuda: bool):
    kwargs = {"dtype": torch.float16 if cuda else torch.float32}
    if cuda:
        if mode == "split":
            n = torch.cuda.device_count()
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {i: "7GiB" for i in range(n)}     # force a spread over both GPUs
        else:
            kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()
    return model


def set_variant(variant: str):
    """Configure attention for a variant; returns a context manager for the step."""
    import contextlib
    import transformers.integrations.sdpa_attention as sdpa
    if not hasattr(sdpa, "_orig_use_gqa"):
        sdpa._orig_use_gqa = sdpa.use_gqa_in_sdpa
    sdpa.use_gqa_in_sdpa = sdpa._orig_use_gqa if variant == "default" else (lambda *a, **k: False)
    if variant == "efficient":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return lambda: sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    return contextlib.nullcontext


def step(model, length: int, cuda: bool) -> float:
    """One forward + backward on a random example of `length` tokens; seconds."""
    base = model.get_base_model()
    first = base.get_input_embeddings().weight.device
    last = base.lm_head.weight.device                 # the loss runs where the head is
    answer = min(int(length * ANSWER_SHARE), MAX_ANSWER)
    ids = torch.randint(100, 30000, (1, length))
    labels = ids.clone()
    labels[:, :length - answer] = IGNORE
    batch = {"input_ids": ids.to(first), "attention_mask": torch.ones_like(ids).to(first),
             "labels": labels.to(last)}
    if cuda:
        torch.cuda.synchronize()
    t = time.time()
    loss = lean_loss(model, batch, fp16=cuda)
    loss.backward()
    if cuda:
        torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    return time.time() - t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 6144, 8192, 12288, 16384, 20480])
    ap.add_argument("--modes", nargs="+", default=["one"], choices=["one", "split"])
    ap.add_argument("--variants", nargs="+", default=["default", "nogqa", "efficient"],
                    choices=["default", "nogqa", "efficient"])
    ap.add_argument("--out", default="memory_probe.json")
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    gpus = torch.cuda.device_count() if cuda else 0
    names = [torch.cuda.get_device_name(i) for i in range(gpus)]
    totals = [torch.cuda.get_device_properties(i).total_memory for i in range(gpus)]
    print(f"GPUs: {names or 'none (CPU smoke test)'}", flush=True)
    results = []
    for mode in args.modes:
        if mode == "split" and gpus < 2:
            print("split: skipped (needs two GPUs)", flush=True)
            continue
        model = load(args.base, mode, cuda)
        attn = model.get_base_model().config._attn_implementation
        used = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"}) if cuda else []
        for variant in args.variants:
            ctx = set_variant(variant)
            print(f"\n== mode {mode}, attention {attn}/{variant}, GPUs used {used}", flush=True)
            for length in sorted(args.lengths):
                rec = {"mode": mode, "variant": variant, "length": length, "attention": attn}
                try:
                    with ctx():
                        step(model, length, cuda)                  # warm-up (kernels, allocator)
                    if cuda:
                        for i in used:
                            torch.cuda.reset_peak_memory_stats(i)
                    with ctx():
                        secs = step(model, length, cuda)
                    peaks = [torch.cuda.max_memory_allocated(i) for i in used] if cuda else []
                    share = max((p / totals[i] for p, i in zip(peaks, used)), default=0.0)
                    rec.update(ok=True, seconds=round(secs, 2), tokens_per_s=round(length / secs),
                               peak_gb=[round(p / 2**30, 2) for p in peaks], peak_share=round(share, 3),
                               fits=(share < MEMORY_HEADROOM) if cuda else None)
                    print(f"  {length:6d} tokens: {secs:6.1f} s/step, {length / secs:6.0f} tok/s, peak "
                          f"{rec['peak_gb']} GB ({share:.0%} of a GPU) -> "
                          f"{'FITS' if rec['fits'] else 'too tight' if cuda else 'ran'}", flush=True)
                except torch.OutOfMemoryError as e:
                    rec.update(ok=False, fits=False, error="out of memory: " + str(e).split(". ")[0][:200])
                    print(f"  {length:6d} tokens: {rec['error']} -- longer lengths skipped", flush=True)
                    results.append(rec)
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    if cuda:
                        torch.cuda.empty_cache()
                    break
                except RuntimeError as e:                          # e.g. no usable kernel for 'efficient'
                    rec.update(ok=False, fits=False, error=str(e).splitlines()[0][:200])
                    print(f"  {length:6d} tokens: ERROR {rec['error']} -- variant skipped", flush=True)
                    results.append(rec)
                    model.zero_grad(set_to_none=True)
                    break
                results.append(rec)
        del model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

    print("\n##### VERDICT")
    for mode, variant in [(m, v) for m in args.modes for v in args.variants]:
        rs = [r for r in results if r["mode"] == mode and r["variant"] == variant]
        if not rs:
            continue
        mode = f"{mode}/{variant}"
        fit = [r for r in rs if r.get("fits")]
        best = max(fit, key=lambda r: r["length"]) if fit else None
        if best:
            hours = TRAIN_TOKENS / best["tokens_per_s"] / 3600
            print(f"  {mode:15s}: fits up to {best['length']} tokens (peak {best['peak_share']:.0%}); at that "
                  f"length {best['tokens_per_s']} tok/s -> one 7.8M-token run ~{hours:.1f} h on this setup "
                  f"(two processes on two GPUs would halve it for 'one')")
        else:
            print(f"  {mode:15s}: nothing fits under {MEMORY_HEADROOM:.0%} at the lengths tried")
    Path(args.out).write_text(json.dumps({"gpus": names, "headroom": MEMORY_HEADROOM, "results": results},
                                         indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

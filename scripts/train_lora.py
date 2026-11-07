"""LoRA fine-tune a small open model on the train split (Phase 2).

    python scripts/train_lora.py --out lora/qwen2.5-coder-1.5b-graphart
    python scripts/train_lora.py --out lora/smoke --smoke-steps 20      # does it learn at all?

Each training example is one chat: the `zero_shot` prompt exactly as the runner
renders it (system + user), then the reference graph as the assistant's reply.
The loss is computed on the reply only, so the model is taught to answer, not to
recite the instructions. Using the same prompt before and after tuning keeps the
comparison with the prompted models clean.

The val split chooses the checkpoint (lowest val loss). The test split is never
read here. `--train-split` / `--val-split` take any splits under data/generated/,
comma-separated, so the library-style set (build_large_dataset.py) can be added or
used alone:

    python scripts/train_lora.py --out lora/mixed --train-split train,large_train --val-split val,large_val --max-len 8192

The adapter it saves is scored like any other model:

    python -m pipeline.run_prompting --backend hf --model lora/qwen2.5-coder-1.5b-graphart --prompt zero_shot --stratify 10

`--two-step` trains the two-step format instead (graphart/two_step.py): every program
in data/generated/two_step/<split>.jsonl (scripts/build_two_step.py) gives two
examples, the `two_step_nodes_brief` prompt answered with its step 1 and the
`two_step_edges_brief` prompt -- given the reference step 1 -- answered with its step
2. The brief prompts leave out the rules: the examples teach the format, and the rules
were 36% of the tokens. Score such an adapter with `pipeline/run_two_step.py --brief`.
`--full-prompts` trains on the full-rules `two_step_nodes` / `two_step_edges` instead (the
brief-prompt adapter wrote a method name, `_forward`, into module paths -- the rule that a
target is the attribute path was one of those left out); score that one without --brief.

A split written `train/2` takes every second program of it -- every second, not the
first half, because the splits are stored in tier order.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Must be set before CUDA starts. Without it, examples of very different lengths leave
# the cache fragmented: the first Kaggle run of the mixed set died asking for 3.5 GB
# with 3.4 GB free and 2.1 GB more reserved but unusable.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments)

from pipeline.backends import HF_DEFAULT
from pipeline.run_prompting import _render, load_prompt

PROMPT = "zero_shot"
TWO_STEP_PROMPTS = (("two_step_nodes_brief", "step1"), ("two_step_edges_brief", "step2"))
TWO_STEP_FULL_PROMPTS = (("two_step_nodes", "step1"), ("two_step_edges", "step2"))
IGNORE = -100                       # label value the loss skips

# Exit codes the notebook acts on. Only these two are worth a retry in fp32; any
# other failure (a crash with a traceback, exit 1) is a bug and stops the session.
EXIT_NOT_LEARNING = 3
EXIT_NONFINITE_LOSS = 4
EXIT_OUT_OF_MEMORY = 5               # --memory-check: the longest examples do not fit
MEMORY_HEADROOM = 0.85               # --memory-check fails above this share of the GPU
LOSS_CHUNK = 1024                    # answer tokens whose logits exist at once


class NonFiniteLoss(RuntimeError):
    pass


def has_dead_branch(row: dict) -> bool:
    """An `if self.<flag>:` in the source -- in this generator the flag is always set to
    False in __init__, so the branch never runs and must be left out of the graph."""
    import ast
    return any(isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
               and isinstance(n.test.value, ast.Name) and n.test.value.id == "self"
               for n in ast.walk(ast.parse(row["code"])))


# Constructs that can be held out of training entirely (RQ6), by name.
CONSTRUCTS = {"dead_branch": has_dead_branch}


def load_rows(split: str, limit: int | None, exclude: str | None = None,
              two_step: bool = False) -> list[dict]:
    """Rows of one split, or of several given comma-separated, in that order."""
    rows = []
    for name in split.split(","):
        name, _, every = name.strip().partition("/")
        path = ROOT / "data" / "generated" / ("two_step" if two_step else "") / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} not found -- build that split first")
        rows += [json.loads(line) for line in path.open(encoding="utf-8")][::int(every or 1)]
    if exclude:
        kept = [r for r in rows if not CONSTRUCTS[exclude](r)]
        print(f"{split}: held out {len(rows) - len(kept)} of {len(rows)} programs with a {exclude}",
              flush=True)
        rows = kept
    return rows[:limit] if limit else rows


def encode(tokenizer, system: str, template: str, row: dict, max_len: int, answer: str = "target") -> dict:
    """Token ids for one chat, with the prompt masked out of the labels."""
    prompt = [{"role": "system", "content": system},
              {"role": "user", "content": _render(template, row)}]
    full = prompt + [{"role": "assistant", "content": row[answer].strip()}]
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full, tokenize=False)
    # Masking by length is only right if the prompt is a strict prefix of the chat.
    assert full_text.startswith(prompt_text), "chat template does not extend the prompt"
    n_prompt = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(ids) > max_len:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceeds --max-len {max_len}")
    return {"input_ids": ids, "labels": [IGNORE] * n_prompt + ids[n_prompt:]}


def collate(pad_id: int):
    def fn(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, value: seq + [value] * (width - len(seq))
        return {"input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
                "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], IGNORE) for b in batch])}
    return fn


def lean_loss(model, batch: dict, num_items=None, fp16: bool = False):
    """The causal-LM loss on the answer tokens, without the full logits.

    The default forward builds logits over the 151,936-token vocabulary at every
    position and upcasts them to fp32 for the loss: ~3.5 GB per copy at 6,000 tokens,
    most of a T4's memory, and spent on prompt positions the loss ignores. Here the
    logits are made for answer positions only, one chunk at a time, and recomputed in
    backward. Same value as the default loss: a sum over answer tokens divided by
    `num_items` (the Trainer's count across gradient accumulation) or by their number.
    """
    causal = model.get_base_model()
    with torch.autocast(device_type=batch["input_ids"].device.type, dtype=torch.float16, enabled=fp16):
        hidden = causal.model(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"]).last_hidden_state
        labels = batch["labels"][:, 1:]          # position t predicts token t+1
        keep = labels != IGNORE
        h, y = hidden[:, :-1][keep], labels[keep]

        def chunk_loss(h, y):
            return F.cross_entropy(causal.lm_head(h).float(), y, reduction="sum")

        total = sum(checkpoint(chunk_loss, h[i:i + LOSS_CHUNK], y[i:i + LOSS_CHUNK], use_reentrant=False)
                    for i in range(0, len(y), LOSS_CHUNK))
    return total / (num_items if num_items is not None else len(y))


class LeanLossTrainer(Trainer):
    def __init__(self, *args, fp16: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.lean_fp16 = fp16

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = lean_loss(self.accelerator.unwrap_model(model), inputs, num_items_in_batch, self.lean_fp16)
        return (loss, {"loss": loss}) if return_outputs else loss


class Guard(TrainerCallback):
    """Stop on a non-finite loss (Qwen2.5 can overflow in fp16), and when the time
    budget runs out -- the best checkpoint so far is still kept."""

    def __init__(self, budget_s: float):
        self.start, self.budget_s, self.losses = time.time(), budget_s, []

    def on_log(self, args, state, control, logs=None, **_):
        loss = (logs or {}).get("loss")
        if loss is not None:
            self.losses.append(loss)
            if not math.isfinite(loss):
                raise NonFiniteLoss(f"loss is {loss} at step {state.global_step} -- "
                                    "fp16 overflow? rerun with --fp32")

    def on_step_end(self, args, state, control, **_):
        if time.time() - self.start > self.budget_s:
            print(f"time budget spent at step {state.global_step}; stopping", flush=True)
            control.should_training_stop = True
            control.should_save = True
            control.should_evaluate = True       # so the last steps can be the best checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=HF_DEFAULT, help="hub id or local path of the base model")
    ap.add_argument("--out", required=True, help="where the adapter is saved")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    # One example per step: two 2300-token examples together peak at ~10 GB on the
    # logits alone, too close to a T4's 15 GB for a failure that might only come hours
    # in. Same effective batch (16), a little slower.
    ap.add_argument("--batch", type=int, default=1, help="per-step micro-batch")
    ap.add_argument("--grad-accum", type=int, default=16, help="effective batch = batch x this")
    # Measured with the Qwen2.5 tokenizer: mean ~1400, longest 2330 (a tier-4 train
    # sample); 35 of the 1440 train+val examples exceed 2048. Over-long examples are
    # an error, never truncated -- a truncated target teaches a wrong graph.
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--check-only", action="store_true",
                    help="encode every train/val example, report lengths, and exit -- "
                         "no model is loaded, so it runs anywhere in seconds")
    ap.add_argument("--eval-steps", type=int, default=40)
    ap.add_argument("--train-split", default="train", help="split(s) to train on, comma-separated")
    ap.add_argument("--val-split", default="val", help="split(s) that choose the checkpoint, comma-separated")
    ap.add_argument("--limit-train", type=int, help="only the first N train samples")
    ap.add_argument("--limit-val", type=int, help="only the first N val samples")
    ap.add_argument("--exclude-construct", choices=sorted(CONSTRUCTS),
                    help="leave every train and val program containing this construct out (RQ6)")
    ap.add_argument("--smoke-steps", type=int,
                    help="train N steps only, then exit non-zero unless the loss is finite and falling")
    ap.add_argument("--drop-over-max", action="store_true",
                    help="leave out examples longer than --max-len (reported) instead of stopping")
    ap.add_argument("--memory-check", type=int, metavar="N",
                    help="run forward and backward on the N longest train examples, report peak GPU "
                         "memory, and exit (5 if they do not fit) -- minutes, not hours, to find out")
    ap.add_argument("--time-budget-hours", type=float, default=9.0)
    ap.add_argument("--fp32", action="store_true", help="no fp16, even on a GPU (slower, never overflows)")
    ap.add_argument("--two-step", action="store_true",
                    help="train the two-step format: two examples per program (see the docstring)")
    ap.add_argument("--full-prompts", action="store_true",
                    help="with --two-step: the full-rules prompts instead of the brief ones")
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    fp16 = cuda and not args.fp32
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # (system, template, answer field) per example of a program
    if args.full_prompts and not args.two_step:
        raise SystemExit("--full-prompts only applies with --two-step")
    two_step_prompts = TWO_STEP_FULL_PROMPTS if args.full_prompts else TWO_STEP_PROMPTS
    prompts = ([(*load_prompt(name), field) for name, field in two_step_prompts] if args.two_step
               else [(*load_prompt(PROMPT), "target")])

    def encode_all(rows: list[dict], name: str) -> list[dict]:
        out, dropped = [], []
        for r in rows:
            for system, template, answer in prompts:
                try:
                    out.append(encode(tokenizer, system, template, r, args.max_len, answer))
                except ValueError:
                    if not args.drop_over_max:
                        raise
                    dropped.append(r["id"] + ("" if answer == "target" else f"/{answer}"))
        if dropped:
            print(f"{name}: left out {len(dropped)} examples over --max-len {args.max_len}: "
                  f"{', '.join(dropped[:8])}{' ...' if len(dropped) > 8 else ''}", flush=True)
        return out

    train = encode_all(load_rows(args.train_split, args.limit_train, args.exclude_construct, args.two_step), "train")
    val = encode_all(load_rows(args.val_split, args.limit_val, args.exclude_construct, args.two_step), "val")
    lengths = [len(e["input_ids"]) for e in train + val]
    print(f"{len(train)} train / {len(val)} val examples; tokens per example "
          f"mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)} (limit {args.max_len})",
          flush=True)
    if args.check_only:
        return

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16 if fp16 else torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    if args.memory_check:
        # The longest examples set the peak, through the same loss as training. Forward
        # + backward only -- the LoRA optimizer state is a few MB. Fitting is not
        # enough: the Trainer and the cache need room too, so the peak must stay under
        # MEMORY_HEADROOM of the GPU (the first run passed at 95% and died 10 steps in).
        device = "cuda" if cuda else "cpu"
        model.to(device).train()
        longest = sorted(train, key=lambda e: len(e["input_ids"]), reverse=True)[:args.memory_check]
        total = torch.cuda.get_device_properties(0).total_memory if cuda else 0
        try:
            for e in longest:
                batch = collate(pad_id)([e])
                batch = {k: v.to(device) for k, v in batch.items()}
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                lean_loss(model, batch, fp16=fp16).backward()
                model.zero_grad(set_to_none=True)
                if cuda:
                    peak = torch.cuda.max_memory_allocated()
                    print(f"  {len(e['input_ids'])} tokens: peak {peak / 2**30:.1f} GB "
                          f"of {total / 2**30:.1f}", flush=True)
                    if peak > MEMORY_HEADROOM * total:
                        print(f"TOO CLOSE: over {MEMORY_HEADROOM:.0%} of the GPU on a {len(e['input_ids'])}-token "
                              "example -- use a smaller --max-len with --drop-over-max", flush=True)
                        raise SystemExit(EXIT_OUT_OF_MEMORY)
        except torch.cuda.OutOfMemoryError:
            print(f"OUT OF MEMORY on a {len(e['input_ids'])}-token example -- use a smaller "
                  "--max-len with --drop-over-max", flush=True)
            raise SystemExit(EXIT_OUT_OF_MEMORY)
        print(f"memory check passed on the {len(longest)} longest examples "
              f"({len(longest[0]['input_ids'])} tokens at most)", flush=True)
        return

    smoke = args.smoke_steps is not None
    training_args = TrainingArguments(
        output_dir=str(Path(args.out) / "checkpoints"),
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.smoke_steps if smoke else -1,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03,   # a float is a ratio
        logging_steps=1 if smoke else 10,
        eval_strategy="no" if smoke else "steps", eval_steps=args.eval_steps,
        save_strategy="no" if smoke else "steps", save_steps=args.eval_steps, save_total_limit=2,
        load_best_model_at_end=not smoke, metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=fp16, report_to="none", seed=0, data_seed=0, remove_unused_columns=False,
        dataloader_num_workers=0)
    guard = Guard(args.time_budget_hours * 3600)
    trainer = LeanLossTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=val,
                              data_collator=collate(pad_id), callbacks=[guard], fp16=fp16)

    started = time.time()
    try:
        trainer.train()
    except NonFiniteLoss as exc:
        print(f"STOPPED: {exc}", flush=True)
        raise SystemExit(EXIT_NONFINITE_LOSS)
    elapsed = time.time() - started

    if smoke:
        # A clear drop, not noise: the last few steps must average 10% below the first
        # few. A model that is learning the output format falls much faster than that.
        mean = lambda xs: sum(xs) / len(xs)
        first, last = mean(guard.losses[:3]), mean(guard.losses[-3:])
        ok = all(math.isfinite(x) for x in guard.losses) and last < 0.9 * first
        print(f"smoke: loss {first:.3f} -> {last:.3f} over {len(guard.losses)} steps, "
              f"{elapsed / max(len(guard.losses), 1):.1f} s/step -> {'OK' if ok else 'NOT LEARNING'}",
              flush=True)
        raise SystemExit(0 if ok else EXIT_NOT_LEARNING)

    out = Path(args.out)
    trainer.save_model(str(out))             # the best checkpoint by val loss
    tokenizer.save_pretrained(str(out))
    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    summary = {"base": args.base,
               "prompt": ("two_step" if args.full_prompts else "two_step_brief") if args.two_step else PROMPT, "excluded_construct": args.exclude_construct,
               "train_split": args.train_split, "val_split": args.val_split, "max_len": args.max_len,
               "train_examples": len(train),
               "val_examples": len(val), "epochs": args.epochs, "lr": args.lr,
               "rank": args.rank, "alpha": args.alpha,
               "effective_batch": args.batch * args.grad_accum, "fp16": fp16,
               "steps": trainer.state.global_step, "minutes": round(elapsed / 60, 1),
               "best_eval_loss": trainer.state.best_metric,
               "best_checkpoint": trainer.state.best_model_checkpoint,
               "eval_history": [{"step": h["step"], "eval_loss": h["eval_loss"]} for h in evals],
               "train_loss_history": guard.losses}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved adapter to {out} after {summary['minutes']} min; "
          f"best val loss {summary['best_eval_loss']}", flush=True)


if __name__ == "__main__":
    main()

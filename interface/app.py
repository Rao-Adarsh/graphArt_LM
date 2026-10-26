"""Pensieve graph interface: paste a PyTorch module, see the forward-pass graph the
fine-tuned model draws for it beside the one torch.fx traces, with every mismatch
coloured and the project's metrics.

    .venv/Scripts/python interface/app.py          then open http://127.0.0.1:7860

The model is Qwen2.5-Coder 1.5B with the project's LoRA adapter (models/), run on this
machine's CPU: about 1-3 minutes per program. It is near-perfect on code written like
its training data and flattens most real-world code into a chain -- see the README.

Examples from the test, torchvision and edit sets come with the model's answers saved
during the experiments (results/*.jsonl), so they display instantly; untick "use the
saved answer" to run the model live instead.

The reference graph is traced by running the pasted code's __init__ with torch.fx --
which executes it. The app listens on 127.0.0.1 only; paste only code you trust.
"""

import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx

from graphart.extract import graph_from_code
from graphart.generate import generate_program
from graphart.metrics import compare
from graphart.render import MISSING, SPURIOUS, compare_figure, draw_graph
from graphart.schema import to_text
from graphart.validate import node_label, to_networkx
from pipeline.backends import make_backend
from pipeline.run_prompting import _render, load_prompt
from pipeline.score import load_truths
from scripts.classify_errors import classify, parse

ADAPTER = ROOT / "models" / "qwen2.5-coder-1.5b-graphart-lora"
SAVED_RUN = "hf__qwen2.5-coder-1.5b-graphart-lora__zero_shot"
OUT = Path(tempfile.mkdtemp(prefix="pensieve_"))

SETS = {"test (our generator)": "test", "torchvision (real code)": "torchvision",
        "edits (controlled changes)": "edits"}

CATEGORY_TEXT = {
    "exact": "Exactly the graph torch.fx traces.",
    "notation": "Right graph; layer settings or function-vs-method spelled differently.",
    "spelling": "Right graph; an operation spelled another way (e.g. relu as a function where the code has nn.ReLU).",
    "commutative": "Right graph; only the operands of + or * are swapped.",
    "arg_slots": "Right connections, but an input sits in the wrong argument slot (e.g. the order inside cat).",
    "coarse_container": "Right at a coarser level: an nn.Sequential/ModuleList drawn as one box.",
    "coarse_custom": "Right at a coarser level: a custom block class drawn as one box.",
    "gave_original": "Wrong: this is the graph of the program BEFORE the edit.",
    "miswired": "Wrong: the right operations, connected differently.",
    "wrong_ops": "Wrong: operations missing or invented.",
    "unparseable": "The reply could not be read as a graph.",
}
RIGHT = {"exact", "notation", "spelling", "commutative"}


# --------------------------------------------------------------------------
# examples, and the answers saved during the experiments
# --------------------------------------------------------------------------
def _jsonl(path: Path) -> list[dict]:
    return list(map(json.loads, path.open(encoding="utf-8"))) if path.exists() else []


DATA = {s: {r["id"]: r for r in _jsonl(ROOT / "data" / "generated" / f"{s}.jsonl")} for s in SETS.values()}
TRUTHS = {s: load_truths(s) for s, rows in DATA.items() if rows}
SAVED = {s: {r["id"]: r["reply"] for r in _jsonl(ROOT / "results" / (
             f"{SAVED_RUN}.jsonl" if s == "test" else f"{SAVED_RUN}__{s}.jsonl"))} for s in SETS.values()}


TIERS = {"1 flat: one layer per line": 1, "2 loops: Sequential, ModuleList, loops": 2,
         "3 branches: residuals, concats, dead branches, reuse": 3,
         "4 blocks: residual, inception, transformer, U-net": 4}


def _graph_hash(graph: dict) -> str:
    """Weisfeiler-Lehman hash under the exact-match labels: equal graphs always get
    equal hashes, so a hash not in the training set means a graph the model never saw."""
    g = to_networkx(graph)
    for n, data in g.nodes(data=True):
        data["h"] = str(node_label(data))
    for a, b, data in g.edges(data=True):
        data["h"] = str(data["label"])
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="h", edge_attr="h")


TRAIN = {}          # graph hash -> a training program with that graph, built on first use


def generate(tier_label: str, seed: float | None):
    """A fresh program from the project's generator, and whether its graph is in training."""
    if not TRAIN:
        for r in _jsonl(ROOT / "data" / "generated" / "train.jsonl"):
            TRAIN.setdefault(_graph_hash(r["graph"]), r["id"])
    seed = int(seed) if seed not in (None, "") else random.randrange(10 ** 6)
    code = generate_program(random.Random(seed), TIERS[tier_label])
    graph, _ = graph_from_code(code)
    seen = TRAIN.get(_graph_hash(graph))
    note = (f"Seed {seed}. **This graph is in the training data** (same as `{seen}`), so a right answer may "
            f"be recall. Generate again for a new one." if seen else
            f"Seed {seed}. A new graph: no training program has it ({len(graph['nodes'])} nodes).")
    return code, None, note


def example_ids(set_label: str) -> list[str]:
    """The examples that have a saved answer: the 40 stratified test programs, all of
    torchvision, all of the edits."""
    split = SETS[set_label]
    return [i for i in DATA[split] if i in SAVED[split]]


# --------------------------------------------------------------------------
# the model, loaded once on first use
# --------------------------------------------------------------------------
MODEL = []


def model():
    if not MODEL:
        if not (ADAPTER / "adapter_config.json").exists():
            raise gr.Error(f"No adapter at {ADAPTER.relative_to(ROOT)} -- see README, 'Interface'.")
        MODEL.append(make_backend("hf", str(ADAPTER), max_tokens=2000))
    return MODEL[0]


# --------------------------------------------------------------------------
# one request
# --------------------------------------------------------------------------
def _figure(truth: dict | None, pred: dict | None) -> str:
    path = OUT / f"graph_{time.time_ns()}.png"
    if truth and pred:
        compare_figure(truth, pred, path=str(path))
        return str(path)
    graph, style, label = (pred, SPURIOUS, "model output") if pred else (truth, MISSING, "reference only")
    fig, ax = plt.subplots(figsize=(7, max(5.0, 0.35 * len(graph["nodes"]) + 2)))
    draw_graph(graph, ax, bad_style=style, title=label)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _metrics(scores: dict | None, category: str | None) -> list[list]:
    if scores is None:
        return [["(no score: needs both a reference and a readable reply)", ""]]
    yes = lambda b: "yes" if b else "no"
    shape = scores["shape_acc"]
    return [["exact (labelled isomorphism)", yes(scores["exact"])],
            ["topology (layer settings ignored)", yes(scores["exact_topo"])],
            ["right graph (spelling forgiven)", yes(category in RIGHT)],
            ["node F1", f"{scores['node_f1']:.2f}"],
            ["edge F1", f"{scores['edge_f1']:.2f}"],
            ["connectivity preserved (reach F1)", f"{scores['reach_f1']:.2f}"],
            ["shape accuracy", "-" if shape is None else f"{shape:.2f}"],
            ["valid (well-formed DAG, consistent shapes)", yes(scores["valid"])]]


def run(code: str, use_saved: bool, example: dict | None):
    """Generator: a status line first, then the full result."""
    code = (code or "").strip()
    if not code:
        raise gr.Error("Paste a PyTorch nn.Module, upload a .py file, or pick an example.")
    from_example = bool(example) and example["code"].strip() == code
    yield "Tracing the reference graph with torch.fx ...", None, None, "", "", ""

    truth, trace_error = None, ""
    if from_example:
        truth = TRUTHS[example["split"]][example["id"]]
    else:
        try:
            truth, _ = graph_from_code(code)
        except Exception as exc:          # pasted code: anything can go wrong
            trace_error = f"{type(exc).__name__}: {exc}"

    reply = SAVED[example["split"]].get(example["id"]) if (from_example and use_saved) else None
    source = "answer saved during the experiments"
    if reply is None:
        yield "Running the fine-tuned model on the CPU -- about 1-3 minutes ...", None, None, "", "", ""
        system, user = load_prompt("zero_shot")
        t = time.time()
        try:
            reply = model()(system, _render(user, {"code": code}))
        except Exception as exc:
            raise gr.Error(f"The model could not answer: {exc}")
        source = f"live answer, {time.time() - t:.0f} s"

    pred, parse_error = parse(reply)
    scores = compare(pred, truth) if (pred and truth) else None
    original = None
    if from_example and example["split"] == "edits":
        row = DATA["edits"][example["id"]]
        original = TRUTHS["test"][row["base_id"]] if row["kind"] == "meaningful" else None
    category = classify(reply, truth, code, original)[0] if truth else None

    lines = [f"**Answer:** {source}."]
    if category:
        lines.append(f"**Verdict:** `{category}` — {CATEGORY_TEXT[category]}")
    if trace_error:
        lines.append(f"**No reference graph:** torch.fx could not trace this code ({trace_error}). "
                     "The source must define the module and create one instance; add "
                     "`example_input = torch.randn(...)` for shapes.")
    if parse_error:
        lines.append(f"**Reply did not parse:** {parse_error}")
    if scores and scores["errors"]:
        lines.append("**Validator:** " + "; ".join(scores["errors"][:4]))
    if from_example:
        lines.append(f"*Example `{example['id']}` from the {example['split']} set.*")
    figure = _figure(truth, pred) if (truth or pred) else None
    yield ("Done.", figure, _metrics(scores, category), "\n\n".join(lines),
           to_text(truth) if truth else "", reply)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------
def load_example(set_label: str, sample_id: str | None):
    row = DATA[SETS[set_label]].get(sample_id or "")
    if not row:
        return gr.update(), None
    return row["code"], {"split": SETS[set_label], "id": sample_id, "code": row["code"]}


def load_file(file):
    if file is None:
        return gr.update(), None
    return Path(file).read_text(encoding="utf-8", errors="replace"), None


INTRO = """
# Pensieve: code → forward-pass graph
Paste a PyTorch `nn.Module` — define the class, create one instance, and optionally set
`example_input = torch.randn(...)` for shapes. The fine-tuned model reads the code, without running it, and draws
the graph; torch.fx traces the true one. **Red**: in the reference, missed by the model. **Orange**: invented by the model.
"""

with gr.Blocks(title="Pensieve graph interface") as demo:
    gr.Markdown(INTRO)
    example_state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Tab("Examples"):
                ex_set = gr.Dropdown(list(SETS), value="test (our generator)", label="set")
                ex_id = gr.Dropdown(example_ids("test (our generator)"), label="example")
            with gr.Tab("Generate"):
                gen_tier = gr.Dropdown(list(TIERS), value=list(TIERS)[2], label="tier")
                gen_seed = gr.Number(None, precision=0, label="seed (empty: random)")
                gen_go = gr.Button("Generate a program")
                gen_note = gr.Markdown("")
            with gr.Tab("Upload a .py file"):
                upload = gr.File(file_types=[".py"], type="filepath", label="python file")
            code = gr.Code(language="python", label="code (paste or edit here)", lines=18)
            use_saved = gr.Checkbox(True, label="for examples, use the answer saved during the experiments (instant)")
            go = gr.Button("Draw the graph", variant="primary")
        with gr.Column(scale=7):
            status = gr.Markdown("")
            image = gr.Image(label="reference (left) vs model (right)", type="filepath", height=620)
            verdict = gr.Markdown("")
            metrics = gr.Dataframe(headers=["metric", "value"], label="metrics", interactive=False)
            with gr.Accordion("graph text: reference and raw model reply", open=False):
                ref_text = gr.Code(label="reference (torch.fx)", lines=12)
                reply_text = gr.Code(label="model reply", lines=12)

    ex_set.change(lambda s: gr.update(choices=example_ids(s), value=None), ex_set, ex_id)
    ex_id.change(load_example, [ex_set, ex_id], [code, example_state])
    upload.change(load_file, upload, [code, example_state])
    gen_go.click(generate, [gen_tier, gen_seed], [code, example_state, gen_note])
    go.click(run, [code, use_saved, example_state], [status, image, metrics, verdict, ref_text, reply_text])


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)

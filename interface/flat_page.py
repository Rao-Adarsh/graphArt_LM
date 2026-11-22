"""Page 2, the flat format: the whole torch.fx graph written out, one node per line.

Answers come either from the runs saved during the experiments (instant), or live from
the fine-tuned 1.5B on this machine: the final pipeline (mixed-v2 lists the nodes, the
1.5B wirer connects them) or mixed-v2 alone.
"""

from __future__ import annotations

import random
import time

import gradio as gr
import networkx as nx

from interface.common import FROZEN_FLAT, MODEL, data, draw, find, jsonl
from graphart.extract import graph_from_code
from graphart.generate import generate_program
from graphart.metrics import compare
from graphart.schema import ParseError, to_text
from graphart.two_step import assemble
from graphart.validate import node_label, to_networkx
from pipeline.run_prompting import _render, load_prompt
from pipeline.score import load_truths
from scripts.classify_errors import classify, parse

SETS = {"generated test, tiers 1-4 [D1]": "test",
        "torchvision dev, 21 real blocks [D3]": "torchvision",
        "torchvision holdout, 17 unseen blocks [D4] (saved answers only)": FROZEN_FLAT,
        "edits of the test programs [D5]": "edits"}

_LORA = "hf__qwen2.5-coder-1.5b-graphart-lora"
FINAL = "final: mixed-v2 lists + 1.5B wirer connects [M7]"
# label -> results/<run>.jsonl (test) or results/<run>__<set>.jsonl
RUNS = {
    FINAL: f"{_LORA}-wirer__two_step_brief_nodes_from_mixed-v2",
    "mixed-v2 lists + 14B wirer connects [M7, E1]": "ollama__qwen2.5-coder-14b__edges_only_cos_nodes_from_mixed-v2",
    "mixed-v2 alone, one step [M5 + D2]": f"{_LORA}-mixed-v2__zero_shot",
    "26 Oct 1.5B, one step [M5]": f"{_LORA}__zero_shot",
    "1.5B two-step, brief [M6]": f"{_LORA}-two-step__two_step_brief",
    "1.5B wirer given the true nodes (diagnostic)": f"{_LORA}-wirer__two_step_brief_nodes_from_truenodes",
    "Qwen 14B, zero-shot [M1]": "ollama__qwen2.5-coder-14b__zero_shot",
    "Qwen 14B, chain of simulation [M4]": "ollama__qwen2.5-coder-14b__cos",
    "Qwen 7B, zero-shot [M1]": "ollama__qwen2.5-coder-7b__zero_shot",
    "Qwen 1.5B untuned, zero-shot [M1]": "hf__Qwen-Qwen2.5-Coder-1.5B-Instruct__zero_shot",
    "Gemma 31B, zero-shot [M1]": "openrouter__google-gemma-4-31b-it-free__zero_shot",
}
LIVE = {"final pipeline: mixed-v2 lists, 1.5B wirer connects (2 calls)": "pipeline",
        "mixed-v2 alone (1 call)": "mixed"}
SAVED_MODE, LIVE_MODE = "saved answer (instant)", "run the 1.5B live on this laptop"

CATEGORY_TEXT = {
    "exact": "Exactly the graph torch.fx traces.",
    "notation": "Right graph; layer settings or function-vs-method written differently.",
    "spelling": "Right graph; an operation spelled another way (relu as a function where the code has nn.ReLU).",
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

DATA = {s: {r["id"]: r for r in data(s)} for s in SETS.values()}
TRUTHS = {s: load_truths(s) for s, rows in DATA.items() if rows}
SAVED = {s: {label: replies for label, run in RUNS.items()
             if (replies := {r["id"]: r["reply"] for r in jsonl(
                 find(f"results/{run}.jsonl" if s == "test" else f"results/{run}__{s}.jsonl"))})}
         for s in SETS.values()}


def run_labels(set_label: str) -> list[str]:
    return list(SAVED[SETS[set_label]])


def example_ids(set_label: str) -> list[str]:
    split = SETS[set_label]
    return [i for i in DATA[split] if any(i in replies for replies in SAVED[split].values())]


# --------------------------------------------------------------------------
# generate a program
# --------------------------------------------------------------------------
TIERS = {"1 flat: one layer per line": 1, "2 loops: Sequential, ModuleList, loops": 2,
         "3 branches: residuals, concats, dead branches, reuse": 3,
         "4 blocks: residual, inception, transformer, U-net": 4}
TRAIN = {}          # graph hash -> a training program with that graph, built on first use


def _graph_hash(graph: dict) -> str:
    """Weisfeiler-Lehman hash under the exact-match labels: a hash not in the training set
    means a graph the model never saw."""
    g = to_networkx(graph)
    for _, d in g.nodes(data=True):
        d["h"] = str(node_label(d))
    for _, _, d in g.edges(data=True):
        d["h"] = str(d["label"])
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="h", edge_attr="h")


def generate(tier_label: str, seed):
    if not TRAIN:
        for r in data("train"):
            TRAIN.setdefault(_graph_hash(r["graph"]), r["id"])
    seed = int(seed) if seed not in (None, "") else random.randrange(10 ** 6)
    code = generate_program(random.Random(seed), TIERS[tier_label])
    graph, _ = graph_from_code(code)
    seen = TRAIN.get(_graph_hash(graph))
    note = (f"Seed {seed}. **This graph is in the training data** (as `{seen}`): a right answer may be "
            "recall. Generate again for a new one." if seen else
            f"Seed {seed}. A new graph, in no training program ({len(graph['nodes'])} nodes).")
    return code, None, note, gr.update(value=LIVE_MODE)


# --------------------------------------------------------------------------
# one request
# --------------------------------------------------------------------------
def _metrics(scores: dict | None, category: str | None) -> list[list]:
    if scores is None:
        return [["(no score: needs a reference graph and a readable reply)", ""]]
    yes = lambda b: "yes" if b else "no"
    shape = scores["shape_acc"]
    return [["exact (labelled graph isomorphism)", yes(scores["exact"])],
            ["topology (layer settings ignored)", yes(scores["exact_topo"])],
            ["lenient: right graph, notation forgiven", yes(category in RIGHT)],
            ["node F1", f"{scores['node_f1']:.2f}"],
            ["edge F1", f"{scores['edge_f1']:.2f}"],
            ["connectivity kept (reachability F1)", f"{scores['reach_f1']:.2f}"],
            ["shape accuracy", "-" if shape is None else f"{shape:.2f}"],
            ["valid (well-formed DAG, consistent shapes)", yes(scores["valid"])]]


def live_reply(code: str, mode: str):
    """Yields (status, text so far, final reply or None)."""
    steps = 2 if mode == "pipeline" else 1
    sys1, tpl1 = load_prompt("zero_shot")
    t0, last = time.time(), 0.0
    if MODEL.net is None:
        yield "Loading Qwen2.5-Coder 1.5B and the adapters (first run only, about 30 s) ...", "", None
    step1 = ""
    for step1 in MODEL.stream("mixed-v2", sys1, _render(tpl1, {"code": code}), 2000):
        if time.time() - last > 0.4:
            last = time.time()
            yield f"Step 1 of {steps}: mixed-v2 writes the graph ... {time.time() - t0:.0f} s", step1, None
    if mode == "mixed":
        yield f"Done in {time.time() - t0:.0f} s.", step1, step1
        return
    sys2, tpl2 = load_prompt("two_step_edges_brief")
    step2 = ""
    for step2 in MODEL.stream("wirer", sys2, _render(tpl2, {"code": code, "step1": step1}), 2000):
        if time.time() - last > 0.4:
            last = time.time()
            yield (f"Step 2 of 2: the wirer connects mixed-v2's nodes ... {time.time() - t0:.0f} s",
                   step1 + "\n\n--- step 2 ---\n" + step2, None)
    try:
        reply = assemble(step1, step2)
    except ParseError:
        reply = step1.strip() + "\n" + step2.strip()
    yield f"Done in {time.time() - t0:.0f} s.", reply, reply


def run(code: str, mode: str, run_label: str | None, live_label: str, example: dict | None):
    code = (code or "").strip()
    if not code:
        raise gr.Error("Pick an example, generate a program, upload a .py file, or paste code.")
    from_example = bool(example) and example["code"].strip() == code
    blank = (None, None, "", "", "")
    yield ("Tracing the reference graph with torch.fx ...", *blank)

    truth, trace_error = None, ""
    if from_example:
        truth = TRUTHS[example["split"]][example["id"]]
    else:
        try:
            truth, _ = graph_from_code(code)
        except Exception as exc:                  # pasted code: anything can go wrong
            trace_error = f"{type(exc).__name__}: {exc}"

    if mode == SAVED_MODE:
        if not from_example:
            raise gr.Error("Saved answers exist only for the examples, unchanged. Choose 'run the 1.5B live'.")
        saved = SAVED[example["split"]].get(run_label or "", {})
        if example["id"] not in saved:
            raise gr.Error(f"'{run_label}' has no saved answer for {example['id']}: pick another run.")
        reply, source = saved[example["id"]], f"saved answer of **{run_label}**"
    else:
        if (example or {}).get("split") == FROZEN_FLAT and from_example:
            raise gr.Error("The holdout is scored once and is spent: only its saved answers are shown.")
        if MODEL.missing():
            raise gr.Error("Adapters not found: " + ", ".join(MODEL.missing()))
        reply, t0 = None, time.time()
        for status, text, final in live_reply(code, LIVE[live_label]):
            reply = final
            yield (status, None, None, "", "", text)
        source = f"live answer, {live_label}, {time.time() - t0:.0f} s"

    pred, parse_error = parse(reply)
    scores = compare(pred, truth) if (pred and truth) else None
    original = None
    if from_example and example["split"] == "edits":
        row = DATA["edits"][example["id"]]
        original = TRUTHS["test"][row["base_id"]] if row["kind"] == "meaningful" else None
    category = classify(reply, truth, code, original)[0] if truth else None

    lines = [f"**Answer:** {source}."]
    if category:
        cls = "verdict-right" if category in RIGHT else "verdict-wrong"
        lines.append(f"<span class='{cls}'>Verdict: {category}</span>: {CATEGORY_TEXT[category]}")
    if trace_error:
        lines.append(f"**No reference graph:** torch.fx could not trace this code ({trace_error}). Define the "
                     "module, create one instance, and add `example_input = torch.randn(...)` for shapes.")
    if parse_error:
        lines.append(f"**The reply did not parse:** {parse_error}")
    if scores and scores["errors"]:
        lines.append("**Validator:** " + "; ".join(scores["errors"][:4]))
    if from_example:
        lines.append(f"*Example `{example['id']}`, {example['split']} set.*")
    yield ("Done.", draw(truth, pred), _metrics(scores, category), "\n\n".join(lines),
           to_text(truth) if truth else "", reply)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------
INTRO = """
## Flat format: the whole torch.fx graph
The model reads the code, without running it, and writes every node of the forward pass and
what feeds it. torch.fx traces the true graph. **Red**: in the reference, missed by the model.
**Orange**: invented by the model.
"""


def load_example(set_label: str, sample_id: str | None):
    row = DATA[SETS[set_label]].get(sample_id or "")
    if not row:
        return gr.update(), None
    return row["code"], {"split": SETS[set_label], "id": sample_id, "code": row["code"]}


def build():
    gr.Markdown(INTRO)
    first = list(SETS)[1]
    # the page opens on the first torchvision block, already drawn with the final pipeline
    sid = example_ids(first)[0]
    ex0 = {"split": SETS[first], "id": sid, "code": DATA[SETS[first]][sid]["code"]}
    _, img0, m0, v0, ref0, r0 = list(run(ex0["code"], SAVED_MODE, FINAL, list(LIVE)[0], ex0))[-1]
    example = gr.State(ex0)
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.Tab("Examples"):
                    ex_set = gr.Dropdown(list(SETS), value=first, label="set")
                    ex_id = gr.Dropdown(example_ids(first), value=sid, label="program")
                with gr.Tab("Generate"):
                    gen_tier = gr.Dropdown(list(TIERS), value=list(TIERS)[2], label="tier")
                    gen_seed = gr.Number(None, precision=0, label="seed (empty: random)")
                    gen_go = gr.Button("Generate a program")
                    gen_note = gr.Markdown("")
                with gr.Tab("Upload or paste"):
                    upload = gr.File(file_types=[".py"], type="filepath", label=".py file")
                    gr.Markdown("Or paste into the code box. Define the class and create one instance; "
                                "`example_input = torch.randn(...)` adds shapes.")
            code = gr.Code(ex0["code"], language="python", label="code", lines=16, max_lines=28)
            mode = gr.Radio([SAVED_MODE, LIVE_MODE], value=SAVED_MODE, label="answer")
            run_label = gr.Dropdown(run_labels(first), value=FINAL, label="saved answer from")
            live_label = gr.Dropdown(list(LIVE), value=list(LIVE)[0], label="live model", visible=False)
            go = gr.Button("Draw the graph", variant="primary")
        with gr.Column(scale=7):
            status = gr.Markdown("")
            image = gr.Image(img0, label="reference (left) and model (right)", type="filepath", height=600)
            verdict = gr.Markdown(v0)
            with gr.Accordion("scores", open=False):
                metrics = gr.Dataframe(m0, headers=["measure", "value"], interactive=False)
            with gr.Accordion("graph text: reference and the model's reply", open=False):
                ref_text = gr.Code(ref0, label="reference (torch.fx)", lines=12)
                reply_text = gr.Code(r0, label="model reply (streams in during a live run)", lines=12)

    mode.change(lambda m: (gr.update(visible=m == SAVED_MODE), gr.update(visible=m == LIVE_MODE)),
                mode, [run_label, live_label])
    ex_set.change(lambda s: (gr.update(choices=example_ids(s), value=None),
                             gr.update(choices=run_labels(s), value=FINAL if FINAL in run_labels(s) else
                                       run_labels(s)[0])),
                  ex_set, [ex_id, run_label])
    picked = ex_id.change(load_example, [ex_set, ex_id], [code, example])
    upload.change(lambda f: (open(f, encoding="utf-8", errors="replace").read(), None, gr.update(value=LIVE_MODE))
                  if f else (gr.update(), None, gr.update()), upload, [code, example, mode])
    gen_go.click(generate, [gen_tier, gen_seed], [code, example, gen_note, mode])
    outputs = [status, image, metrics, verdict, ref_text, reply_text]
    go.click(run, [code, mode, run_label, live_label, example], outputs)

    def redraw(code_, mode_, run_label_, live_label_, example_):
        """Switching the saved run on a loaded example redraws at once."""
        if not (example_ and code_ and mode_ == SAVED_MODE and code_.strip() == example_["code"].strip()):
            yield tuple(gr.update() for _ in outputs)
            return
        yield from run(code_, mode_, run_label_, live_label_, example_)
    run_label.change(redraw, [code, mode, run_label, live_label, example], outputs)
    picked.then(redraw, [code, mode, run_label, live_label, example], outputs)

"""Page 4, does the model follow the code? The controlled edits [D5], experiment E2.

Each edit changes one test program a little. A meaningful edit changes the graph (an input
rerouted, a concat swapped, a call dropped, a branch switched on); a neutral one does not
(a rename, an alias, reordered lines, a dead flag). A model that pattern-matches answers a
meaningful edit with the graph of the program before the edit.
"""

from __future__ import annotations

import difflib
import html

import gradio as gr

from interface.common import draw, find, jsonl
from interface.flat_page import CATEGORY_TEXT, DATA, RIGHT, TRUTHS
from scripts.classify_errors import classify, parse

_LORA = "hf__qwen2.5-coder-1.5b-graphart-lora"
RUNS = {"two-step, brief: names the variable on each edge [M6]": f"{_LORA}-two-step__two_step_brief",
        "mixed-v2, one step [M5 + D2]": f"{_LORA}-mixed-v2__zero_shot",
        "26 Oct 1.5B, one step [M5]": f"{_LORA}__zero_shot",
        "Qwen 14B, zero-shot [M1]": "ollama__qwen2.5-coder-14b__zero_shot"}
SAVED = {label: {r["id"]: r["reply"] for r in jsonl(find(f"results/{run}__edits.jsonl"))}
         for label, run in RUNS.items()}
SAVED = {k: v for k, v in SAVED.items() if v}

EDITS = {"reroute: an input now comes from elsewhere": "reroute",
         "swap_cat: the order inside a concatenation swapped": "swap_cat",
         "drop_call: a call removed": "drop_call",
         "branch_on: a switched-off branch turned on": "branch_on",
         "rename (neutral)": "rename", "alias (neutral)": "alias",
         "reorder (neutral)": "reorder", "dead_flag (neutral)": "dead_flag"}

E2_TABLE = """
| 69 meaningful edits | mixed-v2, one step | two-step, brief |
|---|---|---|
| answered with the **unedited** graph | 39 (57%) | **14 (20%)** |
| right graph | 17 (25%) | **47 (68%)** |
| *124 neutral edits:* right graph | 111 (90%) | 109 (88%) |

Exact McNemar on the paired answers: unedited graph 26 vs 1, p ≈ 4e-7. Criterion written down
before the run: ≥ 20 points fewer unedited graphs, neutral edits ≥ 85% right.
"""


def edit_ids(edit_label: str) -> list[str]:
    kind = EDITS[edit_label]
    return [i for i, r in DATA["edits"].items() if r["edit"] == kind and any(i in s for s in SAVED.values())]


def code_diff_html(before: str, after: str) -> str:
    a, b = before.splitlines(), after.splitlines()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    left, right = [], []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        n = max(a1 - a0, b1 - b0)
        for k in range(n):
            la = a[a0 + k] if a0 + k < a1 else ""
            lb = b[b0 + k] if b0 + k < b1 else ""
            left.append((la, op != "equal"))
            right.append((lb, op != "equal"))

    def col(rows, colour, title):
        body = "".join(f"<div style='white-space:pre;min-height:1.3em;{'background:' + colour + ';color:#1d1d1d' if bad else ''}'>"
                       f"{html.escape(l)}</div>" for l, bad in rows)
        return (f"<div style='flex:1;min-width:0'><b>{title}</b><div style='font-family:monospace;font-size:12px;"
                f"line-height:1.3;max-height:520px;overflow:auto;border:1px solid #ccc;padding:6px'>{body}</div></div>")
    return ("<div style='display:flex;gap:12px'>" + col(left, "#e3edfb", "before the edit") +
            col(right, "#fff1df", "after the edit (what the model reads)") + "</div>")


def verdict(reply: str, row: dict) -> str:
    truth = TRUTHS["edits"][row["id"]]
    original = TRUTHS["test"][row["base_id"]] if row["kind"] == "meaningful" else None
    return classify(reply, truth, row["code"], original)[0]


def show(sample_id: str | None, run_label: str):
    row = DATA["edits"].get(sample_id or "")
    if not row:
        return "", None, "", [["", ""]], ""
    before = DATA["test"][row["base_id"]]["code"]
    truth = TRUTHS["edits"][row["id"]]
    table = []
    for label, replies in SAVED.items():
        if row["id"] in replies:
            c = verdict(replies[row["id"]], row)
            table.append([label, ("right" if c in RIGHT else "UNEDITED graph" if c == "gave_original" else "wrong")
                          + f" ({c})"])
    reply = SAVED.get(run_label, {}).get(row["id"])
    if reply is None:
        return code_diff_html(before, row["code"]), draw(truth, None), f"No saved answer from {run_label}.", table, ""
    pred, _ = parse(reply)
    c = verdict(reply, row)
    cls = "verdict-right" if c in RIGHT else "verdict-wrong"
    what = ("changes the graph" if row["kind"] == "meaningful" else "leaves the graph as it was")
    text = (f"Edit **{row['edit']}** on `{row['base_id']}`: it {what}.\n\n"
            f"<span class='{cls}'>Verdict for {run_label}: {c}</span>: {CATEGORY_TEXT[c]}")
    return code_diff_html(before, row["code"]), draw(truth, pred), text, table, reply


def build():
    gr.Markdown("## Does the model follow the code? Controlled edits [D5], experiment E2\n"
                "One test program, changed a little. If the model only recognises the program, it answers the "
                "edited code with the graph from **before** the edit.")
    with gr.Accordion("the E2 result", open=False):
        gr.Markdown(E2_TABLE)
    first = list(EDITS)[0]
    first_id = edit_ids(first)[0]
    # the page opens on its first example, already drawn
    d0, i0, t0, tab0, r0 = show(first_id, list(SAVED)[0])
    with gr.Row():
        edit = gr.Dropdown(list(EDITS), value=first, label="kind of edit", scale=3)
        sample = gr.Dropdown(edit_ids(first), value=first_id, label="edited program", scale=2)
        run_label = gr.Dropdown(list(SAVED), value=list(SAVED)[0], label="saved answer from", scale=3)
    text = gr.Markdown(t0)
    with gr.Tabs():
        with gr.Tab("Graph: after the edit (left) vs the model"):
            image = gr.Image(i0, type="filepath", height=560, label="true graph after the edit (left), model (right)")
        with gr.Tab("Code: before and after"):
            diff = gr.HTML(d0)
        with gr.Tab("Every model on this edit"):
            table = gr.Dataframe(tab0, headers=["model", "answer"], interactive=False)
        with gr.Tab("Raw reply"):
            reply = gr.Code(r0, lines=16)
    edit.change(lambda e: gr.update(choices=edit_ids(e), value=(edit_ids(e) or [None])[0]), edit, sample)
    outs = [diff, image, text, table, reply]
    sample.change(show, [sample, run_label], outs)
    run_label.change(show, [sample, run_label], outs)

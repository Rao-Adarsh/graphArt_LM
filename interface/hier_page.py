"""Page 3, the hierarchical format: a module tree plus one small wiring graph per class.

Saved answers: the tuned 1.5B [M8] and the prompted 14B [M9], each with its own tree and
with the true tree given. Live: the tuned 1.5B, tree first, then one wiring call per block
class, joined by code (pipeline/run_hier.py, here with streaming).

Only hier_dev and hier_gen_test are read. The locked sets (hier_holdout, hier_timm_test)
are never opened.
"""

from __future__ import annotations

import difflib
import html
import re
import time

import gradio as gr

from interface.common import MODEL, data, draw, find, jsonl
from graphart.hier import HierError, _child, _parse_graph, _parse_tree, hier_from_code, inline, tree_text
from graphart.hier_score import instances, same_wiring, score_answer
from graphart.hier_tasks import TREE_PROMPT, WIRE_PROMPT, _fill, class_source, instance_subtree
from graphart.reduce import reduce_source
from pipeline.run_hier import block_tags, clean, graph_lines, tree_lines
from pipeline.run_prompting import load_prompt

SETS = {"torchvision dev: 21 blocks + 30 whole networks [D7]": "hier_dev",
        "generated test, generator v3: 24 blocks + 51 networks [D8]": "hier_gen_test"}
_K = "kaggle_results/graphart_hier_eval/results/hf__qwen2.5-coder-1.5b-graphart-lora-hier__"
_K14 = "kaggle_results/graphart_hier_14b/results/ollama__qwen2.5-coder-14b__"
OWN = "tuned 1.5B, its own tree [M8]"
RUNS = {OWN: _K + "hier__{split}.jsonl",
        "tuned 1.5B, true tree given (wiring alone)": _K + "hier_oracle_tree__{split}.jsonl",
        # the run's own file keeps an empty reply where the tree did not parse; this copy keeps
        # the tree the 14B wrote, so the page can show it (it scores the same: unreadable)
        "prompted 14B, 2 worked examples [M9]": "kaggle_results/graphart_hier_14b/results/"
                                                "_14b_own_with_raw_trees.jsonl?hier_dev",
        "prompted 14B, true tree given": _K14 + "hier_oracle_tree_2shot__{split}.jsonl"}
SAVED_MODE, LIVE_MODE = "saved answer (instant)", "run the 1.5B live on this laptop"

DATA = {s: {r["id"]: r for r in data(s)} for s in SETS.values()}
def _run_file(pat: str, split: str):
    if "?" in pat:                   # a file for one split only: "<path>?hier_dev"
        path, only = pat.split("?")
        return find(path) if split == only else None
    return find(pat.format(split=split))


SAVED = {s: {label: replies for label, pat in RUNS.items()
             if (replies := {r["id"]: r["reply"] for r in jsonl(_run_file(pat, s))})}
         for s in SETS.values()}

CATEGORY = {
    "right": ("Right", "expanded by code, it is exactly the graph torch.fx traces."),
    "parse": ("Unreadable", "the reply could not be read as TREE + GRAPHs."),
    "tree": ("Tree wrong", "the classes or the nesting of the module tree differ."),
    "params": ("Layer settings wrong", "the tree has the right shape, but a layer's parameters differ."),
    "variants": ("Variants wrong", "the tree is right, but the blocks are grouped into the wrong variants."),
    "wiring": ("Wiring wrong", "the tree is right, but a class's wiring graph is wrong."),
    "assembly": ("Assembly differs", "every part matches, but the expanded graph does not."),
}


KINDS = {"blocks": "block", "whole networks": "network"}


def sample_choices(split: str, kind: str = "blocks") -> list[tuple[str, str]]:
    rows = [r for r in DATA[split].values() if r["kind"] == KINDS[kind]]
    return [(f"{r['id']}  ({r.get('flat_nodes', '?')} fx nodes)", r["id"])
            for r in sorted(rows, key=lambda r: (r.get("flat_nodes") or 0, r["id"]))]


def run_labels(split: str) -> list[str]:
    return list(SAVED[split])


# --------------------------------------------------------------------------
# reading an answer, part by part, so one bad graph does not hide the rest
# --------------------------------------------------------------------------
def read_answer(reply: str) -> tuple[dict | None, list[str], dict[str, list[dict]], list[str]]:
    """(tree or None, the tree lines as written, tag -> wiring nodes, problems)."""
    problems, tree, graphs = [], None, {}
    lines = tree_lines(reply)
    try:
        tree = _parse_tree(lines)
    except (HierError, ValueError) as e:
        problems.append(f"tree: {e}")
    body = clean(reply)
    heads = [i for i, l in enumerate(body) if l.startswith("GRAPH")]
    for k, i in enumerate(heads):
        tag = body[i].split(None, 1)[1].strip() if len(body[i].split()) > 1 else "?"
        end = heads[k + 1] if k + 1 < len(heads) else len(body)
        try:
            graphs[tag] = _parse_graph(body[i + 1:end])
        except (HierError, ValueError) as e:
            problems.append(f"GRAPH {tag}: {e}")
    return tree, lines, graphs, problems


def _instance(tree: dict, path: str) -> dict | None:
    return tree if path == "" else _child(tree, path)


def drawable(nodes: list[dict], inst: dict | None) -> dict:
    """A wiring graph in the flat schema, so graphart.render can draw it. A child module's
    box shows its class (from the tree); argument slots label the edges."""
    op = {"input": "placeholder", "output": "output", "module": "call_module", "function": "call_function",
          "method": "call_method", "get_attr": "get_attr"}
    gn, slots = [], {}
    for i, n in enumerate(nodes):
        cls, params = "", {}
        if n["kind"] == "module" and inst is not None:
            c = _child(inst, n["target"])
            if c is not None:
                cls = c["cls"]
                params = c["params"] if c["kind"] == "leaf" else {}
        gn.append({"id": i, "op": op.get(n["kind"], "get_attr"),
                   "target": n["target"] + ("*" if n.get("pack") else ""), "cls": cls, "params": params,
                   "shape": None})
        for slot, tok in enumerate(n.get("src", [])):
            m = re.match(r"\d+", tok)
            if m and int(m.group()) < len(nodes):
                slots.setdefault((int(m.group()), i), []).append(slot)
    return {"nodes": gn, "edges": [{"src": a, "dst": b, "args": s} for (a, b), s in sorted(slots.items())]}


def tree_diff_html(truth_lines: list[str], model_lines: list[str]) -> str:
    """Two columns; red: a truth line the model does not have, orange: a line it invented."""
    sm = difflib.SequenceMatcher(a=truth_lines, b=model_lines, autojunk=False)
    left, right = [], []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        for l in truth_lines[a0:a1]:
            left.append((l, op != "equal"))
        for l in model_lines[b0:b1]:
            right.append((l, op != "equal"))

    def col(rows, colour, title):
        body = "".join(
            f"<div style='white-space:pre;{'background:' + colour + ';color:#1d1d1d;' if bad else ''}'>"
            f"{html.escape(l) or '&nbsp;'}</div>"
            for l, bad in rows)
        return (f"<div style='flex:1;min-width:0;overflow-x:auto'><b>{title}</b>"
                f"<div style='font-family:monospace;font-size:12px;line-height:1.35;"
                f"max-height:420px;overflow:auto;border:1px solid #ccc;padding:6px'>{body}</div></div>")
    return ("<div style='display:flex;gap:12px'>" + col(left, "#fdeaea", "true tree (red: missed)") +
            col(right, "#fff1df", "model's tree (orange: different)") + "</div>")


# --------------------------------------------------------------------------
# live: tree call, then one wiring call per block class
# --------------------------------------------------------------------------
def live_reply(code: str, tree_cap: int, wire_cap: int):
    """Yields (status, text so far, final reply or None)."""
    (tsys, ttpl), (wsys, wtpl) = load_prompt(TREE_PROMPT), load_prompt(WIRE_PROMPT)
    reduced = reduce_source(code)
    t0, last = time.time(), 0.0
    if MODEL.net is None:
        yield "Loading Qwen2.5-Coder 1.5B and the adapters (first run only, about 30 s) ...", "", None
    tree_reply = ""
    for tree_reply in MODEL.stream("hier", tsys, _fill(ttpl, code=reduced.strip()), tree_cap):
        if time.time() - last > 0.4:
            last = time.time()
            yield f"Writing the module tree ... {time.time() - t0:.0f} s", tree_reply, None
    lines = tree_lines(tree_reply)
    try:
        tree = _parse_tree(lines)
    except (HierError, ValueError) as e:
        yield f"The tree could not be read ({e}); no wiring calls.", tree_reply, "TREE\n" + "\n".join(lines)
        return
    h = {"tree": tree, "variants": {}, "problems": []}
    tags = [t for t in block_tags(tree) if class_source(reduced, t.split("#")[0])]
    done = "TREE\n" + "\n".join(lines)
    for k, tag in enumerate(tags, 1):
        user = _fill(wtpl, cls=tag.split("#")[0], code=class_source(reduced, tag.split("#")[0]).strip(),
                     subtree=instance_subtree(h, tag), tag=tag)
        g = ""
        for g in MODEL.stream("hier", wsys, user, wire_cap):
            if time.time() - last > 0.4:
                last = time.time()
                yield (f"Wiring {tag} ({k} of {len(tags)}) ... {time.time() - t0:.0f} s",
                       done + f"\n\nGRAPH {tag}\n" + g, None)
        done += f"\n\nGRAPH {tag}\n" + "\n".join(graph_lines(g))
    yield f"Done in {time.time() - t0:.0f} s: 1 tree call, {len(tags)} wiring call(s).", done, done


# --------------------------------------------------------------------------
# one request
# --------------------------------------------------------------------------
def _yes(b) -> str:
    return "yes" if b else "no"


def show(reply: str, truth: dict | None, source: str):
    """Everything the right-hand side shows for one answer."""
    tree, mlines, graphs, problems = read_answer(reply)
    lines = [f"**Answer:** {source}."]
    rows, classes = [], []
    if truth is not None:
        s = score_answer(reply, truth)
        name, text = CATEGORY[s["category"]]
        cls = "verdict-right" if s["category"] == "right" else "verdict-wrong"
        lines.append(f"<span class='{cls}'>{name}</span>: {text}")
        rows = [["reads as TREE + GRAPHs", _yes(s["parse"])],
                ["tree: classes and nesting", _yes(s["tree_struct"])],
                ["tree: with layer settings", _yes(s["tree_exact"])],
                ["tree modules F1", f"{s['tree_f1']:.2f}"],
                ["blocks grouped into the right variants", _yes(s["variants"])],
                ["class wiring graphs right", f"{s['wiring_right']} of {s['wiring_total']}"],
                ["the whole drawing right (tree + variants + every graph)", _yes(s["hier_right"])],
                ["expanded = the fx graph (exact)", _yes(s["flat_exact"])],
                ["expanded = the fx graph (layer settings ignored)", _yes(s["flat_topo"])]]
        ti = instances(truth["tree"])
        pi = instances(tree) if tree else {}
        first = {}
        for path, t in ti.items():
            first.setdefault(t, path)
        for t, path in first.items():
            if t in truth["variants"] and not truth["variants"][t]["hidden"]:
                p = pi.get(path)
                ok = p in graphs and same_wiring(graphs[p], truth["variants"][t]["nodes"])
                classes.append((f"{'✓' if ok else '✗'} {t}  at {path or '(top)'}", t, path, p))
        truth_lines = tree_text(truth).splitlines()
    else:
        lines.append("No reference: torch.fx could not trace this code, so nothing is scored.")
        truth_lines = []
        for path, p in (instances(tree) if tree else {}).items():
            if p in graphs and all(c[3] != p for c in classes):
                classes.append((f"{p}  at {path or '(top)'}", None, path, p))
    if problems:
        lines.append("**Could not read:** " + "; ".join(problems[:3]))
    state = {"reply": reply, "truth": truth, "classes": classes}
    choices = [(c[0], i) for i, c in enumerate(classes)]
    return ("\n\n".join(lines), rows or [["(nothing scored)", ""]], tree_diff_html(truth_lines, mlines),
            gr.update(choices=choices, value=0 if choices else None), class_figure(state, 0 if choices else None),
            reply, state)


def class_figure(state: dict | None, idx):
    if not state or idx is None or idx >= len(state["classes"]):
        return None
    _, t, path, p = state["classes"][idx]
    truth = state["truth"]
    tree, _, graphs, _ = read_answer(state["reply"])
    tg = pg = None
    if truth is not None and t is not None:
        tg = drawable(truth["variants"][t]["nodes"], _instance(truth["tree"], path))
    if p in graphs:
        pg = drawable(graphs[p], _instance(tree, path) if tree else None)
    return draw(tg, pg, title_p=f"model's graph for {p}")


def run(code: str, mode: str, run_label: str, example: dict | None, tree_cap: int, wire_cap: int):
    code = (code or "").strip()
    if not code:
        raise gr.Error("Pick a program, or paste code.")
    from_example = bool(example) and example["code"].strip() == code
    empty = ("", [["", ""]], "", gr.update(choices=[], value=None), None, "", None)
    yield ("Reading the reference ...", *empty)
    truth, note = None, ""
    if from_example:
        truth = DATA[example["split"]][example["id"]]["hier"]
    else:
        try:
            truth, _ = hier_from_code(code)
        except Exception as exc:                  # pasted code: anything can go wrong
            note = f" (no reference: {type(exc).__name__}: {exc})"
    if mode == SAVED_MODE:
        if not from_example:
            raise gr.Error("Saved answers exist only for the listed programs, unchanged. Choose the live run.")
        saved = SAVED[example["split"]].get(run_label, {})
        if example["id"] not in saved:
            raise gr.Error(f"'{run_label}' has no saved answer for {example['id']}.")
        yield ("Done.", *show(saved[example["id"]], truth, f"saved answer of **{run_label}**"))
        return
    if MODEL.missing():
        raise gr.Error("Adapters not found: " + ", ".join(MODEL.missing()))
    reply = ""
    for status, text, final in live_reply(code, int(tree_cap), int(wire_cap)):
        reply = final if final is not None else reply
        if final is None:
            yield (status + note, "", [["", ""]], "", gr.update(), None, text, None)
        else:
            yield (status + note, *show(final, truth, "live answer of the tuned 1.5B"))


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------
INTRO = """
## Hierarchical format: a module tree + one wiring graph per class
Real networks are long when every repeated block is written out (median 177 fx nodes, up to 548).
Here the answer is split: a **tree** of modules (what is inside what, how many copies, `0-2: Block#1`)
and a small **wiring graph per class**, each written once. Code expands it back into the fx graph,
so torch.fx stays the answer key.
"""


def build():
    gr.Markdown(INTRO)
    first = list(SETS)[0]
    # the page opens on the first block, already drawn
    sid = sample_choices(SETS[first])[0][1]
    ex0 = {"split": SETS[first], "id": sid, "code": DATA[SETS[first]][sid]["code"]}
    v0, m0, h0, _, img0, r0, s0 = show(SAVED[ex0["split"]][OWN][sid], DATA[ex0["split"]][sid]["hier"],
                                       f"saved answer of **{OWN}**")
    c0 = [(c[0], i) for i, c in enumerate(s0["classes"])]
    example = gr.State(ex0)
    state = gr.State(s0)
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.Tab("Programs"):
                    ex_set = gr.Dropdown(list(SETS), value=first, label="set")
                    ex_kind = gr.Radio(list(KINDS), value="blocks", label="programs", show_label=False)
                    ex_id = gr.Dropdown(sample_choices(SETS[first]), value=sid, label="program (smallest first)")
                with gr.Tab("Paste your own"):
                    gr.Markdown("Paste a module into the code box below and use the live run. Define the "
                                "classes and create one instance as `model`; `example_input` adds shapes.")
            code = gr.Code(ex0["code"], language="python", label="code", lines=14, max_lines=26)
            mode = gr.Radio([SAVED_MODE, LIVE_MODE], value=SAVED_MODE, label="answer")
            run_label = gr.Dropdown(run_labels(SETS[first]), value=OWN, label="saved answer from")
            with gr.Accordion("live run settings", open=False, visible=False) as live_box:
                gr.Markdown("A block takes about half a minute on this CPU; a whole network several minutes. "
                            "The caps stop a runaway answer.")
                tree_cap = gr.Slider(200, 4000, value=1500, step=100, label="tree: max new tokens")
                wire_cap = gr.Slider(100, 2000, value=800, step=100, label="each wiring graph: max new tokens")
            go = gr.Button("Draw the answer", variant="primary")
        with gr.Column(scale=8):
            status = gr.Markdown("")
            verdict = gr.Markdown(v0)
            with gr.Tabs():
                with gr.Tab("Wiring graph per class"):
                    cls_pick = gr.Dropdown(c0, value=0 if c0 else None,
                                           label="class (✓ right, ✗ wrong); its first instance in the tree")
                    image = gr.Image(img0, label="true wiring (left) and the model's (right)", type="filepath",
                                     height=560)
                with gr.Tab("Module tree"):
                    tree_html = gr.HTML(h0)
                with gr.Tab("Scores"):
                    metrics = gr.Dataframe(m0, headers=["part", "value"], interactive=False)
                with gr.Tab("Raw reply"):
                    reply_text = gr.Code(r0, label="the model's reply (streams in during a live run)", lines=20)

    mode.change(lambda m: (gr.update(visible=m == SAVED_MODE), gr.update(visible=m == LIVE_MODE)),
                mode, [run_label, live_box])

    def pick_set(lbl, kind):
        s = SETS[lbl]
        return gr.update(choices=sample_choices(s, kind), value=None), gr.update(choices=run_labels(s), value=OWN)
    ex_set.change(pick_set, [ex_set, ex_kind], [ex_id, run_label])
    ex_kind.change(lambda lbl, kind: gr.update(choices=sample_choices(SETS[lbl], kind), value=None),
                   [ex_set, ex_kind], ex_id)

    def load(lbl, sid):
        s = SETS[lbl]
        row = DATA[s].get(sid or "")
        if not row:
            return gr.update(), None
        return row["code"], {"split": s, "id": sid, "code": row["code"]}
    picked = ex_id.change(load, [ex_set, ex_id], [code, example])

    outputs = [status, verdict, metrics, tree_html, cls_pick, image, reply_text, state]
    go.click(run, [code, mode, run_label, example, tree_cap, wire_cap], outputs)

    def redraw(code_, mode_, run_label_, example_, tc, wc):
        if not (example_ and code_ and mode_ == SAVED_MODE and code_.strip() == example_["code"].strip()):
            yield tuple(gr.update() for _ in outputs)
            return
        yield from run(code_, mode_, run_label_, example_, tc, wc)
    picked.then(redraw, [code, mode, run_label, example, tree_cap, wire_cap], outputs)
    run_label.change(redraw, [code, mode, run_label, example, tree_cap, wire_cap], outputs)
    cls_pick.change(class_figure, [state, cls_pick], image)

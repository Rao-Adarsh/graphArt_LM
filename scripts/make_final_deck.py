"""Build the IE 643 final-presentation deck (Stage 1).

    .venv/Scripts/python scripts/make_final_deck.py

Writes docs/presentation/Pensieve_IE643_CourseProject_Final.pptx: 18 slides plus
references, in the order the course instructions give. Visual-first: every slide is
built around one figure, diagram or number, with a few words; the talking points
live in the speaker notes, timed to the 12-minute limit.

Figures come from scripts/make_report_figures.py (run it first), so the slides and
the report always show the same numbers. Look and feel are the prep deck's
(scripts/make_slides.py), imported rather than copied.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import make_slides as ms
from make_slides import (ACCENT, GREY, INK, LIGHT, MARGIN, MONO_FONT, NAVY, RULE, SLIDE_H,
                         SLIDE_W, WHITE, WIRE1, _arrowhead, _caption, _line, _para, _textbox,
                         chips, code_box, draw_circuit, draw_dag, draw_strip, footer, new_slide,
                         notes)

TITLE = "Making Large Models Draw Topological Graphs"
TEAM, MEMBERS = "Pensieve", [("Adarsh A Rao", "24B2132")]
FIG, DOCFIG, DECKFIG = ROOT / "report" / "figures", ROOT / "docs" / "figures", ROOT / "report" / "deck_figures"
OUT = ROOT / "docs" / "presentation" / f"{TEAM}_IE643_CourseProject_Final.pptx"
WARM = RGBColor(0xFD, 0xF3, 0xE3)
GOOD = RGBColor(0x1B, 0xAF, 0x7A)

ms.TEAM_NAME = TEAM            # the shared footer reads it


# --------------------------------------------------------------------------
# small visual pieces
# --------------------------------------------------------------------------
def pic(slide, path, left, top, *, width=None, height=None):
    kw = {"width": Inches(width)} if width else {"height": Inches(height)}
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), **kw)


def graph_only(slide, path, left, top, height, *, src=(2580, 1460)):
    """Crop a docs/figures code-and-graph figure down to its graph, which sits in the
    middle-right of the image, and keep the cropped picture's true aspect ratio."""
    l, r, t, b = 0.45, 0.08, 0.08, 0.01
    p = pic(slide, path, left, top, height=height)
    p.crop_left, p.crop_right, p.crop_top, p.crop_bottom = l, r, t, b
    p.width = Inches(height * ((1 - l - r) * src[0]) / ((1 - t - b) * src[1]))
    return p


def box(slide, x, y, w, h, *, fill=LIGHT, edge=RULE, width=1.0, radius=0.10):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = edge
    shp.line.width = Pt(width)
    shp.shadow.inherit = False
    shp.adjustments[0] = radius
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08)
    return shp, tf


def stat(slide, x, y, w, h, number, label, *, color=NAVY, fill=LIGHT, edge=RULE, size=34):
    """A hero number with a short label under it."""
    _, tf = box(slide, x, y, w, h, fill=fill, edge=edge, radius=0.08)
    _para(tf, number, size, bold=True, color=color, align=PP_ALIGN.CENTER, first=True)
    _para(tf, label, 12, color=GREY, align=PP_ALIGN.CENTER, space_before=2)


def badge(slide, x, y, n, *, color=WIRE1):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.42), Inches(0.42))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _para(tf, str(n), 14, bold=True, color=WHITE, align=PP_ALIGN.CENTER, first=True)


def callout(slide, x, y, w, n, head, sub):
    badge(slide, x, y + 0.04, n)
    tf = _textbox(slide, x + 0.58, y, w - 0.58, 0.8)
    _para(tf, head, 15, bold=True, color=INK, first=True)
    _para(tf, sub, 12, color=GREY, space_before=1)


def arrow(slide, x1, y1, x2, y2, color=ACCENT, width=2.0):
    ln = _line(slide, x1, y1, x2, y2, color, width)
    _arrowhead(ln, color, width)


def label(slide, x, y, w, text, *, size=12.5, color=ACCENT, bold=True, align=PP_ALIGN.LEFT):
    _caption(slide, x, y, w, text, size=size, color=color, bold=bold, align=align)


# --------------------------------------------------------------------------
# the slides
# --------------------------------------------------------------------------
def s_title(prs):
    slide = new_slide(prs)
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), Inches(SLIDE_H))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = NAVY
    stripe.line.fill.background()
    tf = _textbox(slide, 1.0, 1.15, 7.6, 0.5)
    _para(tf, "IE 643 COURSE PROJECT  ·  FINAL PRESENTATION", 13, bold=True, color=ACCENT, first=True)
    tf = _textbox(slide, 1.0, 1.65, 6.7, 2.4)
    _para(tf, TITLE, 38, bold=True, color=NAVY, first=True)
    tf = _textbox(slide, 1.0, 4.05, 6.7, 0.9)
    _para(tf, "Can a language model read a network's code and draw its forward-pass graph "
              "— without running it?", 16, color=GREY, first=True)
    tf = _textbox(slide, 1.0, 5.25, 7.0, 1.4)
    _para(tf, "Team " + TEAM, 20, bold=True, color=NAVY, first=True)
    for name, roll in MEMBERS:
        _para(tf, f"{name}  ·  Roll No. {roll}", 15, color=INK, space_before=3)
    _para(tf, "IIT Bombay  ·  October 2026", 12, color=GREY, space_before=8)
    # the task in one picture: the graph half of the tier-3 figure
    graph_only(slide, DOCFIG / "tier3.png", 8.05, 0.85, 5.75)
    notes(slide, """
[0:00-0:20]
• Team Pensieve; Adarsh Rao.
• The question: give a model a network's source code; can it draw the forward-pass graph, without running it?
• The picture on the right is one such graph, traced from code.
""")
    return slide


def s_outline(prs):
    slide = new_slide(prs, "Outline")
    items = [("1", "The task"), ("2", "Workflow"), ("3", "Before & after\nthe review"), ("4", "Approach"),
             ("5", "Data"), ("6", "Setup"), ("7", "Results"), ("8", "Conclusions\n& next")]
    for i, (n, text) in enumerate(items):
        x, y = 0.9 + (i % 4) * 3.0, 2.0 + (i // 4) * 2.25
        _, tf = box(slide, x, y, 2.65, 1.75, fill=LIGHT if i != 6 else WARM, edge=RULE if i != 6 else WIRE1)
        _para(tf, n, 30, bold=True, color=ACCENT if i != 6 else WIRE1, align=PP_ALIGN.CENTER, first=True)
        _para(tf, text, 15, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    notes(slide, """
[0:20-0:40]
• Eight parts; most of the time goes to results (7).
• Arc: the task -> how it was built -> what the models did -> what it means.
""")
    return slide


def s_task(prs):
    slide = new_slide(prs, "Code in, graph out — no execution", kicker="the task")
    code_box(slide, 0.55, 1.95, 4.15, 4.95,
             "def forward(self, x):\n    x = self.fc1(x)\n    x = F.sigmoid(x)\n    if self.wide:      # False\n"
             "        x = self.fc2(x)\n    x = self.fc3(x)\n    x = F.sigmoid(x)\n    x = self.fc3(x)\n"
             "    p = self.fc4(x)\n    a = self.fc5(x)\n    x = torch.cat([p, a], dim=1)\n    x = self.fc6(x)\n"
             "    return x", size=12.5, caption="what the model sees")
    arrow(slide, 4.85, 4.4, 5.35, 4.4)
    label(slide, 5.45, 1.62, 4.4, "what it must produce", size=12)
    graph_only(slide, DOCFIG / "tier3.png", 5.45, 1.95, 4.95)
    chips_y = 1.75
    for i, (head, sub) in enumerate([("no execution", "read the code only"),
                                     ("exact answer", "torch.fx traces the truth"),
                                     ("order matters", "a − b ≠ b − a: edges carry argument slots"),
                                     ("code hides it", "loops, dead branches, reused layers")]):
        _, tf = box(slide, 10.2, chips_y + i * 1.3, 2.6, 1.08, fill=LIGHT if i else WARM,
                    edge=RULE if i else WIRE1)
        _para(tf, head, 14.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True)
        _para(tf, sub, 11, color=GREY, align=PP_ALIGN.CENTER, space_before=1)
    notes(slide, """
[0:40-1:30]
• Left: the source a model sees. Right: the graph it must write — every operation, and which value feeds which argument.
• No execution allowed: the model reads, like a person reviewing code.
• Ground truth is free and exact: PyTorch's torch.fx traces the real graph.
• Why it is hard (point at the code): 'if self.wide' never runs, so fc2 is NOT in the graph; fc3 is called twice, so it appears twice; the cat merges two branches — argument order matters.
""")
    return slide


def s_workflow(prs):
    slide = new_slide(prs, "Workflow", kicker="from problem statement to results")
    pic(slide, FIG / "workflow.png", 0.75, 1.75, width=11.85)
    dates = [("4 Sep", "plan"), ("20 Sep", "prep review"), ("16–17 Oct", "PyTorch rebuild"),
             ("17–22 Oct", "prompting"), ("23–25 Oct", "fine-tuning"), ("25–26 Oct", "closing tests")]
    for i, (d, what) in enumerate(dates):
        x = 0.95 + i * 1.95
        _, tf = box(slide, x, 5.05, 1.75, 0.95, fill=WHITE, edge=RULE)
        _para(tf, d, 14, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True)
        _para(tf, what, 11, color=GREY, align=PP_ALIGN.CENTER)
        if i:
            arrow(slide, x - 0.18, 5.52, x - 0.03, 5.52, RULE, 1.5)
    notes(slide, """
[1:30-2:10]
• Problem -> domain -> data -> harness -> prompting -> fine-tuning -> closing tests.
• The pivot (second box): the prep review used quantum circuits; corrected to PyTorch, as the task asks.
• Blue boxes = experiments; everything before them is the measuring instrument.
• Dates along the bottom: the experiments themselves took about ten days.
""")
    return slide


def s_before(prs):
    slide = new_slide(prs, "Before the review: quantum-circuit graphs", kicker="work before the prep review")
    label(slide, 0.75, 1.8, 4.6, "a Qiskit circuit…")
    draw_circuit(slide, 1.35, 2.35, annotate=False)
    arrow(slide, 6.1, 2.85, 6.8, 2.85)
    label(slide, 7.05, 1.8, 5.2, "…and its DAG (exact truth from Qiskit)")
    draw_dag(slide, 7.05, 2.2)
    chips(slide, 0.75, 4.35, ["tiered generator", "DAG text format", "validator + metrics", "prompting pipeline"],
          w=2.8, h=0.6, gap=0.2, size=13)
    _, tf = box(slide, 0.75, 5.35, 11.8, 0.95, fill=WARM, edge=WIRE1, width=1.4, radius=0.15)
    _para(tf, "TA correction:  the task is a neural network's forward pass  →  rebuilt on PyTorch + torch.fx",
          16, bold=True, color=INK, align=PP_ALIGN.CENTER, first=True)
    notes(slide, """
[2:10-2:50]
• Before the review: the same question on quantum circuits — Qiskit gives the exact DAG for free.
• Built then: the generator, the graph text format, a validator, metrics, a prompting pipeline.
• The review: the TA pointed out the task is a deep network's forward pass.
• What carried over: compare GRAPHS (isomorphism), not lists; the whole harness was domain-agnostic, so the rebuild took two days.
""")
    return slide


def s_after(prs):
    slide = new_slide(prs, "After the review: rebuilt, then measured", kicker="work after the prep review")
    draw_strip(slide, 0.75, 1.95, [("ground truth", "torch.fx\ntrace"), ("data", "1,600 programs\n4 tiers"),
                                   ("phase 1", "4 models\n6 prompts"), ("phase 2", "LoRA on\na 1.5B model"),
                                   ("closing", "is it\nunderstanding?")], w=1.95, h=1.35, gap=0.53)
    for i, (n, lab) in enumerate([("1,600", "programs"), ("4", "models, 1.5B–31B"), ("6", "prompts"),
                                  ("97 min", "fine-tuning, 1 free GPU")]):
        stat(slide, 0.75 + i * 3.02, 4.15, 2.75, 1.65, n, lab)
    notes(slide, """
[2:50-3:25]
• After the review: rebuilt around torch.fx, built a 1,600-program dataset, then three rounds of experiments.
• Phase 1: prompting four models with six prompts. Phase 2: fine-tuning a small model.
• Closing: when fine-tuning scored perfectly, I went back to ask whether that is understanding or memorisation.
""")
    return slide


def s_grading(prs):
    slide = new_slide(prs, "Exact grading, for free", kicker="approach · formulation")
    code_box(slide, 0.6, 1.95, 3.6, 2.2, "def forward(self, x):\n    h = self.fc1(x)\n    h = torch.relu(h)\n"
             "    h = self.fc2(h) + x\n    return self.out(h)", size=12, caption="source")
    arrow(slide, 4.3, 3.05, 5.0, 3.05)
    label(slide, 4.15, 2.55, 1.0, "fx", size=12, align=PP_ALIGN.CENTER)
    code_box(slide, 5.1, 1.95, 3.75, 2.2, "0 placeholder x\n1 call_module fc1\n2 call_function relu\n"
             "3 call_module fc2\n4 call_function add\n…\n0 -> 4 : arg1", size=11.5, caption="reference graph")
    _, tf = box(slide, 9.4, 1.95, 3.3, 2.2, fill=WHITE, edge=ACCENT, width=1.4)
    _para(tf, "model's reply", 13, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True)
    _para(tf, "↓ parse   ↓ validate", 12, color=GREY, align=PP_ALIGN.CENTER, space_before=6)
    _para(tf, "labelled graph\nisomorphism", 14, bold=True, color=ACCENT, align=PP_ALIGN.CENTER, space_before=6)
    arrow(slide, 8.95, 3.05, 9.35, 3.05, ACCENT)
    metrics = [("valid", "well-formed, no answer needed"), ("exact", "same graph, strictly"),
               ("topo", "same graph, notation ignored"), ("F1", "partial credit")]
    for i, (m, sub) in enumerate(metrics):
        _, tf = box(slide, 0.6 + i * 3.05, 4.75, 2.85, 1.35, fill=LIGHT if m != "topo" else WARM,
                    edge=RULE if m != "topo" else WIRE1)
        _para(tf, m, 22, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True, font=MONO_FONT)
        _para(tf, sub, 11.5, color=GREY, align=PP_ALIGN.CENTER)
    notes(slide, """
[3:25-4:15]
• Formulation: source in, graph text out. Edges name the argument slot — 'x feeds the add's second input'.
• torch.fx gives the reference; the reply is parsed, checked for structural sanity without the answer, then compared by labelled graph isomorphism — order of lines does not matter, structure does.
• Four scores: valid; exact (strict); topo — exact but ignoring notation like parameter spelling, added after the first reply failed on notation, not structure; F1 for partial credit.
""")
    return slide


def s_prompting(prs):
    slide = new_slide(prs, "Six prompts — one of them ours", kicker="approach · prompting")
    prompts = [("zero-shot", "rules only"), ("few-shot", "+ 3 worked examples"), ("CoT", "think step by step"),
               ("CoS  ★", "simulate the code"), ("nodes only", "list the operations"),
               ("edges only", "wire given operations")]
    for i, (p, sub) in enumerate(prompts):
        x, y = 0.6 + (i % 2) * 2.85, 1.85 + (i // 2) * 1.45
        ours = p.startswith("CoS")
        _, tf = box(slide, x, y, 2.6, 1.2, fill=WARM if ours else LIGHT, edge=WIRE1 if ours else RULE,
                    width=1.4 if ours else 1.0)
        _para(tf, p, 16, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True)
        _para(tf, sub, 11, color=GREY, align=PP_ALIGN.CENTER)
    code_box(slide, 6.45, 2.1, 6.3, 1.45,
             "h1 = self.fc_1(x)    ->  n1 fc_1   <- n0\n"
             "h2 = func.gelu(h1)   ->  n2 gelu   <- n1\n"
             "h3 = h1 + h2         ->  n3 add    <- n1 arg0, n2 arg1\n"
             "return h3            ->  n4 output <- n3",
             size=12, caption="chain-of-simulation (ours): track which node each variable holds")
    label(slide, 6.45, 4.55, 6.3, "decomposition: is the hard part reading, or wiring?", size=13)
    _, tf = box(slide, 6.45, 5.0, 3.0, 1.0)
    _para(tf, "code  →  operations", 13.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True)
    _, tf = box(slide, 9.75, 5.0, 3.0, 1.0)
    _para(tf, "operations  →  edges", 13.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True)
    notes(slide, """
[4:15-5:05]
• Six prompts, each isolating one thing. CoT is the control for ours.
• Ours, chain-of-simulation: act as the interpreter — one line per statement, recording which graph node each variable now refers to. A residual edge then falls out mechanically (n1 still feeds the add).
• Decomposition: 'nodes only' tests reading the code; 'edges only' hands over the correct operations and tests wiring alone.
""")
    return slide


def s_lora(prs):
    slide = new_slide(prs, "Fine-tuning a small model with LoRA", kicker="approach · fine-tuning")
    # frozen weight + low-rank update
    label(slide, 0.7, 1.75, 5.5, "every attention + MLP projection", size=12.5)
    _, tf = box(slide, 0.7, 2.25, 2.4, 2.4, fill=RGBColor(0xE6, 0xE9, 0xEE), edge=GREY)
    _para(tf, "W₀", 30, bold=True, color=GREY, align=PP_ALIGN.CENTER, first=True)
    _para(tf, "frozen\n1.56 B", 12, color=GREY, align=PP_ALIGN.CENTER)
    tf = _textbox(slide, 3.2, 3.15, 0.5, 0.6)
    _para(tf, "+", 30, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True)
    _, tf = box(slide, 3.75, 2.25, 0.5, 2.4, fill=WARM, edge=WIRE1)
    _para(tf, "B", 18, bold=True, color=WIRE1, align=PP_ALIGN.CENTER, first=True)
    _, tf = box(slide, 4.35, 3.2, 1.6, 0.5, fill=WARM, edge=WIRE1)
    _para(tf, "A", 18, bold=True, color=WIRE1, align=PP_ALIGN.CENTER, first=True)
    label(slide, 3.7, 4.75, 2.6, "rank 16, trained", size=12, color=WIRE1)
    stat(slide, 0.7, 5.3, 5.3, 1.25, "18.5 M  of  1.56 B", "parameters trained (1.2%)", size=26)
    # loss mask
    label(slide, 6.75, 1.75, 6.0, "loss only on the answer", size=12.5)
    tokens = [("system rules", False), ("code…", False), ("INPUTS 1", True), ("NODES", True),
              ("0 placeholder x", True)]
    x, first_scored = 6.75, None
    for text, scored in tokens:
        w = 0.095 * len(text) + 0.25
        if scored and first_scored is None:
            first_scored = x
        _, tf = box(slide, x, 2.25, w, 0.55, fill=WARM if scored else RGBColor(0xE6, 0xE9, 0xEE),
                    edge=WIRE1 if scored else GREY, radius=0.25)
        _para(tf, text, 11, color=INK, align=PP_ALIGN.CENTER, first=True, font=MONO_FONT)
        x += w + 0.08
    label(slide, 6.75, 2.92, 3.0, "masked", size=11, color=GREY, bold=False)
    label(slide, first_scored, 2.92, 3.0, "cross-entropy", size=11, color=WIRE1, bold=False)
    for i, (k, v) in enumerate([("base", "Qwen2.5-Coder 1.5B"), ("data", "1,280 programs"),
                                ("optimiser", "AdamW · lr 2e-4 · cosine"), ("budget", "2 epochs · 97 min · 1× T4")]):
        _, tf = box(slide, 6.75 + (i % 2) * 3.05, 3.65 + (i // 2) * 1.45, 2.85, 1.2, fill=WHITE, edge=RULE)
        _para(tf, k.upper(), 10, bold=True, color=ACCENT, align=PP_ALIGN.CENTER, first=True)
        _para(tf, v, 13.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER, space_before=2)
    notes(slide, """
[5:05-5:55]
• Base: Qwen2.5-Coder 1.5B, a decoder-only transformer — the model that scored 5% zero-shot.
• LoRA: the original weights W0 stay frozen; each projection learns a rank-16 update B·A. 18.5 M trainable parameters, 1.2% — fits one free 16 GB T4.
• Training data: 1,280 programs; input = the same zero-shot prompt, target = the reference graph.
• Loss: token cross-entropy on the answer only — prompt tokens masked — so it learns to answer, not to recite.
• AdamW, lr 2e-4 cosine, 2 epochs, 97 minutes; the validation split picks the checkpoint.
""")
    return slide


def s_data(prs):
    slide = new_slide(prs, "1,600 programs, four levels of hidden structure", kicker="data")
    tiers = [("T1 flat", "7–14 nodes", "tier1.png"), ("T2 loops", "8–13 nodes", "tier2.png"),
             ("T3 branches", "8–16 nodes · 2 merges", "tier3.png"), ("T4 blocks", "12–25 nodes · 2 merges", "tier4.png")]
    for i, (name, sub, img) in enumerate(tiers):
        x = 0.6 + i * 3.08
        h_img = 2.95
        p = pic(slide, DOCFIG / img, x, 2.35, height=h_img)
        src_h = 1460 if img == "tier3.png" else 1581
        p.crop_left, p.crop_top = 0.37, 0.08
        p.width = Inches(h_img * (0.63 * 2580) / (0.92 * src_h))
        label(slide, x, 1.75, 2.9, name, size=15, color=NAVY)
        label(slide, x, 2.02, 2.9, sub, size=11, color=GREY, bold=False)
    # split bar, proportional
    label(slide, 0.6, 5.5, 3.0, "split", size=12)
    for (part, n, color), x0 in zip([("train 1,280", 1280, ACCENT), ("val 160", 160, WIRE1), ("test 160", 160, GOOD)],
                                    [0, 1280, 1440]):
        shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.4 + 7.0 * x0 / 1600), Inches(5.5),
                                     Inches(7.0 * n / 1600 - 0.03), Inches(0.42))
        shp.fill.solid()
        shp.fill.fore_color.rgb = color
        shp.line.fill.background()
        shp.shadow.inherit = False
        _para(shp.text_frame, part, 10.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER, first=True)
    chips(slide, 8.7, 5.35, ["style randomised", "torch.fx truth"], w=1.95, h=0.72, gap=0.15, size=12)
    notes(slide, """
[5:55-6:40]
• Fully synthetic; my generator writes the programs, torch.fx traces each one.
• Tiers vary how much the CODE HIDES the graph, not how big it is: flat lines; loops and Sequential; branches (residuals, concatenation, dead branches, reused layers); whole blocks (ResNet, Inception, transformer, U-net).
• Style is randomised separately — class names, imports, naming — so a model cannot shortcut on house style.
• 80/10/10 per tier; duplicate graphs removed. Every result here uses the same 40 test programs, 10 per tier.
""")
    return slide


def s_setup(prs):
    slide = new_slide(prs, "Where it ran", kicker="experimental setup")
    cols = [("hardware", [("laptop", "CPU only"), ("Kaggle", "2× T4 · 16 GB"), ("OpenRouter", "free tier")]),
            ("models", [("Qwen2.5-Coder", "1.5B · 7B · 14B"), ("Gemma 4", "31B"), ("+ LoRA", "on the 1.5B")]),
            ("evaluation", [("40", "test programs"), ("20", "fresh programs"), ("193 + 21", "edits · torchvision")])]
    for c, (head, rows) in enumerate(cols):
        x = 0.75 + c * 4.0
        _, tf = box(slide, x, 1.8, 3.7, 0.6, fill=NAVY, edge=NAVY, radius=0.18)
        _para(tf, head.upper(), 13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, first=True)
        for r, (big, small) in enumerate(rows):
            _, tf = box(slide, x, 2.6 + r * 1.2, 3.7, 1.02, fill=WHITE, edge=RULE)
            _para(tf, big, 18, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True)
            _para(tf, small, 12, color=GREY, align=PP_ALIGN.CENTER)
    chips(slide, 0.75, 6.25, ["greedy decoding", "same prompt, every model", "same 40 programs"],
          w=3.75, h=0.5, gap=0.27, size=12.5)
    notes(slide, """
[6:40-7:15]
• Three places: the laptop for small local runs, free Kaggle T4 GPUs for the 7B/14B and all fine-tuning, OpenRouter's free tier for Gemma 31B. No money spent.
• Models span 1.5B to 31B; the 1.5B is also fine-tuned.
• Decoding is greedy everywhere, the same prompt and the same 40 programs for every model — so rows compare.
• Plus 20 fresh programs, and two closing sets: 193 edited programs and 21 real torchvision programs.
""")
    return slide


def s_res_scale(prs):
    slide = new_slide(prs, "Topology appears with scale — chains first", kicker="results · RQ1")
    pic(slide, FIG / "scale_by_tier.png", 0.55, 1.65, width=9.0)
    stat(slide, 9.9, 1.95, 2.85, 1.85, "0.05 → 0.93", "zero-shot topology\n1.5B → 31B", size=26)
    stat(slide, 9.9, 4.05, 2.85, 1.85, "0 / 20", "branching programs\nsolved by the 7B", size=28, color=WIRE1,
         fill=WARM, edge=WIRE1)
    notes(slide, """
[7:15-8:05]
• Share of programs with the right topology, zero-shot, by tier.
• Clear scale curve: 5% at 1.5B, 25% at 7B, 60% at 14B, 93% at Gemma 31B.
• Chains come first: the 7B handles half the flat and loop programs, but none of the branching ones (tiers 3–4). Branching first appears at 14B; the 31B solves tier 3 completely.
• Gemma's few misses: notation (x.tanh written as a function) and tuple unpacking in attention (getitem nodes).
""")
    return slide


def s_res_prompts(prs):
    slide = new_slide(prs, "Simulation helps on branches; reading and wiring both break there",
                      kicker="results · RQ2 · Qwen 14B")
    pic(slide, FIG / "prompts_by_tier.png", 0.5, 1.7, width=7.9)
    pic(slide, FIG / "decomposition.png", 8.7, 1.75, width=4.1)
    chips(slide, 0.6, 5.4, ["CoT < zero-shot", "CoS best on T3–T4", "CoS: 3/10 on loops"],
          w=2.5, h=0.62, gap=0.15, size=12.5)
    _, tf = box(slide, 8.75, 4.85, 4.0, 1.2, fill=WARM, edge=WIRE1)
    _para(tf, "0.70 × 0.75 ≈ 0.53", 18, bold=True, color=INK, align=PP_ALIGN.CENTER, first=True, font=MONO_FONT)
    _para(tf, "no extra cost for doing both", 12, color=GREY, align=PP_ALIGN.CENTER)
    notes(slide, """
[8:05-9:00]
• Left: four prompts on the 14B, by tier. Generic chain-of-thought does WORSE than plain zero-shot: twice the text, no procedure for this task, more format drift.
• Our simulation prompt is the best on the hard tiers (7 of 10 on each), where the code hides the graph. It collapses on loops (3 of 10) — it keeps a node for the Sequential container itself — which is why its total only ties few-shot.
• Few-shot wins overall by fixing FORMAT: worked examples cut unreadable replies from 21 to 4.
• Right: the decomposition. Reading alone 0.70, wiring alone 0.75, both 0.60 — about what two independent skills multiply to. So the hard part is not doing both at once; each skill breaks on branching code.
""")
    return slide


def s_res_finetune(prs):
    slide = new_slide(prs, "On our generator's code, a fine-tuned 1.5B beats a 31B", kicker="results · RQ3")
    pic(slide, FIG / "finetune_vs_prompting.png", 0.75, 1.6, width=7.3)
    stat(slide, 8.85, 1.8, 3.9, 1.35, "40 / 40", "test programs, exact", size=32, color=GOOD)
    stat(slide, 8.85, 3.3, 3.9, 1.35, "20 / 20", "fresh programs, unseen graphs", size=32, color=GOOD)
    stat(slide, 8.85, 4.8, 3.9, 1.35, "39 / 39", "without the one duplicate graph found", size=32, color=GOOD)
    label(slide, 0.6, 5.05, 7.9, "leak audit", size=12.5)
    for i, check in enumerate(["answer not in prompt", "test split never read", "no copied answers",
                               "untuned: 0 / 40"]):
        _, tf = box(slide, 0.6 + i * 1.98, 5.45, 1.85, 0.85, fill=WHITE, edge=GOOD, width=1.2)
        _para(tf, "✓", 14, bold=True, color=GOOD, align=PP_ALIGN.CENTER, first=True)
        _para(tf, check, 10.5, color=INK, align=PP_ALIGN.CENTER)
    notes(slide, """
[9:00-9:45]
• The fine-tuned 1.5B: every one of the 40 test graphs exactly right — the untuned same model got none; Gemma 31B got 30 exact, 37 with spelling forgiven (the green bars) — so the real gap is 3 programs, not 10.
• Too good, so I audited it: the answer is never in the prompt; training never read the test split; no reply copies a training answer; the same scorer gives the untuned model zero. One test graph duplicated a training graph in a different order — without it, still 39/39.
• Then 20 brand-new programs whose graphs appear nowhere in the data: 20 of 20.
• But every one of these programs came from my generator. The next slide is what happened outside it.
""")
    return slide


def s_res_qualitative(prs):
    slide = new_slide(prs, "What going wrong looks like", kicker="results · qualitative · Qwen 14B, zero-shot")
    pic(slide, DECKFIG / "qual_t3-00090.png", 0.45, 1.45, width=8.1)
    label(slide, 0.6, 6.55, 8.5, "left: reference  ·  right: the model  ·  orange/red: mismatches", size=11,
          color=GREY, bold=False)
    callout(slide, 9.3, 1.9, 3.6, 1, "dead branch written in", "fc2 sits under  if self.wide:  — never runs")
    callout(slide, 9.3, 3.35, 3.6, 2, "reused layer renamed", "fc3 called twice → invented  fc3_1")
    callout(slide, 9.3, 4.8, 3.6, 3, "method vs function", "F.sigmoid is traced as a tensor method")
    notes(slide, """
[9:45-10:20]
• A real reply — the same program as on the task slide. Left: the truth; right: the 14B's answer; colour marks every mismatch.
• Three different mistakes in one answer:
  1. It wrote fc2 into the graph, though it sits under a flag that is False — it never runs.
  2. fc3 is called twice; the model invented a new layer, fc3_1.
  3. A notation trap: F.sigmoid is recorded by fx as a tensor method.
• Errors 1 and 2 are reasoning about what code DOES — exactly what the closing experiments probe.
""")
    return slide


def s_res_closing(prs):
    slide = new_slide(prs, "Fine-tuning learned our generator — not the rule", kicker="results · RQ4–RQ6")
    pic(slide, FIG / "closing.png", 0.45, 1.6, width=8.75)
    _, tf = box(slide, 0.6, 5.15, 8.5, 0.85, fill=WARM, edge=WIRE1, width=1.2, radius=0.15)
    _para(tf, "both models answer changed wiring with the OLD graph:  64%  ·  68%", 15, bold=True, color=INK,
          align=PP_ALIGN.CENTER, first=True)
    label(slide, 0.6, 6.15, 8.6, "small groups (4 live flags, 5 in-distribution samples) · one seed · "
          "torchvision split fixed before scoring", size=10.5, color=GREY, bold=False)
    stat(slide, 9.55, 1.65, 3.25, 1.4, "16 / 18", "real-code replies drawn\nas one straight chain", size=30,
         color=WIRE1, fill=WARM, edge=WIRE1)
    stat(slide, 9.55, 3.2, 3.25, 1.4, "0 / 27", "rerouted inputs followed", size=30, color=WIRE1, fill=WARM,
         edge=WIRE1)
    stat(slide, 9.55, 4.75, 3.25, 1.4, "0 / 4", "live flags kept\n(training: always False)", size=30,
         color=WIRE1, fill=WARM, edge=WIRE1)
    notes(slide, """
[10:20-11:10]
• Three tests outside the generator. Blue: the prompted 14B. Orange: the fine-tuned 1.5B.
• Real torchvision code: the tuned model gets none right — and draws 16 of 18 replies as a straight chain; branches, concatenations and skip connections vanish. On our own programs it never does that. The 14B still draws branches, 4 of the 5 samples that look like our data.
• Controlled edits: rename a variable and both are fine. Change which value feeds a layer and both answer with the graph the code USED to have — 64% and 68%. The tuned model followed a rerouted input 0 times in 27.
• Held-out construct: trained without dead branches, it writes dead layers in. Trained with them, it deletes the layer even when the flag is True — in training, every flag was False.
• So: it learned to read my generator's code, with shortcuts the data allowed — not the rule.
""")
    return slide


def s_conclusions(prs):
    slide = new_slide(prs, "Conclusions")
    tiles = [("scale", "0.05 → 0.93", "zero-shot topology,\n1.5B → 31B"),
             ("prompting", "14 / 20", "hard tiers,\nchain-of-simulation"),
             ("fine-tuning", "40 → 0", "our test vs real code\n(of 40, of 21)"),
             ("edits", "64% · 68%", "changed wiring answered\nwith the old graph")]
    for i, (k, n, sub) in enumerate(tiles):
        x = 0.75 + i * 2.98
        _, tf = box(slide, x, 1.8, 2.8, 0.55, fill=NAVY, edge=NAVY, radius=0.2)
        _para(tf, k.upper(), 12.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER, first=True)
        stat(slide, x, 2.5, 2.8, 2.2, n, sub, size=32)
    _, tf = box(slide, 0.75, 5.1, 11.75, 1.05, fill=WARM, edge=WIRE1, width=1.4, radius=0.15)
    _para(tf, "A perfect score and a clean leak audit did not show understanding — probes did", 17,
          bold=True, color=INK, align=PP_ALIGN.CENTER, first=True)
    notes(slide, """
[11:10-11:40]
• Scale: the ability grows with size, branching last.
• Prompting: simulating the code beats generic thinking on the hard tiers; examples beat instructions for format.
• Fine-tuning: perfect on our generator's code, nothing on real code — it learned the family, with its shortcuts.
• Both models answer changed wiring with the familiar graph.
• The lesson: 40 out of 40 and a clean leak audit were not evidence of understanding; edits, real code and a held-out construct — each with exact ground truth — were.
""")
    return slide


def s_novelty(prs):
    slide = new_slide(prs, "Next: take the shortcuts away", kicker="plan for novelty assessment")
    cards = [("1", "live flags", "flags that are sometimes\nTrue, not always False"),
             ("2", "varied code", "input names, helper classes\n& functions, grouped convs"),
             ("3", "edit pairs", "training pairs that differ\nonly in their wiring")]
    for i, (rq, head, sub) in enumerate(cards):
        x = 0.75 + i * 4.0
        _, tf = box(slide, x, 1.8, 3.7, 3.1, fill=WHITE, edge=ACCENT, width=1.6, radius=0.06)
        _para(tf, rq, 30, bold=True, color=ACCENT, align=PP_ALIGN.CENTER, first=True)
        _para(tf, head, 18, bold=True, color=NAVY, align=PP_ALIGN.CENTER, space_before=4)
        _para(tf, sub, 13, color=GREY, align=PP_ALIGN.CENTER, space_before=8)
    chips(slide, 0.75, 5.25, ["retrain → re-run RQ4–RQ6", "Gemma 31B on the edits", "interface: built — live demo"],
          w=3.75, h=0.62, gap=0.27, size=13)
    notes(slide, """
[11:40-12:00]
• Each shortcut the closing runs exposed becomes a change to the data: live flags, varied and real-style code, and training pairs that differ only in wiring.
• Then retrain and re-run the same three probes: do the chain and old-graph signatures go away?
• Plus Gemma 31B on the edits. The interface is already built: paste, upload or generate a program and see the model's graph beside the truth, with the metrics — that is the live demo. Thank you.
""")
    return slide


def s_references(prs):
    slide = new_slide(prs, "References")
    refs = [
        "James K. Reed, Zachary DeVito, Horace He, Ansley Ussery, and Jason Ansel. torch.fx: Practical Program Capture and Transformation for Deep Learning in Python, MLSys, 2022. https://arxiv.org/abs/2112.08429",
        "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models, ICLR, 2022. https://arxiv.org/abs/2106.09685",
        "Jason Wei, Xuezhi Wang, Dale Schuurmans, et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, NeurIPS, 2022. https://arxiv.org/abs/2201.11903",
        "Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, et al. Show Your Work: Scratchpads for Intermediate Computation with Language Models, arXiv, 2021. https://arxiv.org/abs/2112.00114",
        "Alex Gu, Baptiste Rozière, Hugh Leather, Armando Solar-Lezama, Gabriel Synnaeve, and Sida I. Wang. CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution, ICML, 2024. https://arxiv.org/abs/2401.03065",
        "Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a Graph: Encoding Graphs for Large Language Models, ICLR, 2024. https://arxiv.org/abs/2310.04560",
        "Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov. Can Language Models Solve Graph Problems in Natural Language?, NeurIPS, 2023. https://arxiv.org/abs/2305.10037",
        "Binyuan Hui, Jian Yang, Zeyu Cui, et al. Qwen2.5-Coder Technical Report, arXiv, 2024. https://arxiv.org/abs/2409.12186",
        "Gemma Team, Aishwarya Kamath, Johan Ferret, et al. Gemma 3 Technical Report, arXiv, 2025. https://arxiv.org/abs/2503.19786",
        "Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. https://github.com/huggingface/peft, 2022. Accessed on: 26th October, 2026.",
        "TorchVision maintainers and contributors. https://github.com/pytorch/vision, 2016. Accessed on: 26th October, 2026.",
        "Ollama. https://ollama.com · OpenRouter. https://openrouter.ai · Kaggle. https://www.kaggle.com/code. Accessed on: 26th October, 2026.",
    ]
    tf = _textbox(slide, MARGIN, 1.55, SLIDE_W - 2 * MARGIN, 5.4)
    for i, r in enumerate(refs):
        _para(tf, f"[{i + 1}]  {r}", 10.5, color=INK, first=(i == 0), space_before=0 if i == 0 else 4)
    notes(slide, "Leave up during questions. Format: authors, title, venue, year, URL; websites with access date.")
    return slide


SLIDES = [s_title, s_outline, s_task, s_workflow, s_before, s_after, s_grading, s_prompting, s_lora,
          s_data, s_setup, s_res_scale, s_res_prompts, s_res_finetune, s_res_qualitative, s_res_closing,
          s_conclusions, s_novelty, s_references]


def spoken_minutes(prs) -> float:
    words = sum(len(s.notes_slide.notes_text_frame.text.split()) for s in prs.slides)
    return words / 140


def build() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
    for i, make in enumerate(SLIDES, 1):
        slide = make(prs)
        if i > 1:
            footer(slide, i)
    content = len(SLIDES) - 1                       # references excluded, as the instructions allow
    if content > 18:
        raise SystemExit(f"{content} content slides: over the course limit of 18")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"{content} slides + references -> {OUT.relative_to(ROOT)}; "
          f"notes read aloud ~{spoken_minutes(prs):.1f} min at 140 wpm")


if __name__ == "__main__":
    build()

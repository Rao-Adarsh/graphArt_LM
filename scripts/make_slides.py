"""Build the IE 643 prep presentation from the content held below.

Run from the repo root:

    .venv/Scripts/python scripts/make_slides.py

The deck is generated, not hand-edited, so it can be rebuilt after the content
changes -- in particular once the prompting baseline produces numbers: fill in
RESULTS near the bottom of this file and re-run.
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# --------------------------------------------------------------------------
# team / submission identity (instruction 3: topic, team name, roll numbers)
# --------------------------------------------------------------------------
TOPIC_TITLE = "Making Large Models Draw Topological Graphs"
SUBTITLE = "Prep Presentation — IE 643 Course Project, Autumn 2026"
TEAM_NAME = "Pensieve"
MEMBERS = [("Adarsh A Rao", "24B2132")]
ACCESS_DATE = "20th September, 2026"

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "presentation" / f"{TEAM_NAME}_IE643_CourseProject_Prep.pptx"

# --------------------------------------------------------------------------
# look and feel
# --------------------------------------------------------------------------
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.6
BODY_TOP = 1.5

NAVY = RGBColor(0x1F, 0x37, 0x64)
ACCENT = RGBColor(0x3B, 0x5B, 0x92)       # same blue as the node borders in render.py
LIGHT = RGBColor(0xEE, 0xF3, 0xFB)        # same node fill as render.py
INK = RGBColor(0x26, 0x2B, 0x33)
GREY = RGBColor(0x5A, 0x62, 0x6E)
RULE = RGBColor(0xC8, 0xD3, 0xE6)
WIRE0 = RGBColor(0x1F, 0x77, 0xB4)        # matplotlib tab10 blue   -> qubit 0
WIRE1 = RGBColor(0xE3, 0x76, 0x0E)        # matplotlib tab10 orange -> qubit 1
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

BODY_FONT = "Calibri"
MONO_FONT = "Consolas"


def _para(tf, text, size, *, bold=False, italic=False, color=INK, font=BODY_FONT,
          align=PP_ALIGN.LEFT, space_before=0, space_after=0, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font
    return p


def _textbox(slide, left, top, width, height, *, wrap=True, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def _indent(p, level):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.30 + 0.32 * level).emu))
    pPr.set("indent", str(-Inches(0.30).emu))


def _arrowhead(shape, color=ACCENT, width=1.5):
    """python-pptx has no arrowhead API; append the tailEnd element by hand."""
    shape.line.color.rgb = color
    shape.line.width = Pt(width)
    ln = shape.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)


# --------------------------------------------------------------------------
# slide scaffolding
# --------------------------------------------------------------------------
def new_slide(prs, title=None, *, kicker=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    if title:
        tf = _textbox(slide, MARGIN, 0.40, SLIDE_W - 2 * MARGIN, 0.85)
        if kicker:
            _para(tf, kicker.upper(), 11.5, bold=True, color=ACCENT, first=True, space_after=2)
            _para(tf, title, 26, bold=True, color=NAVY)
        else:
            _para(tf, title, 27, bold=True, color=NAVY, first=True)
        rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MARGIN), Inches(1.30),
                                      Inches(SLIDE_W - 2 * MARGIN), Pt(1.6))
        rule.fill.solid()
        rule.fill.fore_color.rgb = RULE
        rule.line.fill.background()
        rule.shadow.inherit = False
    return slide


def footer(slide, number):
    tf = _textbox(slide, MARGIN, SLIDE_H - 0.50, SLIDE_W - 2 * MARGIN, 0.3)
    _para(tf, "IE 643 Course Project  ·  Team " + TEAM_NAME, 9.5, color=GREY, first=True)
    tf2 = _textbox(slide, SLIDE_W - MARGIN - 1.0, SLIDE_H - 0.50, 1.0, 0.3)
    _para(tf2, str(number), 9.5, color=GREY, align=PP_ALIGN.RIGHT, first=True)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def bullets(slide, items, *, top=BODY_TOP, left=MARGIN, width=None, size=18, gap=9):
    """items: (level, text). '||' splits a bold lead-in from the rest of the line."""
    width = width if width is not None else SLIDE_W - 2 * MARGIN
    tf = _textbox(slide, left, top, width, SLIDE_H - top - 0.7)
    for i, (level, text) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(0 if i == 0 else gap)
        marker = "▸  " if level == 0 else "–  "
        run = p.add_run()
        run.text = marker
        run.font.size = Pt(size - 1)
        run.font.color.rgb = ACCENT if level == 0 else RULE
        run.font.name = BODY_FONT
        head, sep, tail = text.partition("||")
        for chunk, bold in ((head, bool(sep)), (tail, False)):
            if not chunk:
                continue
            run = p.add_run()
            run.text = chunk
            run.font.size = Pt(size if level == 0 else size - 2)
            run.font.bold = bold
            run.font.color.rgb = INK if level == 0 else GREY
            run.font.name = BODY_FONT
        _indent(p, level)
    return tf


def code_box(slide, left, top, width, height, code, *, size=13, caption=None):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
                                 Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = RULE
    box.line.width = Pt(1)
    box.shadow.inherit = False
    box.adjustments[0] = 0.04
    tf = box.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.16)
    tf.margin_top = Inches(0.12)
    for i, line in enumerate(code.split("\n")):
        _para(tf, line if line else " ", size, color=INK, font=MONO_FONT, first=(i == 0))
    if caption:
        cap = _textbox(slide, left, top - 0.30, width, 0.26)
        _para(cap, caption, 12, bold=True, color=ACCENT, first=True)
    return box


def table(slide, rows, left, top, width, col_widths, *, size=13, row_h=0.34):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top),
                                   Inches(width), Inches(row_h * len(rows)))
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = True
    for i, w in enumerate(col_widths):
        tbl.columns[i].width = Inches(w)
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(row_h)
        for c, text in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            _para(tf, str(text), size, bold=(r == 0), color=WHITE if r == 0 else INK, first=True)
    return tbl


def picture(slide, name, left, top, width):
    return slide.shapes.add_picture(str(FIGURES / name), Inches(left), Inches(top),
                                    width=Inches(width))


def picture_h(slide, name, box_left, top, box_width, height):
    """Place by height and centre in the box, so side-by-side figures line up."""
    pic = slide.shapes.add_picture(str(FIGURES / name), Inches(box_left), Inches(top),
                                   height=Inches(height))
    pic.left = Inches(box_left + max(0.0, (box_width - pic.width / 914400) / 2))
    return pic


# --------------------------------------------------------------------------
# hand-drawn diagrams
# running example:  qc.h(0); qc.x(1); qc.cx(0, 1); qc.rz(0.785, 1)
# --------------------------------------------------------------------------
def _gate(slide, cx, cy, label, *, w=0.62, h=0.46, size=14):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx - w / 2),
                                 Inches(cy - h / 2), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = ACCENT
    box.line.width = Pt(1.25)
    box.shadow.inherit = False
    box.adjustments[0] = 0.12
    tf = box.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_top = tf.margin_bottom = 0
    _para(tf, label, size, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True)
    return box


def _oval(slide, cx, cy, d, fill, line, label=None):
    ov = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2),
                                Inches(d), Inches(d))
    ov.fill.solid()
    ov.fill.fore_color.rgb = fill
    ov.line.color.rgb = line
    ov.line.width = Pt(1.5)
    ov.shadow.inherit = False
    tf = ov.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    _para(tf, label or "", 13, bold=True, color=line, align=PP_ALIGN.CENTER, first=True)
    return ov


def _line(slide, x1, y1, x2, y2, color, width=1.25):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                    Inches(x2), Inches(y2))
    ln.line.color.rgb = color
    ln.line.width = Pt(width)
    return ln


def _caption(slide, left, top, width, text, *, size=12, color=GREY, align=PP_ALIGN.LEFT,
             bold=False, font=BODY_FONT):
    tf = _textbox(slide, left, top, width, 0.3)
    _para(tf, text, size, bold=bold, color=color, align=align, first=True, font=font)


def draw_circuit(slide, x, y, *, annotate=True):
    """The running example drawn the way a physicist would draw it."""
    y0, y1 = y, y + 1.05
    x_end = x + 4.55
    for wy, lab, col in ((y0, "q0", WIRE0), (y1, "q1", WIRE1)):
        _line(slide, x, wy, x_end, wy, INK, 1.25)
        _caption(slide, x - 0.55, wy - 0.16, 0.45, lab, size=14, color=col, bold=True,
                 align=PP_ALIGN.RIGHT)
    c1, c2, c3 = x + 0.85, x + 2.15, x + 3.60
    _gate(slide, c1, y0, "H")
    _gate(slide, c1, y1, "X")
    _line(slide, c2, y0, c2, y1, INK, 1.4)          # cx: control -- target
    _oval(slide, c2, y0, 0.16, INK, INK)
    _oval(slide, c2, y1, 0.34, WHITE, INK, label="+")
    _gate(slide, c3, y1, "RZ", w=1.05)
    _caption(slide, c3 - 0.525, y1 + 0.23, 1.05, "0.785", size=10, color=GREY,
             align=PP_ALIGN.CENTER)
    if annotate:
        _caption(slide, x - 0.1, y1 + 0.58, 5.2, "time runs left to right  →", size=11)
    return x_end


def draw_dag(slide, x, y):
    """The same four operations drawn as the DAG we want the model to produce."""
    nodes = {}

    def node(nx_, ny_, title, wires, w=1.35):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(nx_), Inches(ny_),
                                     Inches(w), Inches(0.62))
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT
        box.line.color.rgb = ACCENT
        box.line.width = Pt(1.25)
        box.shadow.inherit = False
        box.adjustments[0] = 0.12
        tf = box.text_frame
        tf.word_wrap = False
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_top = tf.margin_bottom = 0
        _para(tf, title, 13, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True)
        _para(tf, wires, 10.5, color=GREY, align=PP_ALIGN.CENTER)
        return {"x": nx_, "y": ny_, "w": w, "h": 0.62}

    nodes[0] = node(x, y, "0: h", "q0")
    nodes[1] = node(x, y + 1.05, "1: x", "q1")
    nodes[2] = node(x + 1.95, y + 0.52, "2: cx", "q0 q1")
    nodes[3] = node(x + 3.90, y + 0.52, "3: rz(0.785)", "q1", w=1.55)

    def arrow(a, b, label, color):
        sa, sb = nodes[a], nodes[b]
        x1, y1 = sa["x"] + sa["w"], sa["y"] + sa["h"] / 2
        x2, y2 = sb["x"], sb["y"] + sb["h"] / 2
        ln = _line(slide, x1, y1, x2, y2, color, 1.75)
        _arrowhead(ln, color, 1.75)
        _caption(slide, (x1 + x2) / 2 - 0.30, (y1 + y2) / 2 - 0.32, 0.6, label, size=10,
                 color=color, align=PP_ALIGN.CENTER, bold=True)

    arrow(0, 2, "q0", WIRE0)
    arrow(1, 2, "q1", WIRE1)
    arrow(2, 3, "q1", WIRE1)
    return nodes


def draw_pipeline(slide, x, y):
    """generate -> extract -> target text, with the model scored against it."""
    def box(bx, by, head, sub, w=2.75, h=0.95, fill=LIGHT, edge=ACCENT):
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(bx), Inches(by),
                                     Inches(w), Inches(h))
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
        shp.line.color.rgb = edge
        shp.line.width = Pt(1.4)
        shp.shadow.inherit = False
        shp.adjustments[0] = 0.10
        tf = shp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        _para(tf, head, 13.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER, first=True,
              font=MONO_FONT)
        _para(tf, sub, 11, color=GREY, align=PP_ALIGN.CENTER)
        return {"x": bx, "y": by, "w": w, "h": h}

    gap, w, h = 0.75, 2.75, 0.95
    row2_y = y + 2.05
    a = box(x, y, "generate.py", "writes a random Qiskit program")
    b = box(x + (w + gap), y, "extract.py", "runs it → Qiskit gives the true DAG")
    c = box(x + 2 * (w + gap), y, "schema.py", "writes that DAG as text = the target")
    d = box(x + (w + gap), row2_y, "pipeline/", "runner: prompt in, reply out",
            fill=RGBColor(0xFD, 0xF3, 0xE3), edge=WIRE1)
    e = box(x + 2 * (w + gap), row2_y, "validate.py / metrics.py",
            "well-formed DAG?  how close to the truth?")

    def h_arrow(s, t, color=ACCENT):
        ln = _line(slide, s["x"] + s["w"], s["y"] + s["h"] / 2, t["x"], t["y"] + t["h"] / 2,
                   color, 1.75)
        _arrowhead(ln, color, 1.75)

    def v_arrow(s, t, color=ACCENT):
        ln = _line(slide, s["x"] + s["w"] / 2, s["y"] + s["h"], t["x"] + t["w"] / 2, t["y"],
                   color, 1.75)
        _arrowhead(ln, color, 1.75)

    h_arrow(a, b)
    h_arrow(b, c)
    h_arrow(d, e, WIRE1)
    v_arrow(c, e)
    ln = _line(slide, a["x"] + a["w"] / 2, a["y"] + a["h"], d["x"] + 0.5, d["y"], WIRE1, 1.75)
    _arrowhead(ln, WIRE1, 1.75)
    _caption(slide, x + 0.1, y + 1.28, 2.6, "the code, and nothing else", size=10.5,
             color=WIRE1)
    _caption(slide, x + 2 * (w + gap) + 0.1, y + 1.28, 2.6, "the answer to beat",
             size=10.5)


# --------------------------------------------------------------------------
# results slot: stays None until the prompting runner produces numbers.
# Fill with rows like ("Tier 1", "0.95", "0.72", "0.91", "0.88") and re-run.
# --------------------------------------------------------------------------
RESULTS = None

RUNNING_CODE = """from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
qc.cx(0, 1)
qc.rz(0.785, 1)"""

RUNNING_TARGET = """QUBITS 2
CLBITS 0
NODES
0 h q0
1 x q1
2 cx q0 q1
3 rz(0.785) q1
EDGES
0 -> 2 : q0
1 -> 2 : q1
2 -> 3 : q1"""


# --------------------------------------------------------------------------
# small visual building blocks
# --------------------------------------------------------------------------
def chips(slide, x, y, labels, *, w=2.9, h=0.52, gap=0.22, size=13, fill=LIGHT, edge=RULE,
          color=NAVY):
    """A row of short labels in rounded boxes -- a caption you can read at a glance."""
    for i, label in enumerate(labels):
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + i * (w + gap)),
                                     Inches(y), Inches(w), Inches(h))
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
        shp.line.color.rgb = edge
        shp.line.width = Pt(1)
        shp.shadow.inherit = False
        shp.adjustments[0] = 0.22
        tf = shp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.06)
        _para(tf, label, size, bold=True, color=color, align=PP_ALIGN.CENTER, first=True)


def draw_strip(slide, x, y, items, *, w=3.8, h=0.95, gap=0.55):
    """kicker/body boxes chained by arrows -- used for the framing strip."""
    for i, (head, body) in enumerate(items):
        bx = x + i * (w + gap)
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(bx), Inches(y),
                                     Inches(w), Inches(h))
        shp.fill.solid()
        shp.fill.fore_color.rgb = LIGHT if i < len(items) - 1 else RGBColor(0xFD, 0xF3, 0xE3)
        shp.line.color.rgb = RULE if i < len(items) - 1 else WIRE1
        shp.line.width = Pt(1.2)
        shp.shadow.inherit = False
        shp.adjustments[0] = 0.10
        tf = shp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.10)
        _para(tf, head.upper(), 9.5, bold=True, color=ACCENT, align=PP_ALIGN.CENTER, first=True)
        _para(tf, body, 12.5, color=INK, align=PP_ALIGN.CENTER)
        if i:
            ln = _line(slide, bx - gap + 0.08, y + h / 2, bx - 0.06, y + h / 2, ACCENT, 1.75)
            _arrowhead(ln, ACCENT, 1.75)


def token_chips(slide, x, y, tokens, *, size=12.5):
    """Show how the tokenizer chops our output format up."""
    cx = x
    for tok in tokens:
        w = max(0.34, 0.115 * len(tok) + 0.26)
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx), Inches(y),
                                     Inches(w), Inches(0.38))
        shp.fill.solid()
        shp.fill.fore_color.rgb = RGBColor(0xFD, 0xF3, 0xE3)
        shp.line.color.rgb = WIRE1
        shp.line.width = Pt(1)
        shp.shadow.inherit = False
        shp.adjustments[0] = 0.25
        tf = shp.text_frame
        tf.word_wrap = False
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = 0
        _para(tf, tok, size, color=INK, font=MONO_FONT, align=PP_ALIGN.CENTER, first=True)
        cx += w + 0.09
    return cx


def phase_card(slide, x, y, w, h, title, rows, failure, *, accent=ACCENT):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                  Inches(w), Inches(h))
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = accent
    card.line.width = Pt(1.75)
    card.shadow.inherit = False
    card.adjustments[0] = 0.06
    card.text_frame.text = ""

    head = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                  Inches(w), Inches(0.72))
    head.fill.solid()
    head.fill.fore_color.rgb = accent
    head.line.fill.background()
    head.shadow.inherit = False
    head.adjustments[0] = 0.18
    tf = head.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _para(tf, title, 15, bold=True, color=WHITE, align=PP_ALIGN.CENTER, first=True)

    tf = _textbox(slide, x + 0.25, y + 0.95, w - 0.5, h - 1.9)
    for i, (k, v) in enumerate(rows):
        p = _para(tf, k, 11, bold=True, color=ACCENT, first=(i == 0), space_before=0 if i == 0 else 9)
        _para(tf, v, 13.5, color=INK, space_before=1)

    fail = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 0.14),
                                  Inches(y + h - 0.82), Inches(w - 0.28), Inches(0.68))
    fail.fill.solid()
    fail.fill.fore_color.rgb = RGBColor(0xFD, 0xF3, 0xE3)
    fail.line.color.rgb = WIRE1
    fail.line.width = Pt(1)
    fail.shadow.inherit = False
    fail.adjustments[0] = 0.18
    tf = fail.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08)
    _para(tf, "LIKELY FAILURE", 8.5, bold=True, color=WIRE1, align=PP_ALIGN.CENTER, first=True)
    _para(tf, failure, 11.5, color=INK, align=PP_ALIGN.CENTER)


# --------------------------------------------------------------------------
# the slides
# --------------------------------------------------------------------------
def slide_title(prs):
    slide = new_slide(prs)
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                    Inches(0.28), Inches(SLIDE_H))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = NAVY
    stripe.line.fill.background()
    stripe.shadow.inherit = False

    tf = _textbox(slide, 1.1, 1.75, 11.0, 1.0)
    _para(tf, SUBTITLE.upper(), 14, bold=True, color=ACCENT, first=True)

    tf = _textbox(slide, 1.1, 2.35, 11.2, 1.6)
    _para(tf, TOPIC_TITLE, 40, bold=True, color=NAVY, first=True)

    tf = _textbox(slide, 1.1, 3.75, 10.6, 0.8)
    _para(tf, "Teaching a language model to read quantum-circuit code and draw the "
              "circuit's DAG — without running the code.", 17, italic=True, color=GREY,
          first=True)

    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.1), Inches(4.75),
                                  Inches(3.2), Pt(2.5))
    rule.fill.solid()
    rule.fill.fore_color.rgb = ACCENT
    rule.line.fill.background()
    rule.shadow.inherit = False

    tf = _textbox(slide, 1.1, 5.05, 11.0, 1.4)
    _para(tf, "Team " + TEAM_NAME, 21, bold=True, color=NAVY, first=True)
    for name, roll in MEMBERS:
        _para(tf, f"{name}   ·   Roll No. {roll}", 16, color=INK, space_before=4)
    _para(tf, "Indian Institute of Technology Bombay  ·  September 2026", 13,
          color=GREY, space_before=10)

    notes(slide, NOTES["title"])
    return slide


def slide_outline(prs):
    slide = new_slide(prs, "Outline")
    bullets(slide, [
        (0, "Background: what a quantum circuit is, and why a circuit is a DAG"),
        (0, "The task stated precisely, and why it is hard for a language model"),
        (0, "Background material read, and the three ways to adapt a model"),
        (0, "What we have built: generator, ground truth, validator, metrics"),
        (0, "The dataset: four difficulty tiers, with an example of each"),
        (0, "Where we stand today, the roadmap of subtasks, and the research questions"),
    ], top=1.75, size=19, gap=16)
    draw_strip(slide, 0.95, 5.35, [
        ("allotted task", "large models draw\ntopological graphs"),
        ("reference instance", "torch.fx traces a model\ninto a computation DAG"),
        ("what we measure on", "Qiskit circuit DAGs — exact\nlabels, checkable by machine"),
    ], w=3.5, h=1.05, gap=0.65)
    notes(slide, NOTES["outline"])
    return slide


def slide_circuit_basics(prs):
    slide = new_slide(prs, "A quantum circuit, in one picture",
                      kicker="background · no physics required")
    code_box(slide, 0.75, 2.35, 3.75, 1.58, RUNNING_CODE.split("\n\n")[1], size=14,
             caption="five lines of Qiskit…")
    arrow = _line(slide, 4.75, 3.3, 5.45, 3.3, ACCENT, 2.0)
    _arrowhead(arrow, ACCENT, 2.0)
    _caption(slide, 6.3, 2.05, 5.4, "…build this circuit", size=12, bold=True, color=ACCENT)
    draw_circuit(slide, 6.85, 2.85, annotate=False)
    chips(slide, 0.75, 4.95, [
        "a qubit is a wire",
        "a gate is a box on a wire",
        "cx spans two wires:  • control,  ⊕ target",
        "straight-line: no branching at run time",
    ], w=2.93, h=0.7, gap=0.2, size=12.5)
    notes(slide, NOTES["circuit"])
    return slide


def slide_why_dag(prs):
    slide = new_slide(prs, "The same circuit, as a graph", kicker="background · the core idea")
    _caption(slide, 0.75, 1.75, 4.6, "the circuit", size=12.5, bold=True, color=ACCENT)
    draw_circuit(slide, 1.35, 2.2, annotate=False)
    arrow = _line(slide, 6.15, 2.7, 6.85, 2.7, ACCENT, 2.0)
    _arrowhead(arrow, ACCENT, 2.0)
    _caption(slide, 7.15, 1.75, 5.2, "its DAG", size=12.5, bold=True, color=ACCENT)
    draw_dag(slide, 7.15, 2.1)
    chips(slide, 0.75, 4.55, [
        "one node per operation",
        "edge  A → B  =  next operation on that wire",
        "no cycles: time only moves forward",
    ], w=3.93, h=0.62, gap=0.2, size=13)
    banner = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.75), Inches(5.5),
                                    Inches(11.83), Inches(0.82))
    banner.fill.solid()
    banner.fill.fore_color.rgb = RGBColor(0xFD, 0xF3, 0xE3)
    banner.line.color.rgb = WIRE1
    banner.line.width = Pt(1.4)
    banner.shadow.inherit = False
    banner.adjustments[0] = 0.15
    tf = banner.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _para(tf, "qc.h(0) and qc.x(1) can be written in either order and give the same DAG "
              "— so we compare graphs, not gate lists.", 15.5, bold=True, color=INK,
          align=PP_ALIGN.CENTER, first=True)
    notes(slide, NOTES["dag"])
    return slide


def slide_qiskit(prs):
    slide = new_slide(prs, "Where the correct answer comes from",
                      kicker="background · ground truth")
    code_box(slide, 0.75, 2.15, 5.85, 3.05,
             "from qiskit import QuantumCircuit\n"
             "from qiskit.converters import circuit_to_dag\n"
             "\n"
             "qc = QuantumCircuit(2)     # two wires\n"
             "qc.h(0)                    # append one\n"
             "qc.x(1)                    # operation\n"
             "qc.cx(0, 1)                # per call\n"
             "qc.rz(0.785, 1)\n"
             "\n"
             "dag = circuit_to_dag(qc)   # DAGCircuit",
             size=13.5, caption="the entire labelling pipeline")
    bullets(slide, [
        (0, "QuantumCircuit(n)||  →  n wires"),
        (0, "each qc.<gate>(…)||  →  appends exactly one operation"),
        (0, "circuit_to_dag||  →  DAGCircuit, the IR Qiskit’s own compiler uses for "
            "optimisation and qubit routing"),
        (0, "So: exact labels, zero hand-annotation, unlimited quantity."),
        (0, "The model is given the source text only."),
    ], top=2.3, size=17, gap=20, left=7.1, width=5.6)
    notes(slide, NOTES["qiskit"])
    return slide


def slide_task(prs):
    slide = new_slide(prs, "The task, stated precisely", kicker="the task")
    code_box(slide, 0.75, 2.25, 5.0, 2.05, RUNNING_CODE, size=14,
             caption="INPUT — what the model is given")
    arrow = _line(slide, 6.1, 3.25, 6.8, 3.25, ACCENT, 2.0)
    _arrowhead(arrow, ACCENT, 2.0)
    code_box(slide, 7.2, 2.25, 5.1, 3.05, RUNNING_TARGET, size=13.5,
             caption="OUTPUT — what it must produce, exactly")
    chips(slide, 0.75, 5.75, [
        "node:   <id> <gate>[(<params>)] <wires…>",
        "edge:   <src> -> <dst> : <wires…>",
        "ids are the model’s choice — we compare up to isomorphism",
    ], w=3.93, h=0.62, gap=0.2, size=12.5)
    notes(slide, NOTES["task"])
    return slide


def slide_hard(prs):
    slide = new_slide(prs, "Why this is hard for a language model", kicker="the task")
    bullets(slide, [
        (0, "Unroll loops, expand calls, discard dead branches, broadcast qc.h(range(n))."),
        (0, "Track “last operation on each wire” — with no scratch memory except "
            "the tokens already written."),
        (0, "One missed gate breaks a chain, and the output stops being a valid DAG."),
        (0, "Our output is almost entirely numbers, and numbers tokenise badly:"),
    ], top=1.8, size=18, gap=17)
    token_chips(slide, 1.1, 4.35,
                ["rz", "(", "0", ".", "785", ")", "␣q", "12"])
    _caption(slide, 1.1, 3.95, 8.0, "rz(0.785) q12   as the model actually reads it",
             size=12.5, bold=True, color=WIRE1, font=MONO_FONT)
    banner = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.75), Inches(5.35),
                                    Inches(11.83), Inches(0.8))
    banner.fill.solid()
    banner.fill.fore_color.rgb = LIGHT
    banner.line.color.rgb = ACCENT
    banner.line.width = Pt(1.4)
    banner.shadow.inherit = False
    banner.adjustments[0] = 0.15
    tf = banner.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _para(tf, "CRUXEval (2024): models that write code well are substantially worse at "
              "predicting what that code does.", 15.5, bold=True, color=NAVY,
          align=PP_ALIGN.CENTER, first=True)
    notes(slide, NOTES["hard"])
    return slide


def slide_adapt(prs):
    slide = new_slide(prs, "Three ways to use a model — our three phases",
                      kicker="background material read")
    w, gap = 3.85, 0.29
    phase_card(slide, 0.75, 1.9, w, 4.05, "Phase 1  ·  Prompting", [
        ("DATA", "0–5 examples in the prompt"),
        ("COMPUTE", "inference only"),
        ("PYTHON", "known, from pretraining"),
    ], "format errors; loses track of wires")
    phase_card(slide, 0.75 + w + gap, 1.9, w, 4.05, "Phase 2  ·  LoRA fine-tuning", [
        ("DATA", "~1k–10k (code, DAG) pairs"),
        ("COMPUTE", "one free GPU, hours"),
        ("PYTHON", "known, from pretraining"),
    ], "overfits our generator’s style")
    phase_card(slide, 0.75 + 2 * (w + gap), 1.9, w, 4.05, "Phase 3  ·  From scratch", [
        ("DATA", "10k–1M — we generate them"),
        ("COMPUTE", "one GPU, hours to days"),
        ("PYTHON", "none — learnt from our data"),
    ], "poor generalisation to real code")
    notes(slide, NOTES["adapt"])
    return slide


def slide_background(prs):
    slide = new_slide(prs, "Background material read", kicker="background material read")
    _caption(slide, 0.75, 1.7, 5.8, "FOUNDATIONS — how language models work", size=12,
             bold=True, color=ACCENT)
    bullets(slide, [
        (0, "3Blue1Brown, Neural Networks||  — what attention computes, visually"),
        (0, "Karpathy, Intro to LLMs||  — pretraining → SFT → preference tuning"),
        (0, "Alammar, The Illustrated Transformer||  — the architecture, block by block"),
        (1, "Taken from these: generation is next-token prediction repeated, so an early "
            "mistake poisons the rest; attention is the mechanism that finds the last node "
            "on a wire; we decode at temperature 0."),
    ], top=2.1, size=15, gap=11, left=0.75, width=5.8)
    _caption(slide, 7.0, 1.7, 5.6, "DIRECTLY RELATED WORK", size=12, bold=True, color=ACCENT)
    bullets(slide, [
        (0, "Talk like a Graph (2023)||  — how a graph is written as text changes LLM "
            "accuracy: our output format is a variable to test"),
        (0, "CRUXEval (2024)||  — predicting what code does, without running it: the "
            "input half of our task"),
        (0, "Qiskit HumanEval (2024)||  — 151 human-written Qiskit tasks: our source of "
            "real code for a held-out test set"),
        (0, "Qiskit DAGCircuit docs||  — defines our target structure exactly"),
        (1, "No public dataset of (code → circuit DAG) pairs exists. Building one is part "
            "of the contribution."),
    ], top=2.1, size=15, gap=11, left=7.0, width=5.6)
    notes(slide, NOTES["background"])
    return slide


def slide_pipeline(prs):
    slide = new_slide(prs, "What has been built so far", kicker="work done")
    draw_pipeline(slide, 1.6, 1.68)
    chips(slide, 1.05, 5.25, [
        "seven modules under graphart/",
        "runner + scorer written and dry-run",
        "not yet done: call a real model",
    ], w=3.75, h=0.62, gap=0.28, size=13)
    _caption(slide, 0.75, 6.15, 11.83,
             "The model is handed the code and nothing else — never the circuit, never "
             "the answer.", size=14, bold=True, color=WIRE1, align=PP_ALIGN.CENTER)
    notes(slide, NOTES["pipeline"])
    return slide


def slide_tiers(prs):
    slide = new_slide(prs, "The dataset: four tiers of difficulty", kicker="the dataset")
    table(slide, [
        ["Tier", "What the code looks like", "Unique samples", "Nodes  min / avg / max"],
        ["1  flat", "one gate per line — reading it is copying", "500", "3 / 9.2 / 17"],
        ["2  loops", "for loops over qubits, repeated layers, % n rings, reversed ranges",
         "500", "3 / 13.5 / 42"],
        ["3  control", "helper functions, if/else on the loop index, dead branches, nested "
         "loops, while, lists of angles", "500", "3 / 8.6 / 23"],
        ["4  algorithms", "GHZ, QFT, Bernstein–Vazirani, Grover (2q), Deutsch–Jozsa, "
         "hardware-efficient ansatz", "93", "6 / 17.4 / 26"],
    ], 0.6, 1.6, 12.13, [1.75, 6.2, 1.85, 2.33], size=12.5, row_h=0.5)
    _caption(slide, 0.6, 4.2, 12.13, "The tiers vary how well the code hides the circuit — "
             "not how big the circuit is.", size=15, bold=True, color=NAVY,
             align=PP_ALIGN.CENTER)
    chips(slide, 0.75, 4.85, [
        "the next four slides:  one worked example per tier",
        "tier 4 is 93 samples — test-only until more templates are added",
    ], w=5.9, h=0.68, gap=0.28, size=13.5)
    notes(slide, NOTES["tiers"])
    return slide


def _tier_slide(prs, tier, title, caption):
    slide = new_slide(prs, title, kicker=f"the dataset · tier {tier} example")
    picture_h(slide, f"tier{tier}.png", 0.6, 1.75, 12.13, 3.15)
    _caption(slide, 0.75, 5.35, 11.83, caption, size=15, bold=True, color=NAVY,
             align=PP_ALIGN.CENTER)
    notes(slide, NOTES[f"tier{tier}"])
    return slide


def slide_tier1(prs):
    return _tier_slide(prs, 1, "Tier 1 — flat",
                       "Every operation is written out. Reading the code is transcription; "
                       "only the edges have to be worked out.")


def slide_tier2(prs):
    return _tier_slide(prs, 2, "Tier 2 — loops",
                       "Two loops over five qubits. Ten operations, five independent chains "
                       "— and not one of them appears in the code.")


def slide_tier3(prs):
    return _tier_slide(prs, 3, "Tier 3 — control flow",
                       "if/else on the loop index picks RX or RZ; a nested loop produces six "
                       "cx gates. Ten operations, none written down.")


def slide_tier4(prs):
    return _tier_slide(prs, 4, "Tier 4 — textbook algorithms",
                       "A GHZ state with measurement. Classical bits become wires too, so "
                       "measure nodes carry both a qubit and a clbit.")


def slide_curation(prs):
    slide = new_slide(prs, "Data curation: how the splits are kept honest",
                      kicker="the dataset")
    bullets(slide, [
        (0, "One command, fixed seed.||  Nothing is hand-labelled or hand-corrected."),
        (0, "Every ground-truth graph passes our own validator before it is kept.||  "
            "A malformed extraction fails the build instead of poisoning the data."),
        (0, "Deduplicated by the graph itself, across the whole dataset.||  A sample whose "
            "DAG has already been seen is thrown away — so no graph can appear in two "
            "splits, and memorising cannot score."),
        (0, "Filtered to 3–60 nodes,||  which keeps every sample inside a small "
            "model’s context window."),
        (0, "Then split 80 / 10 / 10 within each tier:||  1274 train · 159 val · 160 "
            "test.  The generator scales to 4,000 per tier for tiers 1–3 "
            "(≈ 9,700 training pairs) when fine-tuning needs more."),
    ], top=1.75, size=16, gap=13)
    notes(slide, NOTES["curation"])
    return slide


def slide_external(prs):
    slide = new_slide(prs, "Real code for the generalisation tests", kicker="the dataset")
    table(slide, [
        ["Source", "What it is", "How we would use it"],
        ["Qiskit HumanEval", "151 hand-written Qiskit tasks with reference solutions",
         "run each solution that builds a circuit, extract its DAG → a real-code test set"],
        ["MQT Bench", "thousands of benchmark circuits, 2–130 qubits, generated in Python",
         "more tier-4 algorithms, and larger circuits for the size-generalisation test"],
        ["QASMBench", "~130 OpenQASM circuits from chemistry, arithmetic and ML",
         "load the QASM into a QuantumCircuit, then the same extractor"],
        ["QCircuitBench", "a large dataset for LLM-based quantum algorithm design",
         "algorithm code written in a style quite different from our generator"],
    ], 0.6, 1.6, 12.13, [2.5, 4.6, 5.03], size=13, row_h=0.62)
    _caption(slide, 0.6, 4.95, 12.13,
             "All four end up as a QuantumCircuit, so the same circuit_to_dag gives the "
             "ground truth and the same metrics score it.", size=15, bold=True, color=NAVY,
             align=PP_ALIGN.CENTER)
    chips(slide, 1.35, 5.6, [
        "our generator has a house style — a fixed vocabulary of gates and code shapes",
        "a model could learn that style instead of the rule; only real code exposes it",
    ], w=5.15, h=0.72, gap=0.3, size=13)
    notes(slide, NOTES["external"])
    return slide


def slide_metrics(prs):
    slide = new_slide(prs, "How an answer is scored", kicker="evaluation")
    table(slide, [
        ["Metric", "What it means"],
        ["valid", "the output parses; there are no cycles; every edge’s wires belong to "
                  "both of its endpoints; and each wire forms one single chain"],
        ["exact", "the same graph as the ground truth, by labelled graph isomorphism — "
                  "this is the headline number"],
        ["node F1", "overlap of the (gate, qubits, clbits, params) multisets — partial "
                    "credit for getting most operations right"],
        ["edge F1", "overlap of the (source label, target label, wires) multisets — "
                    "partial credit for the structure"],
    ], 0.6, 1.55, 12.13, [1.9, 10.23], size=13.5, row_h=0.56)
    code_box(slide, 0.6, 4.75, 12.13, 0.95,
             "perfect copy      valid=True   exact=True    node_f1=1.00  edge_f1=1.00\n"
             "one wrong gate    valid=True   exact=False   node_f1=0.93  edge_f1=0.92\n"
             "one missing edge  valid=False  exact=False   node_f1=1.00  edge_f1=0.96"
             "   ['wire q2: 5 nodes should be chained by 4 edges, found 3']",
             size=11.5, caption="sanity check — scoring a correct graph and two damaged copies")
    bullets(slide, [
        (0, "Reported per tier and per circuit size. Node ids are ignored throughout."),
    ], top=6.0, size=15)
    notes(slide, NOTES["metrics"])
    return slide


def slide_status(prs):
    slide = new_slide(prs, "Where we stand today", kicker="status and plan")
    if RESULTS is None:
        bullets(slide, [
            (0, "The harness runs end to end, and is checked.||  The sanity check on the "
                "previous slide is the evidence."),
            (0, "The dataset is built and inspected.||  1,593 samples across four tiers, with "
                "a rendered example per tier."),
            (0, "The Phase-1 harness is written, and dry-run end to end.||  Four prompt "
                "variants (zero-shot, few-shot, chain-of-thought, and the gate-list ablation), "
                "a runner with a hosted and a local backend, and a scorer that reports by tier "
                "and by circuit size."),
            (0, "No model has been called yet.||  We would rather report no numbers than "
                "numbers we cannot stand behind. The next command run is the zero-shot "
                "baseline over the 160-sample test split."),
            (0, "First measurements:||  zero-shot and few-shot valid / exact / node F1 / edge "
                "F1 on the 160-sample test split, by tier and by circuit size, for one hosted "
                "model and one small local model."),
            (0, "Then the ablation that answers the central question:||  the model outputs "
                "only the unrolled gate list, and graphart/connect.py builds the edges. That "
                "edge-builder already reproduces the ground truth on all 160 test samples, so "
                "whatever it then scores is purely the model\u2019s reading of the code."),
        ], top=1.7, size=17, gap=16)
    else:
        table(slide, [["Split", "valid", "exact", "node F1", "edge F1"]] + list(RESULTS),
              0.6, 1.7, 12.13, [3.13, 2.25, 2.25, 2.25, 2.25], size=14, row_h=0.5)
        bullets(slide, [
            (0, "Zero-shot, temperature 0, on the 160-sample test split."),
            (0, "Next: few-shot, chain-of-thought, and the “LLM unrolls, code "
                "connects” ablation."),
        ], top=1.75 + 0.5 * (len(RESULTS) + 1) + 0.4, size=16, gap=11)
    notes(slide, NOTES["status"])
    return slide


def slide_roadmap(prs):
    slide = new_slide(prs, "Roadmap: the project as subtasks", kicker="status and plan")
    bullets(slide, [
        (0, "Phase 1 — prompting baselines  (the next two weeks).||  Zero-shot, few-shot, "
            "chain-of-thought, and the “LLM unrolls, code connects” ablation, on one "
            "hosted model and one local model."),
        (0, "Phase 2 — LoRA fine-tuning.||  A 0.5B–1.5B open code model on a free "
            "Colab or Kaggle GPU. Does a fine-tuned 1B model beat a prompted frontier model? "
            "How much data is enough?"),
        (0, "Phase 3 (stretch) — a tiny transformer from scratch.||  nanoGPT-scale, "
            "10–50M parameters, possibly with a custom tokenizer in which q12 is one token."),
        (0, "Phase 4 — generalisation.||  Train on ≤ 5 qubits, test on 6–10; hold "
            "out a construct such as while loops; test on real Qiskit HumanEval code; and "
            "revisit torch.fx graphs as a second domain."),
        (0, "Phase 5 — interface.||  Paste code, see the predicted DAG beside the ground "
            "truth with the differences highlighted."),
        (0, "Immediate subtasks:||  (1) run the zero-shot baseline, (2) freeze the test "
            "split, (3) the Qiskit HumanEval importer, (4) more tier-4 templates."),
    ], top=1.7, size=16, gap=14)
    notes(slide, NOTES["roadmap"])
    return slide


def slide_questions(prs):
    slide = new_slide(prs, "The research questions", kicker="status and plan")
    bullets(slide, [
        (0, "1.  How does accuracy fall as the code hides more of the structure?||  "
            "Tier 1 → 4 at constant circuit size, and separately as the circuit grows."),
        (0, "2.  Is the hard part executing the code, or building the graph?||  The Phase-1 "
            "ablation answers this, and it decides where the rest of the effort goes."),
        (0, "3.  Prompting, fine-tuning or from scratch — best accuracy for the compute "
            "spent?||  All three are scored by the same harness, so the comparison is fair."),
        (0, "4.  Do the models learn the rule, or memorise the pattern?||  Tested on larger "
            "circuits, held-out language constructs, and real human-written code."),
    ], top=1.95, size=18, gap=20)
    notes(slide, NOTES["questions"])
    return slide


def slide_references(prs):
    slide = new_slide(prs, "References")
    tf = _textbox(slide, MARGIN, 1.5, SLIDE_W - 2 * MARGIN, 5.3)
    _para(tf, "Papers", 13.5, bold=True, color=ACCENT, first=True)
    papers = [
        "Bahare Fatemi, Jonathan Halcrow and Bryan Perozzi. Talk like a Graph: Encoding Graphs "
        "for Large Language Models, arXiv preprint, 2023. https://arxiv.org/abs/2310.04560",
        "Alex Gu, Baptiste Rozière, Hugh Leather, Armando Solar-Lezama, Gabriel Synnaeve "
        "and Sida I. Wang. CRUXEval: A Benchmark for Code Reasoning, Understanding and "
        "Execution, arXiv preprint, 2024. https://arxiv.org/abs/2401.03065",
        "Sanjay Vishwakarma, Francis Harkins, Siddharth Golecha, Vishal Sharathchandra Bajpe, "
        "Nicolas Dupuis, Luca Buratti, David Kremer, Ismael Faro, Ruchir Puri and Juan "
        "Cruz-Benito. Qiskit HumanEval: An Evaluation Benchmark For Quantum Code Generative "
        "Models, arXiv preprint, 2024. https://arxiv.org/abs/2406.14712",
        "Rui Yang, Ziruo Wang, Yuntian Gu, Tianyi Chen, Yitao Liang and Tongyang Li. "
        "QCircuitBench: A Large-Scale Dataset for Benchmarking Quantum Algorithm Design, "
        "arXiv preprint, 2024. https://arxiv.org/abs/2410.07961",
        "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, "
        "Lu Wang and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models, "
        "arXiv preprint, 2021. https://arxiv.org/abs/2106.09685",
    ]
    for i, ref in enumerate(papers, 1):
        _para(tf, f"[{i}]  {ref}", 11.5, color=INK, space_before=5)
    _para(tf, "Websites", 13.5, bold=True, color=ACCENT, space_before=13)
    sites = [
        ("Grant Sanderson (3Blue1Brown)",
         "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi"),
        ("Andrej Karpathy", "https://www.youtube.com/watch?v=zjkBMFhNj_g"),
        ("Jay Alammar", "https://jalammar.github.io/illustrated-transformer/"),
        ("IBM Qiskit",
         "https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.dagcircuit.DAGCircuit"),
        ("Munich Quantum Toolkit", "https://github.com/munich-quantum-toolkit/bench"),
        ("Pacific Northwest National Laboratory", "https://github.com/pnnl/QASMBench"),
    ]
    for i, (who, url) in enumerate(sites, len(papers) + 1):
        _para(tf, f"[{i}]  {who}. {url}. Accessed on: {ACCESS_DATE}.", 11.5, color=INK,
              space_before=5)
    notes(slide, NOTES["refs"])
    return slide


# --------------------------------------------------------------------------
# Speaker notes. Timings assume ~140 words per minute with pauses and add up to
# about 12 minutes, leaving margin under the 15-minute cap in the instructions.
# These are talking points, not a script -- reading the slides aloud is penalised.
# --------------------------------------------------------------------------
NOTES = {
"title": """
[0:00-0:25] Hello - I'm Adarsh A Rao, roll number 24B2132, Team Pensieve. My allotted topic
is "Making large models draw topological graphs": can a model read code it is not allowed
to run, and output the graph that code builds? The version an ML audience knows is
torch.fx, which traces a network into a computation DAG - but fx node names depend on the
object you traced, so correct answers cannot be compared automatically. Quantum circuit
DAGs give the same structure with an exact, machine-checkable answer, so that is what I
measure on.
""",

"outline": """
[0:25-0:45] The strip at the bottom is the whole framing: allotted task, the reference
instance, and what I actually measure on. I'll give you just enough quantum background to
read the pictures, then the task, the background reading, what is built, the dataset, and
where I stand.
""",

"circuit": """
[0:45-1:20] All the quantum you need is on this slide. Five lines of Qiskit on the left
build the circuit on the right. A qubit is a wire and time runs left to right. A gate is a
box sitting on a wire - H, X and RZ each touch one. CX touches two: a control dot and a
target plus. And the structural fact everything depends on: a circuit is a straight-line
program. No branching at run time, just a fixed sequence of operations over fixed wires.
""",

"dag": """
[1:20-2:05] Same circuit, different question: what depends on what? Every operation becomes
a node. I draw an arrow from A to B when B is the next thing that happens on some wire
after A, and I label the arrow with that wire. H is first on qubit zero, then the CX -
blue. X is first on qubit one, then the CX - orange. Then CX to RZ. Every wire is a chain,
and time only moves forwards, so there can never be a cycle. It is a DAG. And look at the
banner: H on qubit zero and X on qubit one are independent, so swapping those two lines in
the code gives an identical DAG. The correct answer is a graph, not a sequence - which is
why I compare graphs.
""",

"qiskit": """
[2:05-2:40] Where does the correct answer come from? QuantumCircuit of n gives me n wires,
each method call appends one operation, and then the last line - circuit_to_dag - hands me
the answer. I did not write that function. DAGCircuit is the representation Qiskit's own
compiler uses to cancel gates and route qubits onto hardware. So my entire labelling
pipeline is: generate a program, run it, call the converter. Exact labels, no annotation, as
many as I want. The model gets the source text and nothing else.
""",

"task": """
[2:40-3:10] The task in one picture. Left, the input: plain Python. Right, the output: the
DAG as text in a fixed format - four node lines, then three edge lines, each naming its
wire. The chips at the bottom give the grammar. The one that matters is the last: the ids
are the model's own choice, so numbering the nodes differently but consistently is still
correct, and my metrics compare graphs up to isomorphism.
""",

"hard": """
[3:10-3:50] Four difficulties. It has to unroll loops, expand calls and discard dead
branches. It has to track the last operation on every wire, with no scratch memory except
the tokens it has already written - which is a strong hint that chain-of-thought will
matter. It is brittle: one missed gate breaks a chain and the graph stops being valid. And
the tokenisation problem, which is easy to underestimate - "rz open-bracket zero point
seven-eight-five" comes apart into eight tokens, and q12 is never one symbol. My output is
almost entirely numbers. CRUXEval measured exactly this gap between writing code and
predicting what it does.
""",

"adapt": """
[3:50-4:30] The TA confirmed any approach is fine as long as the results are good, so I
planned all three, cheapest first, and they became my phases. Prompting touches only the
input. LoRA freezes the pretrained weights and trains small adapters alongside - under one
percent of the parameters, which is what puts a one-billion model on a free Colab GPU.
From scratch starts from random weights, normally hopeless, but my language is tiny and
synthetic. The orange strip on each card is what I expect to go wrong, and those three
failures are different enough that the comparison is informative either way.
""",

"background": """
[4:30-5:15] On the left, the foundations I worked through, and what I took from them:
generation is next-token prediction repeated, so an early mistake poisons everything after
it; attention is the mechanism that lets the model look back and find the last node on a
wire - which is literally my task; and on the local model I decode greedily; the current hosted models no longer expose
temperature at all, so what I record with every result is the model version and the effort
level. That matters because the answer is
structured. On the right, the directly related work. Talk like a Graph shows the text
encoding of a graph changes accuracy, so my output format is a variable to test rather than
a matter of taste. CRUXEval is the input half of my problem. Qiskit HumanEval is the
reverse direction and my source of real code. And the gap at the bottom: no public dataset
of code-to-DAG pairs exists.
""",

"pipeline": """
[5:15-6:00] This is the machinery, and it is worth thirty seconds because everything else
rests on it. Top row builds the data. generate.py writes a random Qiskit program.
extract.py actually runs that program and asks Qiskit for the true DAG. schema.py turns
that DAG into the text format you saw earlier - that text is the target. Bottom row is the
experiment. The orange arrow is important: the model receives the code only - never the
circuit, never the answer. Its reply goes back through schema.py to be parsed, validate.py
asks whether it is even a well-formed DAG, and metrics.py asks how close it is to the
truth. Seven modules under graphart/, all working, and the runner and scorer in pipeline/ are
written and dry-run end to end. The one thing left is pointing it at a real model.
""",

"tiers": """
[6:00-6:35] The dataset is four tiers, and the point is on the line under the table: the
tiers change how well the code hides the circuit, not how big the circuit is. Tier one is
flat, tier two adds loops, tier three adds functions and control flow, tier four is
textbook algorithms. That gives me my first research question almost for free - plot
accuracy against tier and you are measuring how much difficulty comes from the code rather
than from the graph. The next four slides show one real example from each.
""",

"tier1": """
[6:35-6:55] Tier one. Every operation is written out on its own line, so reading the code
is transcription. The only real work is the edges - noticing that the swap at the end
follows the z on qubit zero.
""",

"tier2": """
[6:55-7:20] Tier two. Two loops over five qubits. Ten operations, five completely
independent chains - and not one of those ten operations appears literally in the code. The
model has to run the loop in its head, including the angle pi over i plus one changing each
iteration.
""",

"tier3": """
[7:20-7:50] Tier three, which is where it gets interesting. The if-else on the loop index
puts RX on the even qubits and RZ on the odd ones. Then a nested loop produces six cx
gates in a specific order. Ten operations again, exactly like tier one - but none of them
are written down, and getting the edges right means tracking four wires at once.
""",

"tier4": """
[7:50-8:15] Tier four is textbook algorithms - this one is a GHZ state with measurement.
Two things to notice: the model has to recognise the pattern rather than just read it, and
measurement brings classical bits in, so those nodes carry both a qubit and a clbit and the
graph now has two kinds of wire.
""",

"curation": """
[8:15-8:50] Curation, since the instructions ask for it specifically. One command, fixed
seed, nothing labelled by hand. Two safeguards matter. Every ground-truth graph goes
through my own validator before it is kept, so a malformed extraction fails the build
rather than quietly poisoning the data. And I deduplicate on the graph itself across the
whole dataset, so no DAG can appear in both train and test - memorising cannot score. The
last line matters for phase two: five hundred per tier is deliberate for prompting, and the
generator scales to four thousand per tier when fine-tuning needs more.
""",

"external": """
[8:50-9:25] My generated data has one weakness: a house style. A fixed vocabulary of gates,
a fixed set of code shapes. A model could learn my generator's habits instead of the rule,
and my own test split would never tell me. So I bring in real code. The key point is the
line in the middle: every one of these ends up as a Qiskit QuantumCircuit object - either
by running the Python, or by loading OpenQASM - and then the same circuit_to_dag gives me
the ground truth and the same metrics score it. No separate evaluation path, so no way to
flatter the model.
""",

"metrics": """
[9:25-10:05] Four metrics. Valid is a structural check that never looks at the correct
answer: does it parse, no cycles, do each edge's wires belong to both endpoints, is each
wire one unbroken chain. That separates nonsense from well-formed-but-wrong, which are very
different failures. Exact is the headline - the same graph, by labelled isomorphism. The two
F1 scores give partial credit so I can see progress before anything is exactly right. The
box at the bottom is the sanity check on the metrics themselves: a correct graph scores one;
change one gate and it is still valid but no longer exact; delete one edge and the validator
names the broken wire.
""",

"status": """
[10:05-10:55] So where am I? Honestly: the infrastructure is done and verified, and the
model experiments have not been run. That sanity check is my evidence the harness works.
The dataset is built - about sixteen hundred samples - and I have inspected an example from
each tier, which you have just seen. The whole Phase-1 harness is written too - four prompt variants, a runner
with a hosted and a local backend, and a scorer - and I have dry-run it end to end. What I
have not done is call a real model, and I would rather stand here with no numbers than with
numbers I cannot defend.
The experiment I most want is the ablation: the model outputs only the gate list, and a
small module builds the edges from it. I checked that edge-builder against all one hundred
and sixty test samples and it reproduces the ground truth exactly - so whatever it scores is
purely the model's reading of the code. If the score jumps, the hard part is the graph. If
it does not, the hard part is executing the code. That splits my problem in half, cheaply.
""",

"roadmap": """
[10:55-11:30] The roadmap, cheapest first, so every phase gives a usable result even if the
next one does not happen. Phase one is prompting - the next two weeks. Phase two is LoRA
fine-tuning a small open code model on a free GPU, and the interesting question is whether a
fine-tuned one-billion model beats a prompted frontier model on a task this narrow. Phase
three, if there is time, is a tiny transformer from scratch. Phase four is the
generalisation tests, including going back to torch.fx as a second domain. Phase five is a
small interface. The four immediate subtasks are at the bottom.
""",

"questions": """
[11:30-12:05] Which brings me to the four questions I want to answer. How does accuracy fall
as the code hides more of the structure - and is that a different curve from the circuit
simply getting bigger? Is the hard part executing the code, or building the graph?
Prompting, fine-tuning or from scratch - best accuracy for the compute spent? And do the
models learn the rule or memorise the pattern? Bigger circuits, held-out constructs and real
code will tell me. Thank you.
""",

"refs": """
Leave up during questions. Format follows the course instruction: authors, title, venue,
year, URL for papers; authors, URL, access date for websites.
""",
}


# --------------------------------------------------------------------------
def build():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    makers = [
        slide_title, slide_outline,
        slide_circuit_basics, slide_why_dag, slide_qiskit,
        slide_task, slide_hard,
        slide_adapt, slide_background,
        slide_pipeline,
        slide_tiers, slide_tier1, slide_tier2, slide_tier3, slide_tier4,
        slide_curation, slide_external, slide_metrics,
        slide_status, slide_roadmap, slide_questions, slide_references,
    ]
    for i, make in enumerate(makers, 1):
        slide = make(prs)
        if i > 1:
            footer(slide, i)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"{len(makers)} slides -> {OUT}")
    if len(makers) > 25:
        raise SystemExit("ERROR: over the 25-slide limit set by the course instructions")


if __name__ == "__main__":
    build()

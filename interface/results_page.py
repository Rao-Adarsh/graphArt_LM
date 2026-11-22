"""Page 1 (home) and page 5 (results): the numbers and figures of the report and poster."""

from __future__ import annotations

import gradio as gr

from interface.common import ROOT

ASSETS = ROOT / "interface" / "assets"

HOME = """
# Can a language model draw a network's graph from code?
**IE 643, team Pensieve.** A language model reads a PyTorch `nn.Module` (its class code and `forward()`)
and, without running it, writes the forward-pass graph: layers, connections, nesting.
torch.fx traces the true graph; an answer is **exact** when it is the same labelled graph, and
**lenient** when it is the same computation written differently (layer settings left out,
`F.relu` for `nn.ReLU`, `a+b` for `b+a`).

| Page | What it shows |
|---|---|
| **Flat format** | the whole fx graph, one node per line [M1–M7]: saved answers of every model, or the 1.5B live |
| **Hierarchical format** | module tree + one wiring graph per class [M8–M11]: 1.5B and 14B, or the 1.5B live |
| **Edits** | does the answer change when the code changes? [D5, E2] |
| **Results** | the tables and figures of the report |

Live runs use the fine-tuned Qwen2.5-Coder 1.5B on this laptop's CPU (free, no network).
The torchvision holdout [D4] shows saved answers only: it was scored once. The locked
hierarchical sets [D9] are not loaded.
"""

TABLE1 = """
### Graphs right on real torchvision code, exact / lenient
| | Flat: dev [D3], 21 blocks | Flat: unseen [D4], 17 blocks | Hier.: dev blocks, 21 | Hier.: dev networks [D7], 30 |
|---|---|---|---|---|
| 1.5B, fine-tuned | 6 / 11 | 0 / 0 | 9 / 9 | 0 / 0 |
| 14B, prompted | 1 / 4 | not run | 0 / 1 | 0 / 2 |
| 1.5B + 14B, joined | 8 / 13 | not run | 9 / 10 | 0 / 2 |

In distribution (our generator's 40 test programs [D1]) the fine-tuned adapters are 40/40.
"""

RESULTS = """
### R1. Prompting [M1–M4] on generated code [D1]: size decides it
Right topology: 1.5B 5% < 7B 25% < 14B 60% < Gemma 31B 93%.

### R2. Fine-tuning [M5]: perfect at home, lost on real code
The 1.5B after LoRA: 40/40 on generated test code [D1], **0/21** on real torchvision [D3]. It
draws chains: SqueezeNet's Fire block, two branches joined by `cat`, comes out as one line.

### R3. Real-looking training data, then split the task [D2, M6, M7]
| Step | torchvision dev [D3], of 21 |
|---|---|
| 26 Oct one-step 1.5B | 0 |
| + library-style training code [D2] (mixed-v2) | 7 |
| best lister: right nodes | 15 |
| 1.5B wirer, given the true nodes | 16 |
| joined: mixed-v2 lists, 1.5B wirer connects | **11** |
| joined: mixed-v2 lists, 14B wirer connects | 13 |

### R4. The locked test [D4]: 0/17 for every system
Even lenient, no unseen block right. 13 of the 17 import torchvision layers whose code is not in
the prompt (`Conv2dNormActivation`, `SqueezeExcitation`); others hinge on configuration flags or on
how fx handles `nn.Conv3d` subclasses.

### R5. Hierarchical format [M8, D6–D8]: the tree is the bottleneck
Real networks: 0/30 with the model's own tree; 10/30 when the true tree is given.

### R6. Bigger model, less output [M9, M11]
Lighter formats, the same replies of the tuned 1.5B scored for less (found after the results were
seen; to be declared before the locked sets are scored):

| | S3 all | S2 no layer settings | S1 skeleton | S0 class graphs only |
|---|---|---|---|---|
| dev blocks (21) | 9 | 11 | 11 | 16 |
| dev networks (30) | **0** | 3 | 5 | **9** |
| dev networks, true tree given | 10 | 10 | 10 | 10 |
| generated test blocks (24) | 21 | 21 | 22 | 22 |
| generated test networks (51) | 24 | 28 | 30 | 35 |

### E2. Does it follow the code? [D5]
On 69 meaningful edits the one-step mixed-v2 answers with the unedited graph 39 times, the two-step
model 14 times (exact McNemar p ≈ 4e-7); neutral edits stay right (90% and 88%).
"""

FIGURES = [("fig1_mc3.png", "Fig 1. Closest of the 17 unseen blocks [D4]: MC3, right but for one invented ReLU."),
           ("gantt_poster.png", "Fig 2. Workflow, 14 Oct – 22 Nov."),
           ("fig3_demo.png", "Fig 3. One program in both formats."),
           ("fig4_fire.png", "Fig 4. SqueezeNet's Fire block: the tuned 1.5B draws a chain."),
           ("fig5a_hier.png", "Fig 5(a). Hierarchical drawings right; aqua: true tree given."),
           ("fig5b_lighter.png", "Fig 5(b). Same replies, lighter formats: S3 (all) to S0 (class graphs only).")]


def build_home():
    gr.Markdown(HOME)
    gr.Markdown(TABLE1)
    p = ASSETS / "fig3_demo.png"
    if p.exists():
        gr.Image(str(p), label="one program, both formats", show_label=True, height=420, interactive=False)


def build_results():
    with gr.Tabs():
        with gr.Tab("Numbers"):
            gr.Markdown(TABLE1 + RESULTS)
        with gr.Tab("Figures"):
            for i in range(0, len(FIGURES), 2):
                with gr.Row():
                    for name, caption in FIGURES[i:i + 2]:
                        p = ASSETS / name
                        if p.exists():
                            gr.Image(str(p), label=caption, show_label=True, interactive=False, height=360)

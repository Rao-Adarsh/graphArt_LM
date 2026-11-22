"""Pensieve demo: PyTorch code -> forward-pass graph, in both answer formats.

    .venv/Scripts/python interface/app.py          then open http://127.0.0.1:7860

Five pages, each at its own address (the bar at the top switches between them):

    /         home: the question and Table 1                    results_page.build_home
    /flat     the flat format (whole fx graph)                  flat_page.py
    /hier     the hierarchical format (tree + class graphs)     hier_page.py
    /edits    does the answer follow code edits? (E2)           edits_page.py
    /results  the report's numbers and figures                  results_page.build_results

Saved answers (results/, kaggle_results/) display at once. Live answers come from the
fine-tuned Qwen2.5-Coder 1.5B on this machine's CPU: the base model is loaded once with the
three adapters (mixed-v2, the wirer, the hierarchical one) and they are switched per call.
Free; a block takes about half a minute to two minutes.

The reference graph of pasted code is traced by running its __init__ with torch.fx, which
executes it. The app listens on 127.0.0.1 only; paste only code you trust. The flat holdout
shows saved answers only, and the locked hierarchical sets are never opened.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from interface import edits_page, flat_page, hier_page, results_page
from interface.common import CSS

with gr.Blocks(title="Pensieve: code to graph") as demo:
    results_page.build_home()
with demo.route("Flat format", "/flat"):
    flat_page.build()
with demo.route("Hierarchical format", "/hier"):
    hier_page.build()
with demo.route("Edits", "/edits"):
    edits_page.build()
with demo.route("Results", "/results"):
    results_page.build_results()


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, css=CSS)

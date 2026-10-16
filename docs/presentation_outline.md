# Presentation outline (progress talk, ~10 min)

> Superseded by the built deck: `docs/presentation/` (regenerate with `scripts/make_slides.py`). Kept as the record of what the talk was meant to cover.

1. **Problem (1 slide).** "Can an LLM read code and draw the DAG it builds, without running it?" Show `docs/figures/tier3.png`.
2. **Why quantum circuit DAGs (1 slide).** Exact labels for free (`circuit_to_dag`), unlimited synthetic data, difficulty we can control, and strict rules we can check automatically. Also mention the change of direction away from PyTorch graphs, and why.
3. **What a circuit DAG is (1 slide).** Nodes are operations, edges are "next operation on this wire". Point out that gates on different qubits can be written in either order and still give the same DAG. That is why we compare graphs, not gate lists.
4. **Dataset (2 slides).** The 4 tiers with one figure each (`tier1.png` … `tier4.png`), the size table from the README, and the output format. Be upfront that tier 4 is small (93 samples) and only for testing.
5. **Evaluation (1 slide).** valid / exact / node F1 / edge F1, plus the sanity-check output showing how one wrong gate or missing edge is scored.
6. **LLM background (1–2 slides).** Next-token prediction, tokens (the `q12` → ` q`,`12` example), and prompting vs. LoRA fine-tuning vs. training from scratch. Sections 1–3 of `llm_primer.md` cover all of this.
7. **Roadmap and research questions (1 slide).** Phases 1–5 and the 4 questions from the README.
8. **Related work (1 slide).** Talk like a Graph (writing graphs as text), CRUXEval (predicting what code does), Qiskit HumanEval (LLMs writing Qiskit code).
9. **Next two weeks.** Prompting baseline on the test split, broken down by tier, and a Qiskit HumanEval importer.

If there is time before the talk: run the zero-shot prompt by hand in any chat LLM on 3–4 test samples (one per tier), paste the replies into `graphart.schema.from_text` + `graphart.metrics.compare`, and show the scores. Even 4 hand-checked results make the talk much more concrete.

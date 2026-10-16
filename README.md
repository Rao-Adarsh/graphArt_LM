# graphArt_LM

**Teaching a language model to read quantum-circuit code and draw the circuit's DAG, without running the code.**

![code → circuit → DAG](docs/figures/tier3.png)

> **Change of direction (Sept 2026).** The first plan was about PyTorch computational graphs. After talking with the TA, the goal is clearer: the project is about **drawing directed acyclic graphs (DAGs)**. Quantum circuit DAGs are the best fit for that, so the project now uses them only. The reasons are under [Why quantum circuit DAGs](#why-quantum-circuit-dags).

---

## What we are making

**Input:** Python/Qiskit source code that builds a circuit.
**Output:** the circuit's DAG, written as text in a fixed format.

```python
def entangle(qc, a, b):
    qc.h(a)
    qc.cx(a, b)

n = 4
qc = QuantumCircuit(n)
for i in range(0, n - 1, 2):
    entangle(qc, i, i + 1)
add_swaps = False
if add_swaps:
    qc.swap(0, n - 1)
else:
    qc.y(n - 1)
```

```
QUBITS 4
CLBITS 0
NODES
0 h q0
1 cx q0 q1
2 h q2
3 cx q2 q3
4 y q3
EDGES
0 -> 1 : q0
2 -> 3 : q2
3 -> 4 : q3
```

In a **circuit DAG**, each node is an operation (a gate or a measurement). Edge `A -> B` labelled `q3` means *B is the next thing that happens to qubit 3 after A*. Qiskit builds this structure inside itself (`circuit_to_dag`), and compilers use it for optimisation and routing.

### Why this is harder than it looks

1. **The code hides the circuit.** Loops must be unrolled, function calls expanded, and dead `if` branches ignored. Some calls broadcast: `qc.h(range(n))` is *n* gates.
2. **A gate list is not the same as a graph.** Gates on different qubits can be written in any order and still give the same DAG. To produce the edges, the model has to keep track of "the last operation on each wire" all the way through the program.
3. **The output must be exactly right.** One missed gate or edge breaks the chain on that wire. A check tool (`graphart/validate.py`) catches this automatically.

## Why quantum circuit DAGs

| Need | How circuit DAGs meet it |
|---|---|
| Correct answers for free | Run the code and call Qiskit's `circuit_to_dag`. The answer is exact, with no hand labelling. |
| Plenty of data | Write a program that generates random circuit code. The dataset can be as large as we like. |
| Control over difficulty | The code style can be varied (flat → loops → functions → algorithms) while circuit size stays fixed, or the other way round. |
| Automatic checking | A valid circuit DAG must follow strict rules (it has no cycles, and each wire is one chain), so broken outputs are detected without a human. |
| Fits a small context window | Useful circuits have 5–60 nodes. That is short enough for small models. |

The PyTorch version failed on the first row: `torch.fx` graphs use a different vocabulary from anything an LLM would write, so they could not be compared directly.

---

## Project roadmap

The phases go from cheapest to most involved. Each one uses the same dataset and evaluation code, so the results can be compared directly.

### Phase 0: Dataset and evaluation harness  ✅ first version done
- Generator for code → DAG pairs in 4 difficulty tiers (see [Dataset](#dataset)).
- Ground truth from Qiskit, a structure validator, and metrics.
- Renderer that draws code | circuit | DAG side by side.

### Phase 1: Prompting baselines (no training)
- **Zero-shot:** the prompt in [prompts/zero_shot.txt](prompts/zero_shot.txt).
- **Few-shot:** add 2–3 solved examples to the prompt.
- **Chain-of-thought / two-step:** the model first writes the unrolled gate list, then the edges.
- **Ablation, "LLM unrolls, code connects":** the model outputs only the ordered gate list, and a short Python function builds the edges. This separates *understanding the code* from *building the graph*.
- Models: one strong hosted model (Claude / GPT / Gemini API) and one small open model run locally (e.g. through Ollama). Report scores per tier.

### Phase 2: Fine-tuning a small open model
- Fine-tune a small open code model (roughly 0.5B–1.5B parameters, e.g. the Qwen-Coder family) with LoRA on the training split, using Hugging Face `transformers` + `peft` + `trl`. A free Colab or Kaggle GPU is enough.
- Questions: does a fine-tuned 1B model beat a prompted frontier model? How much data is needed (500 / 2k / 10k examples)?

### Phase 3 (stretch): Train a tiny transformer from scratch
- The "language" here is narrow and synthetic, so a small GPT (nanoGPT-style, about 10–50M parameters) trained from scratch may learn it. This gives a direct comparison: general pretrained knowledge versus learning only from our data.

### Phase 4: Generalisation tests
- **Larger circuits:** train on ≤5 qubits, test on 6–10.
- **New code patterns:** hold out one construct (e.g. `while` loops) during training.
- **Real code:** circuits written by people, not by our generator (see [External sources](#external-sources)).

### Phase 5: Interface
- Paste code → predicted DAG drawn next to the ground truth, with differences highlighted.

### Research questions
1. How does accuracy fall as the code hides more of the structure (tier 1 → 4)? And as the circuit grows?
2. Is the difficult part *executing the code in its head* or *building the graph*? (Answered by the Phase 1 ablation.)
3. Prompting vs. fine-tuning vs. from scratch: which gives the best accuracy for the compute spent?
4. Do the models learn the rules (generalise to bigger or new cases) or only memorise patterns?

---

## Dataset

### Synthetic generator (`graphart/generate.py`)

| Tier | What the code looks like | Unique samples (seed 0) | Nodes (min/avg/max) |
|---|---|---|---|
| 1 flat | one gate per line; reading is just copying | 500 | 3 / 9.2 / 17 |
| 2 loops | `for` loops over qubits, repeated layers, `% n` rings, reversed ranges | 500 | 3 / 13.5 / 42 |
| 3 control | helper functions, `if/else` on loop index, dead branches, nested loops, `while`, lists of angles | 500 | 3 / 8.6 / 23 |
| 4 algorithms | GHZ, QFT, Bernstein–Vazirani, Grover (2q), Deutsch–Jozsa, hardware-efficient ansatz | **93** | 6 / 17.4 / 26 |

The splits are 80/10/10 within each tier: **1274 train / 159 val / 160 test**. Every sample in the dataset has a different DAG, so no graph appears in both train and test.

Tier 4 has only 93 samples because only a few algorithm templates exist so far. It is best used as a **test-only** set until more templates or real code are added.

Each line of `data/generated/{train,val,test}.jsonl` looks like this:
```json
{"id": "t3-00012", "tier": 3, "code": "...", "target": "QUBITS 4\n...", "graph": {...},
 "num_qubits": 4, "num_nodes": 7, "num_edges": 5, "depth": 3}
```

### External sources

I could not find a public dataset of *code → circuit DAG* pairs, which is part of what makes this project new. The sources below provide circuits or code that can be turned into test data with the same extractor:

| Source | What it is | How we'd use it |
|---|---|---|
| [Qiskit HumanEval](https://huggingface.co/datasets/Qiskit/qiskit_humaneval) | 151 hand-written Qiskit tasks with reference solutions | **Real-code test set.** Run each solution that builds a circuit and extract its DAG. |
| [MQT Bench](https://github.com/munich-quantum-toolkit/bench) | Thousands of benchmark circuits (2–130 qubits), generated in Python | More algorithms for tier 4, plus larger circuits for Phase 4 |
| [QASMBench](https://github.com/pnnl/QASMBench) | Around 130 OpenQASM circuits from chemistry, arithmetic, ML, … | An "OpenQASM in, DAG out" variant, and realistic circuit shapes |
| [QCircuitBench](https://arxiv.org/abs/2410.07961) | A large dataset for LLM-based quantum algorithm design | Algorithm code in a style different from our generator |

### Graph format (`graphart/schema.py`)
- Only operation nodes appear. Qiskit's input/output nodes are dropped.
- If two consecutive gates share several wires (e.g. `cx(0,1)` then `cz(0,1)`), that is **one** edge listing both wires.
- Angles are rounded to 3 decimal places. Node ids follow Qiskit's topological order. The metrics do not depend on ids, so a model may number nodes differently.

## Evaluation (`graphart/validate.py`, `graphart/metrics.py`)

| Metric | Meaning |
|---|---|
| **valid** | The output parses, has no cycles, every edge's wires belong to both endpoints, and each wire forms a single chain |
| **exact** | The output is the same graph as the ground truth (checked with labelled graph isomorphism). This is the headline number. |
| **node F1** | Overlap between the predicted and true sets of (gate, qubits, clbits, params) |
| **edge F1** | Overlap between the predicted and true sets of (source node, target node, wires) |

Scores are reported per tier and per circuit size.

Sanity check (`scripts/make_figures.py`), comparing a correct graph against two slightly damaged copies:
```
perfect copy      valid=True  exact=True  node_f1=1.00 edge_f1=1.00
one wrong gate    valid=True  exact=False node_f1=0.93 edge_f1=0.92
one missing edge  valid=False exact=False node_f1=1.00 edge_f1=0.96  ['wire q2: 5 nodes should be chained by 4 edges, found 3']
```

---

## Quickstart

```bash
python -m venv .venv
```
```bash
.venv/Scripts/pip install -r requirements.txt
```
```bash
.venv/Scripts/python scripts/build_dataset.py --per-tier 500
```
```bash
.venv/Scripts/python scripts/make_figures.py
```
```bash
.venv/Scripts/python scripts/make_slides.py
```
(On macOS/Linux, use `.venv/bin/` instead of `.venv/Scripts/`.)

### Phase 1: prompting

```bash
.venv/Scripts/pip install -r requirements-pipeline.txt
```

**Spending gate.** Any run that would cost money prints an estimate and then refuses
unless you pass `--yes`. Free models, Ollama and the mock backend proceed on their own,
and still print `$0.0000` so the number is always on screen. If the price of a model
cannot be looked up, the run is treated as paid and blocked.

```
  ==============================================================
  COST ESTIMATE   anthropic / claude-opus-5
    40 samples   ~10,607 input tokens   ~6,059 output tokens
    rate: $5.000/1M in, $25.000/1M out
    ESTIMATED COST: $0.2045
  ==============================================================
```

Find model ids and current prices (no key needed):
```bash
.venv/Scripts/python -m pipeline.list_models qwen coder
```
```bash
.venv/Scripts/python -m pipeline.list_models --free
```

See exactly what would be sent, and what it would cost, without sending anything:
```bash
.venv/Scripts/python -m pipeline.run_prompting --prompt few_shot --backend mock --dry-run
```

Run it, score it, and compare runs side by side:
```bash
.venv/Scripts/python -m pipeline.run_prompting --prompt zero_shot --backend openrouter --model "qwen/qwen-2.5-coder-32b-instruct" --stratify 10 --yes
```
```bash
.venv/Scripts/python -m pipeline.score results/openrouter__qwen-qwen-2.5-coder-32b-instruct__zero_shot.jsonl --slides
```
```bash
.venv/Scripts/python -m pipeline.compare results/*.jsonl --markdown
```

`--stratify 10` takes ten samples per tier — the *same* ten for every model, which is
what makes runs comparable. Every reply is cached under `results/raw/`, so a re-run only
pays for what is missing, and re-scoring is always free.

`--cached-only` rebuilds a results file from replies already on disk. It never constructs
a backend and never opens a connection, so it cannot cost anything and needs no API key:
```bash
.venv/Scripts/python -m pipeline.run_prompting --prompt zero_shot --backend openrouter --model "qwen/qwen3.8-27b:free" --stratify 10 --cached-only
```

**Backends.** `openrouter` needs `OPENROUTER_API_KEY` and reaches every open-weight
model plus GPT and Gemini. `anthropic` needs `ANTHROPIC_API_KEY` from an Anthropic
Console account with credits — a Claude Pro subscription does **not** include API
access — and uses the native SDK rather than a proxy, so it keeps prompt caching,
effort control and refusal handling. `ollama` runs locally and costs nothing, but this
project's reference machine has no discrete GPU, so anything above ~1.5B is impractical.

`--slides` prints the per-tier rows ready to paste into `RESULTS` in
`scripts/make_slides.py`, which turns the deck's status slide into a results table.

## Repository layout

```
graphArt_LM/
├── graphart/
│   ├── schema.py      # graph format: dict <-> text, parser for LLM output
│   ├── generate.py    # synthetic Qiskit programs, tiers 1-4
│   ├── extract.py     # code -> QuantumCircuit -> ground-truth DAG
│   ├── validate.py    # structural checks for a circuit DAG
│   ├── metrics.py     # valid / exact / node F1 / edge F1
│   ├── connect.py     # gate list -> edges (the Phase 1 ablation)
│   └── render.py      # code | circuit | DAG figure
├── pipeline/
│   ├── backends.py    # Anthropic / OpenRouter / Ollama / mock, one class each
│   ├── cost.py        # pre-run cost estimate; paid runs need --yes
│   ├── list_models.py # look up model ids and live prices (no key needed)
│   ├── run_prompting.py   # prompt a model over a split, cache every reply
│   ├── score.py       # parse replies, score, break down by tier and size
│   └── compare.py     # several runs side by side -- the headline table
├── scripts/
│   ├── build_dataset.py
│   ├── make_figures.py
│   └── make_slides.py # builds the IE 643 deck
├── prompts/
│   ├── zero_shot.txt  # instructions only
│   ├── few_shot.txt   # + three solved examples from the train split
│   ├── cot.txt        # gate list first, then the edges
│   └── gates_only.txt # the ablation: no edges, connect.py builds them
├── docs/
│   ├── llm_primer.md  # LLM basics for this project + reading list
│   ├── figures/       # tier1..4 example figures
│   └── presentation/  # IE 643 prep deck (.pptx + .pdf)
└── data/generated/    # built by build_dataset.py (git-ignored; seed 0 reproduces it)
```

## Next steps
- [ ] Run the zero-shot baseline and record the first numbers.
- [ ] Freeze a test set: commit `test.jsonl` once the generator stops changing.
- [ ] Add more tier-4 templates (teleportation without classical control, the swap test, QPE, adders), and support `qc.compose` / `QuantumRegister`.
- [ ] Qiskit HumanEval importer → real-code test set.
- [ ] First results: zero-shot vs. few-shot vs. chain-of-thought on the test split, per tier.
- [ ] Regenerate at `--per-tier 4000` for Phase 2 (tiers 1-3 scale there cleanly; tier 4 is capped at 93 unique graphs until more templates exist).

New to LLMs? Start with **[docs/llm_primer.md](docs/llm_primer.md)**.

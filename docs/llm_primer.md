# LLMs for this project: a primer

This is written for someone who has never worked with LLMs. Every idea is explained using the task in this project: *Qiskit code in, circuit DAG out*. Read sections 1–4 before the presentation. Sections 5–7 matter once experiments begin.

---

## 1. What an LLM actually does

An LLM (large language model) is a function that takes a sequence of **tokens** and returns a **probability for every possible next token**. That is the whole interface. Longer text is produced by repeating one step:

```
prompt tokens ──► model ──► probabilities for the next token ──► pick one ──► append it ──► repeat
```

This is called **autoregressive generation**. For our task, the model does not "draw a graph". It writes the text `0 h q0\n1 cx q0 q1\n...` one token at a time. Each token depends only on the code and on what the model has written so far. A mistake early in the output therefore affects every later line. Our metrics measure exactly this.

### Tokens
Text is split into sub-word pieces before the model sees it. Here is real output from the `o200k_base` tokenizer (used by GPT-4o-class models):

| Text | Tokens |
|---|---|
| `for i in range(n - 1):⏎    qc.cx(i, i + 1)` | `for` · ` i` · ` in` · ` range` · `(n` · ` -` · ` ` · `1` · `):⏎` · `   ` · ` qc` · `.cx` · `(i` · `,` · ` i` · ` +` · ` ` · `1` · `)` (19 tokens) |
| `3 cx q2 q3` | `3` · ` cx` · ` q` · `2` · ` q` · `3` (6 tokens) |
| `rz(0.785) q12` | `rz` · `(` · `0` · `.` · `785` · `)` · ` q` · `12` |

What this means for us:
- The model never sees "qubit 12" as one unit. It sees ` q` followed by `12`. Numbers are awkward for LLMs, and our output is mostly numbers.
- The output format affects accuracy. A shorter format means fewer tokens, which means fewer chances to make a mistake and lower cost. Choosing the format is a real design decision: see the paper *Talk like a Graph* in the reading list.
- **Context window** is the maximum number of tokens (prompt plus output) the model can handle at once. Our samples are about 100–600 tokens, which is fine even for small models.

### Inside the model: the transformer, in one paragraph
Each token becomes a vector of numbers (an **embedding**). These vectors pass through a stack of **transformer layers**. The key operation inside each layer is **attention**: every position builds a query ("what am I looking for?"), compares it with every earlier position, and pulls in information from the best matches. When the model writes the edge `3 -> 4 : q3`, attention lets it look back to find the most recent node that touched `q3`, and to find `qc.y(n - 1)` in the code. The billions of **parameters** (weights) are the numbers inside these layers. Training sets their values.

### Picking the next token: sampling and temperature
The model outputs probabilities, so we still have to choose a token. **Temperature 0** (greedy decoding) always takes the most likely token. Use it for structured output like ours, where the answer should not vary from run to run. Higher temperatures add variety, which suits creative writing.

---

## 2. How an LLM gets built

| Stage | Data | What it learns |
|---|---|---|
| **Pretraining** | Trillions of tokens of web text, books and code (GitHub, which includes plenty of Qiskit) | Predicting the next token. This gives grammar, facts, and coding patterns. |
| **Supervised fine-tuning (SFT) / instruction tuning** | Tens of thousands to millions of (instruction, good answer) pairs | To follow instructions instead of simply continuing text |
| **Preference tuning (RLHF, DPO, …)** | Human or AI judgements of which answer is better | To be helpful, avoid harm, and match a style |

All three stages use the same training loop:
1. Show the model text.
2. Measure how wrong its next-token predictions were. This measure is the **loss**, specifically cross-entropy.
3. Nudge every weight slightly in the direction that lowers the loss. This is **gradient descent**, using gradients from **backpropagation**.
4. Repeat millions of times.

---

## 3. Three ways to use or adapt a model: this project's phases

The TA allowed any of these. Each one is a phase in the [roadmap](../README.md#project-roadmap).

### (a) Prompting: no training (Phase 1)
The weights stay fixed. You only change the input.
- **Zero-shot:** give instructions and the code, and ask for the graph ([prompts/zero_shot.txt](../prompts/zero_shot.txt)).
- **Few-shot:** also include 2–3 solved (code → graph) examples in the prompt. The model copies the format and the approach. This is called *in-context learning*.
- **Chain-of-thought (CoT):** ask the model to work step by step first, e.g. "write out the unrolled gate list, then the edges". Visible intermediate steps often improve accuracy on tasks that require tracking state.
- Cost: API fees, or a local GPU. Effort: low. **This is where to start.**

### (b) Fine-tuning: continue training on our data (Phase 2)
Start from a pretrained open model and keep training on our `(prompt = code, completion = target)` pairs from `train.jsonl`. The loss is computed only on the completion tokens.
- **Full fine-tuning** updates every weight. It needs a lot of GPU memory.
- **LoRA** freezes the original weights and trains small low-rank "adapter" matrices added next to them. Often less than 1% of the parameters are trainable, so a 0.5B–1.5B model fits on a free 16 GB Colab or Kaggle GPU.
- **QLoRA** is LoRA on a model whose frozen weights are stored in 4-bit form, so 7B-class models also fit on those GPUs.
- Tools: Hugging Face `transformers` (models), `datasets` (data loading), `peft` (LoRA), `trl` (`SFTTrainer`, which runs the training loop).
- Watch for **overfitting**: training loss keeps falling while validation loss rises. That is what the val split is for.

### (c) Pretraining a tiny model from scratch (Phase 3, stretch goal)
Start from random weights. A general-purpose LLM is out of reach, but our "language" is small and synthetic, so a tiny GPT (a few million to 50M parameters, in the style of Karpathy's nanoGPT) might learn it within a few GPU-hours. It could even use a custom tokenizer where `q12` is a single token. The scientific question is interesting: *is general pretraining necessary, or is enough synthetic data sufficient?*

| | Prompting | LoRA fine-tuning | From scratch |
|---|---|---|---|
| Data needed | 0–5 examples | ~1k–10k | 10k–1M (we can generate them) |
| Compute | inference only | 1 GPU, hours | 1 GPU, hours to days |
| Knowledge of Python | built in | built in | **none**, must learn from our data |
| Likely failure | format errors, losing track of wires | overfitting to our generator's style | poor generalisation to real code |

---

## 4. Why *this* task is hard for LLMs

- **Executing code in its head.** The model has to run `for i in range(0, n - 1, 2)` mentally. The CRUXEval benchmark found that models which score well at *writing* code often do much worse at *predicting what code does*.
- **Keeping state.** The edges depend on a table of "last node on each wire" that changes with every gate. A transformer has no scratch memory except the tokens it has already written, which is why chain-of-thought (writing the gate list first) may help.
- **Counting and indices.** Off-by-one errors in `range`, `n - j - 1`, and similar expressions are classic LLM mistakes.
- **Long, exact outputs.** One wrong token can make the whole graph invalid. Our metrics separate *invalid* from *valid but wrong* from *exact*.
- **Ordering does not matter, but the model outputs a sequence.** Gates on different qubits can come in any order, and the model has to write *some* order. Our metrics ignore node ids and order for exactly this reason.

---

## 5. Practical setup (when experiments start)

- **Hosted APIs** (Anthropic, OpenAI, Google): the quickest way to get strong baselines. You need an API key and pay per token. Keep the key in an environment variable and never commit it to git.
- **Local open models:** [Ollama](https://ollama.com) runs small models on a laptop with one command. It is good for fast prompting experiments.
- **Free GPUs:** Google Colab or Kaggle notebooks (about a 16 GB T4 or P100) are enough for LoRA on models up to about 1.5B parameters, or 7B with QLoRA.
- **Reproducibility:** fix random seeds, save the exact prompt used, use temperature 0, and record model name and version with every result.

---

## 6. Glossary

| Term | Meaning |
|---|---|
| Token | A piece of text (often part of a word) that the model reads or writes as one unit |
| Context window | Maximum number of tokens (input + output) the model can handle in one go |
| Embedding | The vector of numbers representing a token inside the model |
| Attention | The mechanism that lets each position pull in information from other positions |
| Parameters / weights | The learned numbers inside the model (0.5B = 500 million) |
| Loss | A number measuring how wrong the predictions are. Training tries to make it smaller. |
| Epoch | One full pass over the training data |
| Inference | Using a trained model to generate output, with no learning |
| Zero-shot / few-shot | Prompting with no examples / with a few examples |
| Chain-of-thought | Prompting the model to write intermediate reasoning before the answer |
| SFT | Supervised fine-tuning on (input, correct output) pairs |
| LoRA / QLoRA | Cheap fine-tuning with small adapter matrices / the same on a 4-bit compressed model |
| Overfitting | Memorising the training data instead of learning the underlying rule |
| Hallucination | Confident output that is wrong, e.g. a gate that is not in the code |

---

## 7. Reading and watching list (in order)

**Before the presentation (about 3 hours total)**
1. 3Blue1Brown, *Neural Networks* series, especially the later chapters on GPT/transformers and on attention. Best intuition available. [YouTube playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
2. Andrej Karpathy, *Intro to Large Language Models* (1 h talk). Covers pretraining, fine-tuning and the whole big picture. [YouTube](https://www.youtube.com/watch?v=zjkBMFhNj_g)
3. Jay Alammar, *The Illustrated Transformer*. The standard diagram-based walkthrough. [Blog](https://jalammar.github.io/illustrated-transformer/)
4. Qiskit, `DAGCircuit` docs: what the target structure actually is. [Docs](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.dagcircuit.DAGCircuit)

**When starting Phase 1–2**

5. Hugging Face *LLM Course*, a free hands-on course covering tokenizers, `transformers` and fine-tuning. [Course](https://huggingface.co/learn/llm-course)
6. Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (2022). [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
7. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

**Directly related work (good for the "related work" slide)**

8. Fatemi et al., *Talk like a Graph: Encoding Graphs for Large Language Models* (ICLR 2024): how the way a graph is written as text changes LLM accuracy. [arXiv:2310.04560](https://arxiv.org/abs/2310.04560)
9. Gu et al., *CRUXEval: Code Reasoning, Understanding, and Execution* (2024): LLMs predicting what code does without running it, which is the closest analogue to our input side. [arXiv:2401.03065](https://arxiv.org/abs/2401.03065)
10. *Qiskit HumanEval* (2024): LLMs writing Qiskit code. The reverse direction, and a source of real code for our test set. [arXiv:2406.14712](https://arxiv.org/abs/2406.14712)

**Phase 3 (from scratch)**

11. Karpathy, *Let's build GPT: from scratch, in code, spelled out* (2 h) and the [nanoGPT](https://github.com/karpathy/nanoGPT) repo. [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)

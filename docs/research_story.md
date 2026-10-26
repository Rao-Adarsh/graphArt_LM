# The research story

*How the questions changed as the results came in. Written as it happened, so the
reasoning behind each experiment is on record next to its outcome.*

## 1. The question we started with

Can a language model read the Python source of an `nn.Module` and write down its
forward-pass graph — every operation, and exactly which value feeds which argument —
without running the code? `torch.fx` gives the exact answer for free, so every reply can
be graded precisely, and the difficulty can be dialled: four tiers, from one layer per
line (tier 1) to residual CNNs, inception blocks, transformers and U-nets (tier 4).

## 2. Prompting (Phase 1): it scales, and topology comes last

The same 40 test programs (10 per tier) went to every model.

- **Scale decides it.** Zero-shot, the share of graphs with the right topology climbs
  0.05 → 0.25 → 0.60 from Qwen2.5-Coder 1.5B to 7B to 14B, and Gemma 4 31B reaches 0.93.
- **Chains come before branches.** The 7B handles half the straight chains and none of the
  branching code; branching first appears at 14B and is solved by 31B.
- **Reading and wiring are separate skills, and both break on branches.** Given the code
  alone, the 14B lists the right operations 70% of the time; given the right operations,
  it wires them 75% of the time; doing both at once, 60%.
- **Thinking out loud helps only when it is the right kind.** Generic step-by-step (`cot`)
  scored *below* plain zero-shot. Simulating the code line by line, tracking which node
  each variable points to (`cos`), was the best prompt on the hard tiers — but stumbled
  on containers like `nn.Sequential`. See *Why CoT and CoS did not simply win* below.
- **The big model's misses were conventions, not reasoning.** Every one of Gemma's ten
  strict misses was either writing `x.tanh()` as a function where `torch.fx` records a
  method, or leaving out the `getitem` nodes `torch.fx` creates when a tuple is unpacked.

Along the way the harness itself had to be corrected twice: the prompts did not define the
label format for most layer types (a correct graph failed to parse on `BatchNorm2d(8)`),
and a `topo` score was added so a spelling slip could not hide a correct topology.

### Why CoT and CoS did not simply win

The expectation was that the reasoning prompts would beat the plain ones. Overall they did
not: few-shot was best, and CoS finished level with zero-shot. Split by tier, the picture
changes. Counts are graphs with the right operations and the right connections, spelling
forgiven (the `right graph` level of `scripts/classify_errors.py`), out of 10 per tier,
Qwen2.5-Coder 14B:

| prompt | tier 1 flat | tier 2 loops | tier 3 branches | tier 4 blocks | total |
|---|---|---|---|---|---|
| zero-shot | 10 | 10 | 4 | 4 | 28 |
| few-shot | 10 | 10 | 5 | 6 | 31 |
| **CoS (ours)** | 10 | **3** | **7** | **7** | 27 |
| CoT | 9 | 9 | 3 | 3 | 24 |

- **CoS is the best prompt where the code hides the graph.** On tiers 3 and 4 (dead
  branches, reused layers, residual and inception blocks) it gets 14/20, against 11 for
  few-shot and 8 for zero-shot. Tracing which node each variable holds is exactly the work
  those tiers need.
- **One procedural flaw sinks its total.** On tier 2 (nested `nn.Sequential`), which every
  other prompt solves 10/10, CoS gets 3/10: it adds an extra node for the container, or
  expands the nesting only one level. Without tier 2, CoS leads with 24/30 to few-shot's 21/30.
- **CoT is worse everywhere.** Free-form reasoning more than doubles the reply (median
  1,652 characters against 698) without giving a procedure for this task — the answer is
  already a step-by-step list, so the reasoning restates it — and the longer reply drifts
  from the format more often (8 unreadable replies against 4 for zero-shot, after the
  parameter-stripping retry). CoT's large gains were reported at far larger model sizes, on
  tasks with a short final answer (Wei et al., 2022); this is neither.
- **Few-shot wins by fixing format, not reasoning.** Three worked examples cut replies that
  fail to parse from 21 to 4 (before the retry) and lift exact matches from 7 to 19. It
  shows the conventions instead of describing them.
- **So:** structured simulation helps where reasoning is the bottleneck; unstructured
  "thinking" does not help a mid-sized model on a format-heavy task; and where the format
  is the bottleneck, examples beat instructions.

Caveat: 10 programs per tier, so differences of one or two are noise. The tier-2 collapse
(3 vs 10) is real; the tier-3/4 lead is suggestive. Fixing the tier-2 rule and re-running
CoS would test it.

## 3. Fine-tuning (Phase 2): a result that looked too good

A LoRA adapter on the 1.5B model — the one that scored 0.05 — trained for 97 minutes on
one free T4 with the 1280 training programs. On the 40 test programs it scored **40/40
exact**, every tier, beating the prompted 31B (30/40 exact, 37/40 topology).

## 4. The turning point: "hold on, how did it do so well?"

The obvious worry was a leak. Every route was checked:

| possible leak | finding |
|---|---|
| the answer inside the test prompt | none — the prompt holds only the code |
| training reading the test split | it reads train and val only |
| test programs or answers in the training data | no identical program or answer; **one** test graph matched a training graph written in another node order — the builder deduplicated by text, not by graph. 39/39 without it |
| replies recalled from training answers | none of the 40 replies matches any training answer; all 40 match their own reference exactly |
| a scorer too lenient | the same scorer gives the untuned model 0/40 |

So the model writes new graphs rather than recalling old ones. But that settled less than
it seemed to. Fine-tuning showed the 1.5B 1280 examples of exactly the conventions the big
model got wrong, and every test program came from the same generator as the training
programs. A perfect score on similar programs cannot tell apart two very different things
the model might have learned:

- **the rule** — how data flows through Python code, and how `torch.fx` writes it down; or
- **the habits** of our generator — its templates, layer combinations and naming patterns.

That distinction is the difference between "LLMs can draw neural-network graphs" and
"LLMs can learn one generator's output format". And it is the same distinction as the
deeper question the project was always circling: **does the model understand the
topology, or only produce text that looks like it?**

"Understanding" cannot be measured directly, but it can be pinned to behaviour. A model
that tracks how data really flows should (a) keep working on code it was never shaped
for, (b) give the *same* answer when the code changes in ways that do not change the
graph, and the *right new* answer when it changes in ways that do, and (c) handle a
construct it never saw in training, if the construct follows from rules it did learn.
Because `torch.fx` gives ground truth for any program for free, each of these can be
tested exactly.

## 5. The questions that close the project

| | question | experiment | what each outcome would mean |
|---|---|---|---|
| **RQ4** | Does it work on real code? | `torchvision` building blocks (residual, fire, inverted-residual, inception, dense, shuffle, transformer blocks…) and two whole small networks, traced by `torch.fx` | Good on blocks, poor on whole networks → the limit is length, not reading. Poor on both → the skill is tied to our generator's style |
| **RQ5** | Is it tracking dataflow, or matching patterns? | Controlled edits of the test programs. **Neutral edits** keep the graph (rename variables, add an alias, reorder independent lines); **meaningful edits** change it (swap the operands of `a - b` or a `cat`, switch a dead branch on) | Right on both → behaviour consistent with tracking dataflow. Right on neutral but stuck on the *old* graph after meaningful edits → pattern-matching the template |
| **RQ6** | Can it apply a rule to a construct it never saw? | Fine-tune a second adapter with every program containing one construct removed from training, then test on programs that have it | Handles it → it learned something general. Fails where the full adapter succeeds → it learned the construct by example |
| — | How sure are the numbers? | Score the main models on all 160 test programs, not 40 | Narrower uncertainty for the write-up |

**The subject is the fine-tuned 1.5B.** Its perfect score is what raised these questions,
it is the fastest model we have, and RQ6 can only be asked of a fine-tuned model. So the
closing questions are, in one line: *what did fine-tuning teach it — the rule, or our
generator?* The prompted 14B is run on the same sets as a contrast, a generalist next to
the specialist. With a presentation due, each Kaggle run is held to 2.5 hours, using speeds
measured on earlier runs (the tuned 1.5B at ~24 tokens/s, the 14B at ~16); Gemma 31B, at ~5
tokens/s, would need over three hours for the edit set alone and waits for later.

The first check — the tuned model on 20 brand-new programs whose graphs appear in no split
— is the bridge between section 4 and these questions; its result is recorded below when
it lands.

## 6. Results of the closing experiments

### The bridge: 20 brand-new programs — 20/20 exact

Twenty programs were generated with a seed the dataset never used (five per tier, 9–19
nodes), and any whose graph was isomorphic to *anything* in train, val or test was thrown
away (two were). The tuned 1.5B, run locally on CPU with the same prompt and greedy
decoding, wrote every one of the twenty graphs exactly.

What this settles: the 40/40 was not a lucky test split, a leak, or recall of training
answers — the model produces correct graphs for programs whose graphs it has never seen.
What it does not settle: these programs still come from our generator. Whether the skill
survives code written by other people (RQ4), edits that break the generator's patterns
(RQ5), and a construct it was never shown (RQ6) is what the closing runs measure.

### How the closing runs were scored — and a second "hold on"

The 14B's results came back first, and the first reading of them was an explanation, not
a measurement: a few replies were read by eye and the torchvision failures put down to
"drawing at a different level of detail". The objection was immediate — *how can anyone
take that on trust, and are we grading graph-drawing or torch.fx's spelling?* That forced
two things to be made explicit.

**What every run, from the first to the last, is scored against.** The graph `torch.fx`
records when it traces the program's code, written in our schema, and compared *as a
graph* (labelled isomorphism) — so line order, numbering and layout never matter.
Torchvision is test material, not a referee: its true graphs come from the same tracer.
What a node must match does carry some `torch.fx` habits, though, and three levels of
strictness are reported side by side:

| level | a node must match | forgives |
|---|---|---|
| `exact` | operation type, name, layer class, layer settings | nothing |
| `topo` | the same, without layer settings | parameter spelling, function vs method |
| `right graph` | the operation, however it is spelled | also an activation written as a function where the code has a layer (or the reverse), `F.max_pool2d` vs `max_pool2d`, and `a + b` vs `b + a` |

All three still require every operation present and every connection right. The scorer
itself was **not** changed; `right graph` is one extra column, computed by the same fixed
rules on every run, old and new.

**Why a miss is a miss.** `scripts/classify_errors.py` sorts every reply into one category
by graph-isomorphism tests tried in a fixed order — right graph (`exact`, `notation`,
`spelling`, `commutative`); narrow error (`arg_slots`: right connections, an input in the
wrong slot of `cat`; `coarse_*`: a block drawn as one box and otherwise exactly right,
checked by re-tracing the real code with those blocks kept whole); wrong (`gave_original`,
`miswired`, `wrong_ops`, `unparseable`). Re-tracing reproduces every stored ground truth
checked (274/274), one constructed case per category lands where it should (11/11), and
every reply's category is written to `results/errors/` for audit.

**Why torchvision is a different exam.** It is real PyTorch, like our programs — but
*written by people*, and organised in ways the generator never produced. Counted:

| construct | train (1280) | torchvision (21) |
|---|---|---|
| its own block class, e.g. `BasicConv2d` | 0 | 10 |
| grouped convolutions | 0 | 5 |
| tuple kernel sizes, e.g. `(1,3)` | 0 | 3 |
| functions `torch.fx` does not shorten, e.g. `F.max_pool2d` | 0 | 7 |

The generator varied how code is *spelled* (class names, imports, attribute names,
activation idioms) but not how it is *organised*. So, before the fine-tuned model's results
were opened, the torchvision set was split by a mechanical rule: **in-distribution** (none
of these constructs, nor an `assert`; 5 samples) and **unseen constructs** (16).

*Disclosure.* The rule missed two things the training data never varies either: helper
*functions* (`conv3x3`, `conv1x1`, used by all three ResNet samples) and the input's name
(all 1,280 training programs call it `x`; MNASNet calls it `input`). The split was kept as
fixed rather than changed after seeing results; these are reported instead.

### RQ4 — real code: the fine-tuned model draws a chain

| torchvision, 21 samples | Qwen 14B, prompted | Qwen 1.5B + LoRA |
|---|---|---|
| replies that parse | 14 | 18 |
| exact / topo | 1 / 1 | 0 / 0 |
| **right graph** | **4** | **0** |
| node F1 / edge F1 | 0.31 / 0.20 | 0.36 / 0.17 |
| in-distribution (5), right graph | 4 | 0 |
| unseen constructs (16), right graph | 0 | 0 |
| **readable replies drawn as one straight chain** | 2 of 14 | **16 of 18** |

Only 4 of the 21 true graphs are straight chains. On real code the fine-tuned model lists
the layers in order and connects them one after another: in SqueezeNet's Fire block the
branch and the `cat` vanish; in ResNet's BasicBlock the skip connection is lost and a
`downsample` convolution appears that is `None` in the code. On our own test programs it
never does this (0 of 18 branching graphs drawn as a chain). Its branching skill is tied to
how our generator writes code. The prompted 14B still draws branches on real code and gets
4 of the 5 in-distribution samples; both fail every sample with unseen constructs.

### RQ5 — controlled edits: both models redraw the familiar graph

Counting only programs each model got right before the edit (14B: 24; LoRA: 40):

| | Qwen 14B | Qwen 1.5B + LoRA |
|---|---|---|
| neutral edits, right graph | 87% (61/70) | 94% (116/124) |
| meaningful edits, right graph | 22% (8/36) | 25% (17/69) |
| **meaningful edits answered with the *unedited* graph** | **64% (23/36)** | **68% (47/69)** |
| rerouted input followed | 0 / 14 | **0 / 27** |
| swapped `cat` order followed | 0 / 4 | 0 / 9 |
| deleted call noticed | 8 / 18 | 17 / 29 |

Both are robust to renaming and aliasing, and both fail in the same way when the wiring
changes: the answer is the graph the code *used to* have. The fine-tuned model never once
followed a rerouted input. It notices a deleted line more often than a changed argument —
consistent with matching which lines are present rather than reading which value feeds
which call. Reordering independent lines exposes the same habit: of 13 reorders, 7 only
swap the operands of `+` (the same computation) and 6 put `cat` inputs in written order.

### RQ6 — a held-out construct: a shortcut, not a rule

A second adapter was trained with every program containing a dead branch removed (1,112
examples, one epoch, 43 minutes), and both adapters were run on 84 probe programs:

| adapter | control (22) | dead branch (18) | dead flag added (40) | flag set to `True` (4) | dead layer written in |
|---|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | **0.00** | 0 of 58 |
| dead branches held out | 0.86 | **0.11** | 0.72 | 0.25 | **18 of 58** |

Trained without dead branches, the model does not work out that `if self.flag:` with the
flag `False` never runs — it writes the layer in. And the full model's apparent skill turns
out to be a shortcut: when the flag is set to `True`, so the layer *does* run, it still
deletes it, 4 times out of 4. In the training data every one of the 168 `if self.<flag>:`
branches has the flag `False`, and none has it `True`; "delete what sits under
`if self.flag:`" was a perfectly correct rule on that data, and it is what was learned.

### What the closing experiments answer

*What did fine-tuning teach — the rule, or our generator?* **Mostly our generator.** It
did teach real reading within that family (40/40 on the test split, 20/20 on fresh
programs, against 4/40 for the untuned model even with spelling forgiven). But three
independent tests show it leaning on regularities the data allowed: real code collapses to
a chain, changed wiring is answered with the familiar graph, and a flag is never read for
its value. "A fine-tuned 1.5B beats a prompted 31B" holds on our generator's code; as a
reader of code in general, the fine-tuned model is weaker than the prompted 14B.

The edit signature is shared by the prompted 14B, so it is not something fine-tuning
caused: both models answer partly from what code like this usually looks like. Neither
shows, on this evidence, that it tracks dataflow when the familiar shape and the actual
code disagree.

The result also says how the data should change: flags that are sometimes `True`, varied
input names, helper classes and functions, and layer types outside the current menu — each
one removes a shortcut the model was able to take.

Caveats: small groups (4 flag-`True` programs, 5 in-distribution torchvision samples); one
training seed; the held-out adapter trained one epoch against two, which probably explains
part of its control drop (1.00 → 0.86); the planned re-score on all 160 test programs was
not run.

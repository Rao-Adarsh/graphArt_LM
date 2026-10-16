# What `validate.py` does

**In one line:** it looks at a graph and asks *"could this be a real network at all?"* —
without ever looking at the right answer.

That last part is the important bit. It never compares against the correct graph. So it
separates two very different failures: the model wrote **nonsense**, versus the model wrote
something **sensible but wrong**. Those need different fixes, and a single score would hide
the difference.

## What it checks

| # | Check | Plain meaning |
|---|---|---|
| 1 | Node ids | No two nodes have the same number |
| 2 | Edges | Every arrow points at a node that exists, and nothing points at itself |
| 3 | Slots | Each connection listed once; no two things claim the same input slot |
| 4 | Output | Exactly one, and nothing comes after it |
| 5 | Input | At least one, and nothing feeds into it |
| 6 | Origin | Every node traces back to an input or a stored parameter — nothing appears from nowhere |
| 7 | Direction | No loops. Data only flows forward |
| 8 | Counts | A `Linear` gets one thing, an `add` gets two |
| 9 | Nesting | One layer can't sit inside another if both are listed separately |
| 10 | Sizes | A layer expecting 16 numbers must be fed 16; an `add` must get two compatible things |

## Does it work?

Yes, and it has been tested both ways round, which matters — a checker that approves
everything is useless.

- **All 15 real PyTorch models pass.** ResNet, DenseNet, MobileNet, ViT, Swin and the rest.
- **Eight kinds of deliberate damage are all caught**, each with a message saying what is
  wrong — a missing connection, a loop, two outputs, a wrong layer size, an invented node,
  a duplicated edge.

## The part worth remembering

The first version was **too strict in three places**, and each one rejected a *correct*
network:

- It assumed `view` takes one input. ShuffleNet's takes five, because some of its arguments
  are computed rather than written down.
- It assumed `cat` joins at least two things. DenseNet sometimes joins one.
- It assumed every node must lead to the output. ViT has checking steps that lead nowhere
  by design.

A checker that rejects correct answers is worse than no checker, because the error looks
like the model's fault. These were only found by running it against real networks —
hand-written examples would have passed all three.

## One deliberate choice

Size errors are **reported** but do not ruin the topology score. If the model draws the
network perfectly and gets one dimension wrong, it still counts as correct on structure,
and loses marks only on a separate size score (`shape_acc`). Otherwise a single arithmetic
slip would wipe out a perfect answer to the question the project is actually asking.

This is why `compare()` returns both `valid` (everything, including sizes) and
`valid_topology` (structure only). `exact` is gated on the second.

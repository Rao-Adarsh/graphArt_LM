"""The "model unrolls, code connects" ablation does not transfer to this domain.

In the circuit domain every wire was a chain, so the edges could be rebuilt
from an ordered gate list alone -- the next operation on a wire is simply the
next one that mentions it. That made it possible to ask a model for the
operations only and derive the structure in code, isolating "can it execute the
source" from "can it build the graph".

A forward pass has no such property. A residual connection runs from an early
node to a much later one:

    0 -> 4 : arg1        # the input feeding an add, five nodes downstream

and no ordering of the nodes recovers that edge. A derivation would succeed on
sequential networks and silently invent a chain wherever the topology actually
branched -- that is, it would look like it worked everywhere except where the
project's whole question lies.

The ablation is still worth running; it needs inverting. Give the model the
node list and ask only for the edges, which isolates graph construction
directly. That is a design change rather than a port, so it is not done here.
"""


def add_edges(graph: dict) -> dict:
    raise NotImplementedError(
        "edges cannot be derived from node order in a forward pass -- a skip "
        "connection is invisible to it. See the module docstring for the "
        "replacement experiment."
    )

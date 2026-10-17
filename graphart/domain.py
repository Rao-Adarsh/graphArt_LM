"""The one place that says which domain the harness is currently scoring.

Everything downstream -- `metrics`, and the whole of `pipeline/` -- imports the
schema, validator and edge-builder from here rather than from a specific
domain, so switching domains is an edit to this file alone.

The Qiskit circuit-DAG domain is retired but still intact under
`legacy_qiskit/`; repointing the imports below is all it takes to score it
again, which is the evidence that the harness is genuinely domain-agnostic.
"""

from .connect import add_edges                                     # noqa: F401
from .schema import GRAPH_HEADER, ParseError, from_text, to_text   # noqa: F401
from .validate import check, edge_label, node_label, to_networkx   # noqa: F401

DOMAIN = "pytorch-forward-pass"

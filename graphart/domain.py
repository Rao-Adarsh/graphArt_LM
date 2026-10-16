"""The one place that says which domain the harness is currently scoring.

Everything downstream -- `metrics`, and the whole of `pipeline/` -- imports the
schema, validator and edge-builder from here rather than from a specific
domain, so switching domains is an edit to this file alone.

To move from the Qiskit circuit-DAG domain to the PyTorch forward-pass domain,
repoint the three imports below. Nothing else changes.
"""

from .legacy_qiskit.connect import add_edges                       # noqa: F401
from .legacy_qiskit.schema import ParseError, from_text, to_text   # noqa: F401
from .legacy_qiskit.validate import check, node_label, to_networkx  # noqa: F401

DOMAIN = "qiskit-circuit-dag"

# tests/validator/test_models.py
from attractor.validator.models import Diagnostic, Severity


def test_diagnostic_creation():
    diag = Diagnostic(
        rule="start_node", severity=Severity.ERROR, message="Graph must have exactly one start node"
    )
    assert diag.rule == "start_node"
    assert diag.severity == Severity.ERROR


def test_diagnostic_with_node():
    diag = Diagnostic(
        rule="isolated_node",
        severity=Severity.WARNING,
        message="Node is not reachable from start",
        node_id="orphan_node",
    )
    assert diag.node_id == "orphan_node"


def test_diagnostic_with_edge():
    diag = Diagnostic(
        rule="edge_target",
        severity=Severity.ERROR,
        message="Edge target does not exist",
        edge=("from", "to"),
    )
    assert diag.edge == ("from", "to")

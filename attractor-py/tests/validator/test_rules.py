# tests/validator/test_rules.py
from attractor.parser import parse_dot
from attractor.validator.rules import EdgeTargetExistsRule, StartNodeRule, TerminalNodeRule


def test_start_node_rule_pass():
    source = """
    digraph Test {
        start [shape=Mdiamond]
        end [shape=Msquare]
        start -> end
    }
    """
    graph = parse_dot(source)
    rule = StartNodeRule()
    diagnostics = rule.apply(graph)
    assert len(diagnostics) == 0


def test_start_node_rule_fail_no_start():
    source = """
    digraph Test {
        end [shape=Msquare]
    }
    """
    graph = parse_dot(source)
    rule = StartNodeRule()
    diagnostics = rule.apply(graph)
    assert len(diagnostics) == 1
    assert diagnostics[0].severity.name == "ERROR"


def test_terminal_node_rule_pass():
    source = """
    digraph Test {
        start [shape=Mdiamond]
        end [shape=Msquare]
        start -> end
    }
    """
    graph = parse_dot(source)
    rule = TerminalNodeRule()
    diagnostics = rule.apply(graph)
    assert len(diagnostics) == 0


def test_edge_target_exists_rule_pass():
    source = """
    digraph Test {
        start [shape=Mdiamond]
        end [shape=Msquare]
        start -> end
    }
    """
    graph = parse_dot(source)
    rule = EdgeTargetExistsRule()
    diagnostics = rule.apply(graph)
    assert len(diagnostics) == 0


def test_edge_target_exists_rule_fail():
    source = """
    digraph Test {
        start [shape=Mdiamond]
        start -> nonexistent
    }
    """
    graph = parse_dot(source)
    rule = EdgeTargetExistsRule()
    diagnostics = rule.apply(graph)
    assert len(diagnostics) == 1
    assert "nonexistent" in diagnostics[0].message

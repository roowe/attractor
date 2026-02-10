# tests/parser/test_parser.py
from attractor.parser.parser import parse_dot


def test_parse_simple_graph():
    source = """
    digraph Test {
        graph [goal="Test goal"]
        start [shape=Mdiamond]
        end [shape=Msquare]
        start -> end
    }
    """
    graph = parse_dot(source)
    assert graph.goal == "Test goal"
    assert "start" in graph.nodes
    assert "end" in graph.nodes
    assert len(graph.edges) == 1


def test_parse_node_attributes():
    source = """
    digraph Test {
        test_node [label="Test", prompt="Do this", max_retries=3]
    }
    """
    graph = parse_dot(source)
    node = graph.nodes["test_node"]
    assert node.label == "Test"
    assert node.prompt == "Do this"
    assert node.max_retries == 3


def test_parse_edge_attributes():
    source = """
    digraph Test {
        A -> B [label="next", weight=5, condition="outcome=success"]
    }
    """
    graph = parse_dot(source)
    edge = graph.edges[0]
    assert edge.label == "next"
    assert edge.weight == 5
    assert edge.condition == "outcome=success"

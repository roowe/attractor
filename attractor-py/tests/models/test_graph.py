# tests/models/test_graph.py
from attractor.models.graph import Edge, Graph, Node


def test_graph_creation():
    graph = Graph(id="test", goal="Test goal", nodes={}, edges=[])
    assert graph.id == "test"
    assert graph.goal == "Test goal"


def test_node_creation():
    node = Node(id="test_node", label="Test Node", shape="box")
    assert node.id == "test_node"
    assert node.label == "Test Node"
    assert node.shape == "box"


def test_edge_creation():
    edge = Edge(from_node="start", to_node="end", label="next")
    assert edge.from_node == "start"
    assert edge.to_node == "end"
    assert edge.label == "next"

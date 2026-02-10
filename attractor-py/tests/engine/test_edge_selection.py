# tests/engine/test_edge_selection.py
from attractor.engine.edge_selection import select_edge
from attractor.models import Context, Edge, Graph, Node, Outcome, StageStatus


def test_select_edge_by_condition():
    context = Context()
    outcome = Outcome(status=StageStatus.SUCCESS)
    graph = Graph(
        id="test",
        nodes={"start": Node(id="start"), "success": Node(id="success"), "fail": Node(id="fail")},
        edges=[
            Edge(from_node="start", to_node="success", condition="outcome=success"),
            Edge(from_node="start", to_node="fail", condition="outcome=fail"),
        ],
    )

    edge = select_edge(graph.nodes["start"], outcome, context, graph)
    assert edge is not None
    assert edge.to_node == "success"


def test_select_edge_by_preferred_label():
    context = Context()
    outcome = Outcome(status=StageStatus.SUCCESS, preferred_label="Retry")
    graph = Graph(
        id="test",
        nodes={"start": Node(id="start"), "next": Node(id="next"), "retry": Node(id="retry")},
        edges=[
            Edge(from_node="start", to_node="next", label="Continue"),
            Edge(from_node="start", to_node="retry", label="Retry"),
        ],
    )

    edge = select_edge(graph.nodes["start"], outcome, context, graph)
    assert edge is not None
    assert edge.to_node == "retry"


def test_select_edge_by_weight():
    context = Context()
    outcome = Outcome(status=StageStatus.SUCCESS)
    graph = Graph(
        id="test",
        nodes={"start": Node(id="start"), "a": Node(id="a"), "b": Node(id="b")},
        edges=[
            Edge(from_node="start", to_node="a", weight=1),
            Edge(from_node="start", to_node="b", weight=5),
        ],
    )

    edge = select_edge(graph.nodes["start"], outcome, context, graph)
    assert edge is not None
    assert edge.to_node == "b"


def test_select_edge_lexical_tiebreaker():
    context = Context()
    outcome = Outcome(status=StageStatus.SUCCESS)
    graph = Graph(
        id="test",
        nodes={"start": Node(id="start"), "zebra": Node(id="zebra"), "apple": Node(id="apple")},
        edges=[
            Edge(from_node="start", to_node="zebra", weight=0),
            Edge(from_node="start", to_node="apple", weight=0),
        ],
    )

    edge = select_edge(graph.nodes["start"], outcome, context, graph)
    assert edge is not None
    # 词法决胜：apple < zebra
    assert edge.to_node == "apple"

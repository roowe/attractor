# tests/handlers/test_basic.py
from attractor.handlers.basic import ExitHandler, StartHandler
from attractor.models import Context, Graph, Node, StageStatus


def test_start_handler():
    handler = StartHandler()
    node = Node(id="start", shape="Mdiamond")
    context = Context()
    graph = Graph(id="test", nodes={}, edges=[])

    outcome = handler.execute(node, context, graph, "/tmp/logs")
    assert outcome.status == StageStatus.SUCCESS


def test_exit_handler():
    handler = ExitHandler()
    node = Node(id="exit", shape="Msquare")
    context = Context()
    graph = Graph(id="test", nodes={}, edges=[])

    outcome = handler.execute(node, context, graph, "/tmp/logs")
    assert outcome.status == StageStatus.SUCCESS

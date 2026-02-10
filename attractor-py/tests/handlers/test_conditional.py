# tests/handlers/test_conditional.py
from attractor.handlers.conditional import ConditionalHandler
from attractor.models import Context, Graph, Node, StageStatus


def test_conditional_handler():
    handler = ConditionalHandler()
    node = Node(id="gate", shape="diamond", label="Branch?")
    context = Context()
    graph = Graph(id="test", nodes={}, edges=[])

    outcome = handler.execute(node, context, graph, "/tmp/logs")

    # 条件节点是无操作的，实际路由由引擎处理
    assert outcome.status == StageStatus.SUCCESS
    assert "Conditional node evaluated" in outcome.notes

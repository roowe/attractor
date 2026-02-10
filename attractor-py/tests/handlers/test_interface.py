# tests/handlers/test_interface.py
from attractor.handlers.interface import Handler, HandlerRegistry
from attractor.models import Context, Graph, Node, Outcome, StageStatus


class MockHandler(Handler):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)


def test_handler_interface():
    handler = MockHandler()
    assert callable(handler.execute)


def test_registry_register_and_resolve():
    registry = HandlerRegistry()
    handler = MockHandler()

    registry.register("mock", handler)
    resolved = registry.resolve(Node(id="test", type="mock"))
    assert resolved is handler


def test_registry_default_handler():
    registry = HandlerRegistry()
    default = MockHandler()
    registry.default_handler = default

    node = Node(id="test")
    resolved = registry.resolve(node)
    assert resolved is default


def test_shape_based_resolution():
    registry = HandlerRegistry()
    handler = MockHandler()
    registry.register("start", handler)

    node = Node(id="test", shape="Mdiamond")
    resolved = registry.resolve(node)
    assert resolved is handler

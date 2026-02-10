# tests/handlers/test_codergen.py
from attractor.handlers.codergen import CodergenBackend, CodergenHandler
from attractor.models import Context, Graph, Node, Outcome, StageStatus


class MockBackend(CodergenBackend):
    def run(self, node: Node, prompt: str, context: Context) -> str | Outcome:
        return "Mock response"


def test_codergen_handler(tmp_path):
    handler = CodergenHandler(backend=MockBackend())
    node = Node(id="test", shape="box", prompt="Test prompt", label="Test Node")
    context = Context()
    graph = Graph(id="test", goal="Test goal", nodes={}, edges=[])

    outcome = handler.execute(node, context, graph, str(tmp_path))

    assert outcome.status == StageStatus.SUCCESS
    assert outcome.context_updates.get("last_stage") == "test"

    # 验证文件创建
    assert (tmp_path / "test" / "prompt.md").exists()
    assert (tmp_path / "test" / "response.md").exists()
    assert (tmp_path / "test" / "status.json").exists()

    # 验证内容
    prompt_content = (tmp_path / "test" / "prompt.md").read_text()
    assert "Test prompt" in prompt_content


def test_codergen_goal_expansion():
    handler = CodergenHandler(backend=MockBackend())
    node = Node(id="test", shape="box", prompt="Do this: $goal")
    context = Context()
    graph = Graph(id="test", goal="Save the world", nodes={}, edges=[])

    outcome = handler.execute(node, context, graph, "/tmp")

    # 验证 $goal 被展开
    assert outcome.status == StageStatus.SUCCESS

# tests/engine/test_executor.py
from attractor.engine.executor import ExecutorConfig, PipelineExecutor
from attractor.models import StageStatus
from attractor.parser import parse_dot


class MockBackend:
    def run(self, node, prompt, context):
        return f"Response for {node.id}"


def test_executor_simple_linear_pipeline(tmp_path):
    source = """
    digraph Test {
        graph [goal="Test goal"]
        start [shape=Mdiamond]
        step1 [shape=box, prompt="Step 1"]
        step2 [shape=box, prompt="Step 2"]
        exit [shape=Msquare]

        start -> step1 -> step2 -> exit
    }
    """
    graph = parse_dot(source)

    config = ExecutorConfig(logs_root=str(tmp_path), llm_backend=MockBackend())
    executor = PipelineExecutor(config)

    result = executor.run(graph)

    assert result.status == StageStatus.SUCCESS
    assert "start" in result.completed_nodes
    assert "step1" in result.completed_nodes
    assert "step2" in result.completed_nodes


def test_executor_conditional_branching(tmp_path):
    source = """
    digraph Test {
        graph [goal="Test goal"]
        start [shape=Mdiamond]
        step1 [shape=box, prompt="Step 1"]
        gate [shape=diamond]
        success [shape=box, prompt="Success path"]
        fail [shape=box, prompt="Fail path"]
        exit [shape=Msquare]

        start -> step1 -> gate
        gate -> success [condition="outcome=success"]
        gate -> fail [condition="outcome=fail"]
        success -> exit
        fail -> exit
    }
    """
    graph = parse_dot(source)

    config = ExecutorConfig(logs_root=str(tmp_path), llm_backend=MockBackend())
    executor = PipelineExecutor(config)

    result = executor.run(graph)

    assert result.status == StageStatus.SUCCESS
    # 应该走 success 路径
    assert "success" in result.completed_nodes

# tests/integration/test_smoke.py
from attractor import ExecutorConfig, PipelineExecutor, StageStatus, parse_dot, validate


class MockBackend:
    def run(self, node, prompt, context):
        return f"Mock response for: {prompt}"


def test_full_pipeline_execution(tmp_path):
    """完整的流水线执行测试"""
    dot_source = """
    digraph test_pipeline {
        graph [goal="Create a hello world Python script"]

        start       [shape=Mdiamond]
        plan        [shape=box, prompt="Plan how to create a hello world script for: $goal"]
        implement   [shape=box, prompt="Write the code based on the plan", goal_gate=true]
        review      [shape=box, prompt="Review the code for correctness"]
        done        [shape=Msquare]

        start -> plan
        plan -> implement
        implement -> review [condition="outcome=success"]
        implement -> plan   [condition="outcome=fail", label="Retry"]
        review -> done      [condition="outcome=success"]
        review -> implement [condition="outcome=fail", label="Fix"]
    }
    """

    # 1. 解析
    graph = parse_dot(dot_source)
    assert graph.goal == "Create a hello world Python script"
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 6

    # 2. 验证
    diagnostics = validate(graph)
    errors = [d for d in diagnostics if d.severity.value == "error"]
    assert len(errors) == 0

    # 3. 执行
    config = ExecutorConfig(logs_root=str(tmp_path), llm_backend=MockBackend())
    executor = PipelineExecutor(config)
    result = executor.run(graph)

    # 4. 验证结果
    assert result.status == StageStatus.SUCCESS
    assert "plan" in result.completed_nodes
    assert "implement" in result.completed_nodes
    assert "review" in result.completed_nodes

    # 5. 验证制品
    plan_dir = tmp_path / "plan"
    assert plan_dir.exists()
    assert (plan_dir / "prompt.md").exists()
    assert (plan_dir / "response.md").exists()
    assert (plan_dir / "status.json").exists()

    # 6. 验证目标门
    implement_outcome = result.node_outcomes["implement"]
    assert implement_outcome.status == StageStatus.SUCCESS

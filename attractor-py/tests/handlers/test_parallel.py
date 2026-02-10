# tests/handlers/test_parallel.py
import pytest
from attractor.models import (
    Context,
    Edge,
    Graph,
    Node,
    StageStatus,
)
from attractor.handlers.parallel import BranchResult, FanInHandler, ParallelHandler, ParallelResults


class MockLLMBackend:
    def run(self, node, prompt, context):
        return f"Mock response for: {prompt}"


def test_parallel_handler_creates_branch_results():
    """测试并行处理器创建分支结果"""
    handler = ParallelHandler()

    result = BranchResult(
        branch_id="test_branch",
        node_id="node1",
        status=StageStatus.SUCCESS,
    )

    assert result.branch_id == "test_branch"
    assert result.node_id == "node1"
    assert result.status == StageStatus.SUCCESS


def test_parallel_results_serialization():
    """测试并行结果序列化"""
    results = ParallelResults(
        branches=[
            BranchResult(
                branch_id="branch_0",
                node_id="node1",
                status=StageStatus.SUCCESS,
            ),
            BranchResult(
                branch_id="branch_1",
                node_id="node2",
                status=StageStatus.FAIL,
                error="Test error",
            ),
        ],
        success_count=1,
        failure_count=1,
        total_count=2,
    )

    data = results.to_dict()

    assert data["success_count"] == 1
    assert data["failure_count"] == 1
    assert data["total_count"] == 2
    assert len(data["branches"]) == 2
    assert data["branches"][0]["status"] == "success"
    assert data["branches"][1]["error"] == "Test error"


def test_parallel_handler_no_branches(tmp_path):
    """测试并行处理器处理无分支情况"""
    handler = ParallelHandler()

    node = Node(
        id="parallel_node",
        shape="component",
    )

    context = Context()
    graph = Graph(nodes={}, edges=[])

    outcome = handler.execute(node, context, graph, str(tmp_path))

    assert outcome.status == StageStatus.FAIL
    assert "No outgoing branches" in outcome.failure_reason


def test_fan_in_handler_no_results():
    """测试扇入处理器处理无结果情况"""
    handler = FanInHandler()

    node = Node(
        id="fanin_node",
        shape="tripleoctagon",
    )

    context = Context()
    graph = Graph(nodes={}, edges=[])

    outcome = handler.execute(node, context, graph, "/tmp")

    assert outcome.status == StageStatus.FAIL
    assert "No parallel results" in outcome.failure_reason


def test_fan_in_handler_selects_first_success():
    """测试扇入处理器选择第一个成功的分支"""
    handler = FanInHandler()

    node = Node(
        id="fanin_node",
        shape="tripleoctagon",
    )

    context = Context()
    context.set(
        "parallel.results",
        {
            "branches": [
                {
                    "branch_id": "branch_0",
                    "node_id": "node1",
                    "status": "fail",
                    "error": "Test error",
                },
                {
                    "branch_id": "branch_1",
                    "node_id": "node2",
                    "status": "success",
                },
            ],
            "success_count": 1,
            "failure_count": 1,
            "total_count": 2,
        },
    )

    graph = Graph(nodes={}, edges=[])

    outcome = handler.execute(node, context, graph, "/tmp")

    assert outcome.status == StageStatus.SUCCESS
    assert outcome.context_updates.get("parallel.fan_in.best_id") == "branch_1"


def test_parallel_handler_wait_all_policy(tmp_path):
    """测试并行处理器等待所有分支策略"""
    handler = ParallelHandler()

    # Create a graph with parallel branches
    node = Node(
        id="parallel_node",
        shape="component",
        attrs={"join_policy": "wait_all"},
    )

    context = Context()
    graph = Graph(
        nodes={
            "parallel_node": node,
            "branch1": Node(id="branch1", shape="box"),
            "branch2": Node(id="branch2", shape="box"),
        },
        edges=[
            Edge(from_node="parallel_node", to_node="branch1"),
            Edge(from_node="parallel_node", to_node="branch2"),
        ],
    )

    # Use a mock backend
    handler._llm_backend = MockLLMBackend()

    outcome = handler.execute(node, context, graph, str(tmp_path))

    # Should have results in context
    assert "parallel.results" in outcome.context_updates

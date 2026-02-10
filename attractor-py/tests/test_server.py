# tests/test_server.py
import json
import time
import threading
from http.server import BaseHTTPRequestHandler
import pytest
from attractor.server import (
    PipelineRun,
    PipelineServer,
    PipelineState,
    ServerConfig,
    create_default_server,
)
from attractor.models import StageStatus


class MockLLMBackend:
    def run(self, node, prompt, context):
        return f"Mock response for: {prompt}"


def test_server_config_creation():
    """测试服务器配置创建"""
    config = ServerConfig(
        host="localhost",
        port=9090,
        logs_root="./test_logs",
        llm_backend=MockLLMBackend(),
    )

    assert config.host == "localhost"
    assert config.port == 9090
    assert config.logs_root == "./test_logs"


def test_pipeline_run_creation():
    """测试流水线运行创建"""
    from attractor import parse_dot

    dot_source = '''
    digraph Test {
        graph [goal="Test goal"]
        start [shape=Mdiamond]
        exit [shape=Msquare]
        start -> exit
    }
    '''

    graph = parse_dot(dot_source)

    run = PipelineRun(
        run_id="test-run-1",
        graph=graph,
        logs_root="./logs",
    )

    assert run.run_id == "test-run-1"
    assert run.state == PipelineState.IDLE
    assert run.result is None
    assert run.error is None


def test_pipeline_run_state_transitions():
    """测试流水线运行状态转换"""
    from attractor import parse_dot

    dot_source = '''
    digraph Test {
        start [shape=Mdiamond]
        exit [shape=Msquare]
        start -> exit
    }
    '''

    graph = parse_dot(dot_source)
    run = PipelineRun(run_id="test", graph=graph)

    assert run.state == PipelineState.IDLE

    run.state = PipelineState.RUNNING
    assert run.state == PipelineState.RUNNING

    run.state = PipelineState.COMPLETED
    assert run.state == PipelineState.COMPLETED

    assert run.completed_at is None
    run.completed_at = time.time()
    assert run.completed_at is not None


def test_create_default_server():
    """测试创建默认服务器"""
    backend = MockLLMBackend()
    server = create_default_server(
        host="0.0.0.0",
        port=9999,
        logs_root="./logs",
        llm_backend=backend,
    )

    assert server.config.host == "0.0.0.0"
    assert server.config.port == 9999
    assert server.config.llm_backend is backend
    assert len(server.runs) == 0


def test_server_register_run():
    """测试服务器注册运行"""
    from attractor import parse_dot

    server = create_default_server()

    dot_source = '''
    digraph Test {
        start [shape=Mdiamond]
        exit [shape=Msquare]
        start -> exit
    }
    '''

    graph = parse_dot(dot_source)
    run = PipelineRun(
        run_id="test-run",
        graph=graph,
        logs_root="./logs",
    )

    server.runs["test-run"] = run

    retrieved = server.get_run("test-run")
    assert retrieved is run
    assert retrieved.run_id == "test-run"

    assert server.get_run("nonexistent") is None


def test_pipeline_run_with_error():
    """测试带错误的流水线运行"""
    from attractor import parse_dot

    graph = parse_dot("digraph Test { start [shape=Mdiamond] exit [shape=Msquare] start -> exit }")

    run = PipelineRun(
        run_id="test",
        graph=graph,
        logs_root="./logs",
    )

    run.state = PipelineState.FAILED
    run.error = "Test error message"

    assert run.state == PipelineState.FAILED
    assert run.error == "Test error message"


def test_pipeline_run_with_human_answer():
    """测试带人工答案的流水线运行"""
    from attractor import parse_dot

    graph = parse_dot("digraph Test { start [shape=Mdiamond] exit [shape=Msquare] start -> exit }")

    run = PipelineRun(
        run_id="test",
        graph=graph,
        logs_root="./logs",
    )

    run.state = PipelineState.WAITING_FOR_HUMAN
    run.pending_question = {"text": "Continue?", "options": ["Yes", "No"]}
    run.human_answer = "Yes"

    assert run.state == PipelineState.WAITING_FOR_HUMAN
    assert run.pending_question["text"] == "Continue?"
    assert run.human_answer == "Yes"


def test_pipeline_state_enum():
    """测试流水线状态枚举"""
    assert PipelineState.IDLE.value == "idle"
    assert PipelineState.RUNNING.value == "running"
    assert PipelineState.WAITING_FOR_HUMAN.value == "waiting_for_human"
    assert PipelineState.COMPLETED.value == "completed"
    assert PipelineState.FAILED.value == "failed"


def test_server_stores_multiple_runs():
    """测试服务器存储多个运行"""
    server = create_default_server()

    for i in range(5):
        from attractor import parse_dot

        graph = parse_dot("digraph Test { start [shape=Mdiamond] exit [shape=Msquare] start -> exit }")
        run = PipelineRun(
            run_id=f"run-{i}",
            graph=graph,
            logs_root="./logs",
        )
        server.runs[f"run-{i}"] = run

    assert len(server.runs) == 5

    for i in range(5):
        run = server.get_run(f"run-{i}")
        assert run is not None
        assert run.run_id == f"run-{i}"

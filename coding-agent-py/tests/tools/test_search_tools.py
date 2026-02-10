# coding-agent-py/tests/tools/test_search_tools.py
import pytest
import tempfile
from pathlib import Path
from coding_agent.tools.core.search_tools import (
    register_shell_tool,
    register_grep_tool,
    register_glob_tool,
)
from coding_agent.tools.registry import ToolRegistry
from coding_agent.exec.environment import LocalExecutionEnvironment


@pytest.mark.asyncio
async def test_shell_tool():
    env = LocalExecutionEnvironment()
    registry = ToolRegistry()

    register_shell_tool(registry)

    tool = registry.get("shell")

    result = await tool.executor({
        "command": "echo 'Hello, World!'",
        "description": "Print hello"
    }, env)

    assert "Hello, World!" in result
    assert "exit code" in result.lower()


@pytest.mark.asyncio
async def test_shell_tool_with_timeout():
    env = LocalExecutionEnvironment()
    registry = ToolRegistry()

    register_shell_tool(registry)

    tool = registry.get("shell")

    # Command that will timeout
    result = await tool.executor({
        "command": "sleep 5",
        "timeout_ms": 1000
    }, env)

    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_grep_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_grep_tool(registry)

        # Create test files
        Path(tmpdir, "test1.txt").write_text("hello world\nhello python\n")
        Path(tmpdir, "test2.txt").write_text("goodbye world\n")

        tool = registry.get("grep")

        result = await tool.executor({
            "pattern": "hello",
            "path": "."
        }, env)

        assert "hello" in result.lower()
        assert "test1.txt" in result


@pytest.mark.asyncio
async def test_grep_tool_with_glob_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_grep_tool(registry)

        # Create test files
        Path(tmpdir, "test.txt").write_text("hello\n")
        Path(tmpdir, "test.py").write_text("hello\n")

        tool = registry.get("grep")

        result = await tool.executor({
            "pattern": "hello",
            "path": ".",
            "glob_filter": "*.txt"
        }, env)

        assert "test.txt" in result
        assert "test.py" not in result


@pytest.mark.asyncio
async def test_glob_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_glob_tool(registry)

        # Create test files
        Path(tmpdir, "file1.txt").touch()
        Path(tmpdir, "file2.py").touch()
        Path(tmpdir, "test.txt").touch()

        tool = registry.get("glob")

        result = await tool.executor({
            "pattern": "*.txt",
            "path": "."
        }, env)

        assert "file1.txt" in result
        assert "test.txt" in result
        assert "file2.py" not in result

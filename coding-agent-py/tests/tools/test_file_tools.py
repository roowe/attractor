# coding-agent-py/tests/tools/test_file_tools.py
import pytest
import tempfile
from pathlib import Path
from coding_agent.tools.core.file_tools import (
    register_read_file_tool,
    register_write_file_tool,
    register_edit_file_tool,
)
from coding_agent.tools.registry import ToolRegistry
from coding_agent.exec.environment import LocalExecutionEnvironment


@pytest.mark.asyncio
async def test_read_file_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_read_file_tool(registry)

        # Create a test file
        test_file = Path(tmpdir, "test.txt")
        test_file.write_text("Hello\nWorld\nPython\n")

        # Get the tool
        tool = registry.get("read_file")
        assert tool is not None

        # Execute the tool
        result = await tool.executor({"file_path": "test.txt"}, env)

        assert "1|Hello" in result or "Hello" in result
        assert "World" in result
        assert "Python" in result


@pytest.mark.asyncio
async def test_read_file_tool_with_offset_and_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_read_file_tool(registry)

        # Create a test file with many lines
        test_file = Path(tmpdir, "test.txt")
        test_file.write_text("\n".join([f"Line {i}" for i in range(1, 101)]))

        tool = registry.get("read_file")

        # Read with offset and limit
        result = await tool.executor({
            "file_path": "test.txt",
            "offset": 50,
            "limit": 5
        }, env)

        assert "Line 50" in result
        assert "Line 54" in result
        assert "Line 55" not in result


@pytest.mark.asyncio
async def test_read_file_tool_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_read_file_tool(registry)

        tool = registry.get("read_file")

        result = await tool.executor({"file_path": "nonexistent.txt"}, env)

        assert "Error" in result or "not found" in result.lower()


@pytest.mark.asyncio
async def test_write_file_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_write_file_tool(registry)

        tool = registry.get("write_file")

        result = await tool.executor({
            "file_path": "new_file.txt",
            "content": "Hello, World!"
        }, env)

        assert "written" in result.lower() or "success" in result.lower()

        # Verify file was created
        test_file = Path(tmpdir, "new_file.txt")
        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"


@pytest.mark.asyncio
async def test_edit_file_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_edit_file_tool(registry)

        # Create a test file
        test_file = Path(tmpdir, "test.txt")
        test_file.write_text("Hello World\nGoodbye World\n")

        tool = registry.get("edit_file")

        result = await tool.executor({
            "file_path": "test.txt",
            "old_string": "Hello World",
            "new_string": "Hello Python"
        }, env)

        assert "replaced" in result.lower() or "success" in result.lower()

        # Verify the edit
        content = test_file.read_text()
        assert "Hello Python" in content
        assert "Hello World" not in content


@pytest.mark.asyncio
async def test_edit_file_tool_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_edit_file_tool(registry)

        tool = registry.get("edit_file")

        result = await tool.executor({
            "file_path": "test.txt",
            "old_string": "Nonexistent",
            "new_string": "Replacement"
        }, env)

        assert "not found" in result.lower() or "error" in result.lower()

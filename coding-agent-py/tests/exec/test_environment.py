# coding-agent-py/tests/exec/test_environment.py
import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from coding_agent.exec.environment import (
    ExecutionEnvironment,
    LocalExecutionEnvironment,
    ExecResult,
    DirEntry,
)
from coding_agent.exec.grep_options import GrepOptions


@pytest.mark.asyncio
async def test_local_environment_working_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        assert env.working_directory() == tmpdir


@pytest.mark.asyncio
async def test_local_environment_platform():
    env = LocalExecutionEnvironment()
    assert env.platform() in ["darwin", "linux", "windows"]


@pytest.mark.asyncio
async def test_local_environment_write_and_read_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        await env.write_file("test.txt", "Hello, World!")
        content = await env.read_file("test.txt")

        assert content == "Hello, World!"


@pytest.mark.asyncio
async def test_local_environment_file_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        assert not await env.file_exists("test.txt")

        await env.write_file("test.txt", "content")
        assert await env.file_exists("test.txt")


@pytest.mark.asyncio
async def test_local_environment_exec_command():
    env = LocalExecutionEnvironment()
    result = await env.exec_command("echo hello", timeout_ms=5000)

    assert result.stdout.strip() == "hello"
    assert result.exit_code == 0
    assert not result.timed_out
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_local_environment_exec_command_timeout():
    env = LocalExecutionEnvironment()
    result = await env.exec_command("sleep 5", timeout_ms=1000)

    assert result.timed_out is True
    assert "timeout" in result.stderr.lower() or result.exit_code != 0


@pytest.mark.asyncio
async def test_local_environment_list_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        # Create some files and directories
        Path(tmpdir, "file1.txt").touch()
        Path(tmpdir, "file2.py").touch()
        os.makedirs(Path(tmpdir, "subdir"))
        Path(tmpdir, "subdir", "nested.txt").touch()

        entries = await env.list_directory(".", depth=1)

        names = {e.name for e in entries}
        assert "file1.txt" in names
        assert "file2.py" in names
        assert "subdir" in names

        subdir_entry = next(e for e in entries if e.name == "subdir")
        assert subdir_entry.is_dir is True


@pytest.mark.asyncio
async def test_local_environment_grep():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        # Create test files
        Path(tmpdir, "test1.txt").write_text("hello world\nhello python\n")
        Path(tmpdir, "test2.txt").write_text("goodbye world\n")

        options = GrepOptions(pattern="hello", path=".")
        result = await env.grep(options)

        assert "hello world" in result
        assert "hello python" in result
        assert "test1.txt" in result


@pytest.mark.asyncio
async def test_local_environment_glob():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        # Create test files
        Path(tmpdir, "file1.txt").touch()
        Path(tmpdir, "file2.py").touch()
        Path(tmpdir, "test.txt").touch()

        results = await env.glob("*.txt", path=".")

        assert "file1.txt" in results
        assert "test.txt" in results
        assert "file2.py" not in results


@pytest.mark.asyncio
async def test_local_environment_read_file_with_offset_and_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        # Create file with numbered lines
        content = "\n".join([f"Line {i}" for i in range(1, 101)])
        await env.write_file("test.txt", content)

        # Read with offset
        result = await env.read_file("test.txt", offset=50, limit=10)
        lines = result.strip().split("\n")

        assert len(lines) == 10
        assert "Line 50" in lines[0]


@pytest.mark.asyncio
async def test_local_environment_initialize_and_cleanup():
    env = LocalExecutionEnvironment()
    await env.initialize()
    # Should not raise any errors
    await env.cleanup()

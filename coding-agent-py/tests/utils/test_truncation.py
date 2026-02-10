# coding-agent-py/tests/utils/test_truncation.py
import pytest
from coding_agent.utils.truncation import truncate_tool_output, truncate_output


def test_truncate_output_under_limit():
    output = "x" * 100
    result = truncate_output(output, max_chars=200, mode="head_tail")
    assert result == output


def test_truncate_output_head_tail_mode():
    output = "x" * 1000
    result = truncate_output(output, max_chars=200, mode="head_tail")

    assert result.startswith("x" * 100)
    assert result.endswith("x" * 100)
    assert "truncated" in result.lower() or "warning" in result.lower()
    assert "800" in result or "1000" in result  # Character count mentioned


def test_truncate_output_tail_mode():
    output = "x" * 1000
    result = truncate_output(output, max_chars=200, mode="tail")

    assert result.startswith("[Warning")
    assert result.endswith("x" * 200)
    assert "800" in result or "1000" in result


def test_truncate_tool_output_read_file():
    from coding_agent.models.config import SessionConfig

    config = SessionConfig()
    output = "x" * 60000  # Exceeds 50k default

    result = truncate_tool_output(output, "read_file", config)

    # Should be truncated
    assert len(result) < 60000
    assert "warning" in result.lower() or "truncated" in result.lower()


def test_truncate_tool_output_shell():
    from coding_agent.models.config import SessionConfig

    config = SessionConfig()
    output = "x" * 40000  # Exceeds 30k default

    result = truncate_tool_output(output, "shell", config)

    assert len(result) < 40000


def test_truncate_lines():
    from coding_agent.utils.truncation import truncate_lines

    output = "\n".join([f"Line {i}" for i in range(500)])

    result = truncate_lines(output, max_lines=100)

    lines = result.split("\n")
    # Should have fewer lines due to truncation message
    assert len([l for l in lines if "Line" in l]) <= 100
    assert "omitted" in result.lower()


def test_custom_tool_output_limit():
    from coding_agent.models.config import SessionConfig

    config = SessionConfig(tool_output_limits={"custom_tool": 100})
    output = "x" * 1000

    result = truncate_tool_output(output, "custom_tool", config)

    assert len(result) < 1000

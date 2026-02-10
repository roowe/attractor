# coding-agent-py/tests/models/test_config.py
import pytest
from pydantic import ValidationError
from coding_agent.models.config import SessionConfig


def test_session_config_defaults():
    config = SessionConfig()
    assert config.max_turns == 0
    assert config.max_tool_rounds_per_input == 200
    assert config.default_command_timeout_ms == 10000
    assert config.max_command_timeout_ms == 600000
    assert config.reasoning_effort is None
    assert config.enable_loop_detection is True
    assert config.loop_detection_window == 10
    assert config.max_subagent_depth == 1


def test_session_config_custom_values():
    config = SessionConfig(
        max_turns=100,
        default_command_timeout_ms=30000,
        reasoning_effort="high"
    )
    assert config.max_turns == 100
    assert config.default_command_timeout_ms == 30000
    assert config.reasoning_effort == "high"


def test_session_config_tool_output_limits():
    config = SessionConfig(
        tool_output_limits={"read_file": 100000, "shell": 50000}
    )
    assert config.tool_output_limits["read_file"] == 100000
    assert config.tool_output_limits["shell"] == 50000


def test_session_config_invalid_reasoning_effort():
    with pytest.raises(ValidationError):
        SessionConfig(reasoning_effort="invalid")


def test_session_config_invalid_timeout():
    with pytest.raises(ValidationError):
        SessionConfig(default_command_timeout_ms=-100)

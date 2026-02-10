# coding-agent-py/tests/providers/test_profile.py
import pytest
from coding_agent.providers.profile import AnthropicProfile
from coding_agent.exec.environment import LocalExecutionEnvironment


def test_anthropic_profile_creation():
    profile = AnthropicProfile(model="claude-opus-4-6")

    assert profile.id == "anthropic"
    assert profile.model == "claude-opus-4-6"
    assert profile.supports_reasoning is True
    assert profile.supports_streaming is True
    assert profile.supports_parallel_tool_calls is True


def test_anthropic_profile_tool_definitions():
    profile = AnthropicProfile(model="claude-opus-4-6")

    tools = profile.tools()
    tool_names = [t.name for t in tools]

    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "shell" in tool_names
    assert "grep" in tool_names
    assert "glob" in tool_names


def test_anthropic_profile_system_prompt():
    profile = AnthropicProfile(model="claude-opus-4-6")
    env = LocalExecutionEnvironment()

    prompt = profile.build_system_prompt(env, {})

    assert "Claude" in prompt
    assert "read_file" in prompt or "file" in prompt.lower()


def test_anthropic_profile_provider_options():
    profile = AnthropicProfile(model="claude-opus-4-6")

    options = profile.provider_options()

    assert options is not None
    assert "anthropic" in options


@pytest.mark.asyncio
async def test_profile_tool_registry():
    profile = AnthropicProfile(model="claude-opus-4-6")

    # Get tool from registry
    tool = profile.tool_registry.get("read_file")
    assert tool is not None
    assert tool.definition.name == "read_file"

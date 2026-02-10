# coding-agent-py/tests/test_session.py
import pytest
import asyncio
from coding_agent.session import Session
from coding_agent.providers.profile import AnthropicProfile
from coding_agent.exec.environment import LocalExecutionEnvironment
from coding_agent.models.config import SessionConfig


@pytest.mark.asyncio
async def test_session_creation():
    profile = AnthropicProfile(model="claude-opus-4-6")
    env = LocalExecutionEnvironment()

    session = Session(profile, env)

    assert session.id is not None
    assert session.provider_profile == profile
    assert session.execution_env == env
    assert session.state == "idle"
    assert len(session.history) == 0


@pytest.mark.asyncio
async def test_session_with_custom_config():
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()
    config = SessionConfig(max_turns=100, default_command_timeout_ms=30000)

    session = Session(profile, env, config)

    assert session.config.max_turns == 100
    assert session.config.default_command_timeout_ms == 30000


@pytest.mark.asyncio
async def test_session_steering():
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()

    session = Session(profile, env)

    session.steer("Try a different approach")

    assert len(session.steering_queue) == 1
    assert session.steering_queue[0] == "Try a different approach"


@pytest.mark.asyncio
async def test_session_follow_up():
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()

    session = Session(profile, env)

    session.follow_up("Now do this next task")

    assert len(session.followup_queue) == 1
    assert session.followup_queue[0] == "Now do this next task"


@pytest.mark.asyncio
async def test_session_close():
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()

    session = Session(profile, env)

    await session.close()

    assert session.state == "closed"

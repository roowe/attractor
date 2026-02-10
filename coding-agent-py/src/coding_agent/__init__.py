"""Coding Agent Loop - A programmable coding agent library with provider-aligned toolsets."""

__version__ = "0.1.0"

from coding_agent.session import Session
from coding_agent.models.config import SessionConfig
from coding_agent.models.turn import SessionState
from coding_agent.models.event import EventKind
from coding_agent.providers.profile import (
    AnthropicProfile,
    OpenAIProfile,
    GeminiProfile,
)
from coding_agent.exec.environment import LocalExecutionEnvironment

__all__ = [
    "Session",
    "SessionConfig",
    "SessionState",
    "EventKind",
    "AnthropicProfile",
    "OpenAIProfile",
    "GeminiProfile",
    "LocalExecutionEnvironment",
]

# coding-agent-py/src/coding_agent/models/turn.py
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

from coding_agent.models.tool import ToolCall, ToolResult
from coding_agent.models.usage import Usage


class SessionState(str, Enum):
    """Session lifecycle states."""
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


class UserTurn(BaseModel):
    """A user input turn."""
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AssistantTurn(BaseModel):
    """An assistant response turn."""
    content: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    reasoning: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)
    response_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolResultsTurn(BaseModel):
    """Results from executing tool calls."""
    results: List[ToolResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SystemTurn(BaseModel):
    """A system message turn."""
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SteeringTurn(BaseModel):
    """A steering message injected by the host."""
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Type alias for any turn type
Turn = UserTurn | AssistantTurn | ToolResultsTurn | SystemTurn | SteeringTurn

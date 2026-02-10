# coding-agent-py/src/coding_agent/models/event.py
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class EventKind(str, Enum):
    """Types of events emitted during session execution."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    ASSISTANT_TEXT_START = "assistant_text_start"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_OUTPUT_DELTA = "tool_call_output_delta"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    ERROR = "error"
    WARNING = "warning"


class SessionEvent(BaseModel):
    """An event emitted during session execution."""
    kind: EventKind
    session_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

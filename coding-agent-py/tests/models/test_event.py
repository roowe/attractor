# coding-agent-py/tests/models/test_event.py
import pytest
from datetime import datetime
from coding_agent.models.event import EventKind, SessionEvent


def test_event_kind_values():
    assert EventKind.SESSION_START == "session_start"
    assert EventKind.SESSION_END == "session_end"
    assert EventKind.USER_INPUT == "user_input"
    assert EventKind.ASSISTANT_TEXT_START == "assistant_text_start"
    assert EventKind.ASSISTANT_TEXT_DELTA == "assistant_text_delta"
    assert EventKind.ASSISTANT_TEXT_END == "assistant_text_end"
    assert EventKind.TOOL_CALL_START == "tool_call_start"
    assert EventKind.TOOL_CALL_END == "tool_call_end"
    assert EventKind.STEERING_INJECTED == "steering_injected"
    assert EventKind.TURN_LIMIT == "turn_limit"
    assert EventKind.LOOP_DETECTION == "loop_detection"
    assert EventKind.ERROR == "error"


def test_session_event_creation():
    event = SessionEvent(
        kind=EventKind.USER_INPUT,
        session_id="session_123",
        data={"content": "Hello"}
    )
    assert event.kind == EventKind.USER_INPUT
    assert event.session_id == "session_123"
    assert event.data["content"] == "Hello"
    assert isinstance(event.timestamp, datetime)


def test_session_event_with_tool_call_start():
    event = SessionEvent(
        kind=EventKind.TOOL_CALL_START,
        session_id="session_123",
        data={"tool_name": "read_file", "call_id": "call_456"}
    )
    assert event.kind == EventKind.TOOL_CALL_START
    assert event.data["tool_name"] == "read_file"
    assert event.data["call_id"] == "call_456"


def test_session_event_with_error():
    event = SessionEvent(
        kind=EventKind.ERROR,
        session_id="session_123",
        data={"message": "Something went wrong", "error_type": "ProviderError"}
    )
    assert event.kind == EventKind.ERROR
    assert event.data["message"] == "Something went wrong"
    assert event.data["error_type"] == "ProviderError"

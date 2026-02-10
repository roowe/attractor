# coding-agent-py/tests/models/test_turn.py
import pytest
from datetime import datetime
from coding_agent.models.turn import (
    SessionState,
    UserTurn,
    AssistantTurn,
    ToolResultsTurn,
    SystemTurn,
    SteeringTurn,
)


def test_session_state_values():
    assert SessionState.IDLE == "idle"
    assert SessionState.PROCESSING == "processing"
    assert SessionState.AWAITING_INPUT == "awaiting_input"
    assert SessionState.CLOSED == "closed"


def test_user_turn():
    turn = UserTurn(content="Hello, agent")
    assert turn.content == "Hello, agent"
    assert isinstance(turn.timestamp, datetime)


def test_assistant_turn_with_tool_calls():
    from coding_agent.models.tool import ToolCall
    from coding_agent.models.usage import Usage

    turn = AssistantTurn(
        content="I'll help you",
        tool_calls=[
            ToolCall(
                id="call_123",
                name="read_file",
                arguments={"file_path": "/path/to/file"}
            )
        ],
        reasoning="Let me think...",
        usage=Usage(prompt_tokens=10, completion_tokens=20),
        response_id="resp_456"
    )
    assert turn.content == "I'll help you"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "read_file"
    assert turn.reasoning == "Let me think..."
    assert turn.response_id == "resp_456"


def test_assistant_turn_without_tool_calls():
    turn = AssistantTurn(content="Done!")
    assert turn.content == "Done!"
    assert turn.tool_calls == []
    assert turn.reasoning is None


def test_tool_results_turn():
    from coding_agent.models.tool import ToolResult

    turn = ToolResultsTurn(results=[
        ToolResult(tool_call_id="call_123", content="File content", is_error=False)
    ])
    assert len(turn.results) == 1
    assert turn.results[0].tool_call_id == "call_123"


def test_system_turn():
    turn = SystemTurn(content="System message")
    assert turn.content == "System message"


def test_steering_turn():
    turn = SteeringTurn(content="Try a different approach")
    assert turn.content == "Try a different approach"

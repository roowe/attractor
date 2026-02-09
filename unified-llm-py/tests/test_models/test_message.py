"""Test Message model."""

import pytest
from unified_llm.models._message import Message
from unified_llm.models._content import ContentPart
from unified_llm.models._enums import Role, ContentKind


def test_message_user_factory():
    """Test creating a user message via factory."""
    msg = Message.user("Hello, world!")
    assert msg.role == Role.USER
    assert len(msg.content) == 1
    assert msg.content[0].kind == ContentKind.TEXT
    assert msg.content[0].text == "Hello, world!"


def test_message_system_factory():
    """Test creating a system message via factory."""
    msg = Message.system("You are a helpful assistant.")
    assert msg.role == Role.SYSTEM
    assert msg.content[0].text == "You are a helpful assistant."


def test_message_assistant_factory():
    """Test creating an assistant message via factory."""
    msg = Message.assistant("I can help with that.")
    assert msg.role == Role.ASSISTANT
    assert msg.content[0].text == "I can help with that."


def test_message_tool_result_factory():
    """Test creating a tool result message via factory."""
    msg = Message.tool_result(
        tool_call_id="call_123",
        content="Result text",
        is_error=False
    )
    assert msg.role == Role.TOOL
    assert msg.tool_call_id == "call_123"
    assert msg.content[0].kind == ContentKind.TOOL_RESULT


def test_message_text_accessor():
    """Test the text accessor concatenates text parts."""
    msg = Message(role=Role.USER, content=[
        ContentPart.text("Hello, "),
        ContentPart.text("world!"),
    ])
    assert msg.text == "Hello, world!"


def test_message_text_accessor_empty():
    """Test text accessor returns empty string for no text parts."""
    msg = Message(role=Role.USER, content=[
        ContentPart.image(url="https://example.com/img.png")
    ])
    assert msg.text == ""


def test_message_with_tool_call():
    """Test creating a message with tool call content."""
    msg = Message(role=Role.ASSISTANT, content=[
        ContentPart.tool_call(
            id="call_123",
            name="get_weather",
            arguments={"city": "SF"}
        )
    ])
    assert msg.role == Role.ASSISTANT
    assert msg.content[0].tool_call.name == "get_weather"

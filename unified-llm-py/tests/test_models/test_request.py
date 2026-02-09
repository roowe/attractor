"""Test Request and related models."""

import pytest
from unified_llm.models._request import Request, ToolChoice, ResponseFormat
from unified_llm.models._message import Message


def test_request_minimal():
    """Test creating a minimal request."""
    req = Request(
        model="claude-opus-4-6",
        messages=[Message.user("Hello")]
    )
    assert req.model == "claude-opus-4-6"
    assert len(req.messages) == 1
    assert req.provider is None


def test_request_with_all_fields():
    """Test creating a request with all fields."""
    req = Request(
        model="gpt-5.2",
        messages=[Message.user("Explain quantum computing")],
        provider="openai",
        temperature=0.7,
        max_tokens=1000,
    )
    assert req.provider == "openai"
    assert req.temperature == 0.7
    assert req.max_tokens == 1000


def test_tool_choice_auto():
    """Test ToolChoice auto mode."""
    tc = ToolChoice.auto()
    assert tc.mode == "auto"


def test_tool_choice_none():
    """Test ToolChoice none mode."""
    tc = ToolChoice.none()
    assert tc.mode == "none"


def test_tool_choice_required():
    """Test ToolChoice required mode."""
    tc = ToolChoice.required()
    assert tc.mode == "required"


def test_tool_choice_named():
    """Test ToolChoice named mode."""
    tc = ToolChoice.named("get_weather")
    assert tc.mode == "named"
    assert tc.tool_name == "get_weather"


def test_response_format_text():
    """Test ResponseFormat text type."""
    rf = ResponseFormat.text()
    assert rf.type == "text"


def test_response_format_json():
    """Test ResponseFormat json type."""
    rf = ResponseFormat.json()
    assert rf.type == "json"


def test_response_format_json_schema():
    """Test ResponseFormat json_schema type."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    rf = ResponseFormat.json_schema(schema)
    assert rf.type == "json_schema"
    assert rf.json_schema == schema

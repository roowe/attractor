"""Test Response, Usage, and related models."""

import pytest
from unified_llm.models._response import Response, Usage, FinishReason, RateLimitInfo, Warning
from unified_llm.models._message import Message
from unified_llm.models._enums import Role


def test_usage_basic():
    """Test Usage with basic fields."""
    usage = Usage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150


def test_usage_with_reasoning_tokens():
    """Test Usage with reasoning tokens."""
    usage = Usage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=200,
        reasoning_tokens=100,
    )
    assert usage.reasoning_tokens == 100


def test_usage_with_cache_tokens():
    """Test Usage with cache tokens."""
    usage = Usage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cache_read_tokens=80,
        cache_write_tokens=20,
    )
    assert usage.cache_read_tokens == 80
    assert usage.cache_write_tokens == 20


def test_usage_addition():
    """Test Usage addition."""
    a = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
    b = Usage(input_tokens=200, output_tokens=100, total_tokens=300)
    c = a + b
    assert c.input_tokens == 300
    assert c.output_tokens == 150
    assert c.total_tokens == 450


def test_usage_addition_with_optional_fields():
    """Test Usage addition with optional None fields."""
    a = Usage(input_tokens=100, output_tokens=50, total_tokens=150, reasoning_tokens=50)
    b = Usage(input_tokens=200, output_tokens=100, total_tokens=300)
    c = a + b
    assert c.reasoning_tokens == 50


def test_finish_reason():
    """Test FinishReason."""
    fr = FinishReason(reason="stop", raw="end_turn")
    assert fr.reason == "stop"
    assert fr.raw == "end_turn"


def test_rate_limit_info():
    """Test RateLimitInfo."""
    info = RateLimitInfo(
        requests_remaining=100,
        requests_limit=1000,
        tokens_remaining=50000,
        tokens_limit=100000,
    )
    assert info.requests_remaining == 100


def test_warning():
    """Test Warning."""
    w = Warning(message="Deprecated parameter", code="deprecated")
    assert w.message == "Deprecated parameter"
    assert w.code == "deprecated"


def test_response_basic():
    """Test Response."""
    resp = Response(
        id="resp_123",
        model="claude-opus-4-6",
        provider="anthropic",
        message=Message.assistant("Hello!"),
        finish_reason=FinishReason(reason="stop", raw="end_turn"),
        usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
    )
    assert resp.id == "resp_123"
    assert resp.text == "Hello!"
    assert resp.usage.total_tokens == 30


def test_response_with_tool_calls():
    """Test Response with tool calls."""
    from unified_llm.models._content import ContentPart

    resp = Response(
        id="resp_456",
        model="claude-opus-4-6",
        provider="anthropic",
        message=Message(
            role=Role.ASSISTANT,
            content=[ContentPart.tool_call(
                id="call_123",
                name="get_weather",
                arguments={"city": "SF"}
            )]
        ),
        finish_reason=FinishReason(reason="tool_calls", raw="tool_use"),
        usage=Usage(input_tokens=50, output_tokens=30, total_tokens=80),
    )
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_weather"

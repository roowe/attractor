"""Test content part models."""

import pytest
from unified_llm.models._content import (
    ContentPart,
    ImageData,
    AudioData,
    DocumentData,
    ToolCallData,
    ToolResultData,
    ThinkingData,
    ContentKind,
)


def test_text_content_part():
    """Test creating a text content part."""
    part = ContentPart.text("Hello, world!")
    assert part.kind == ContentKind.TEXT
    assert part.text == "Hello, world!"


def test_image_content_part_with_url():
    """Test creating an image content part with URL."""
    part = ContentPart.image(url="https://example.com/image.png")
    assert part.kind == ContentKind.IMAGE
    assert part.image.url == "https://example.com/image.png"


def test_image_content_part_with_data():
    """Test creating an image content part with base64 data."""
    part = ContentPart.image(
        data=b"fake_image_bytes",
        media_type="image/png"
    )
    assert part.kind == ContentKind.IMAGE
    assert part.image.data == b"fake_image_bytes"
    assert part.image.media_type == "image/png"


def test_tool_call_content_part():
    """Test creating a tool call content part."""
    part = ContentPart.tool_call(
        id="call_123",
        name="get_weather",
        arguments={"city": "San Francisco"}
    )
    assert part.kind == ContentKind.TOOL_CALL
    assert part.tool_call.id == "call_123"
    assert part.tool_call.name == "get_weather"
    assert part.tool_call.arguments == {"city": "San Francisco"}


def test_tool_result_content_part():
    """Test creating a tool result content part."""
    part = ContentPart.tool_result(
        tool_call_id="call_123",
        content="72F and sunny",
        is_error=False
    )
    assert part.kind == ContentKind.TOOL_RESULT
    assert part.tool_result.tool_call_id == "call_123"
    assert part.tool_result.content == "72F and sunny"
    assert part.tool_result.is_error is False


def test_thinking_content_part():
    """Test creating a thinking content part."""
    part = ContentPart.thinking(
        text="Let me think about this...",
        signature="sig_abc123"
    )
    assert part.kind == ContentKind.THINKING
    assert part.thinking.text == "Let me think about this..."
    assert part.thinking.signature == "sig_abc123"


def test_redacted_thinking_content_part():
    """Test creating a redacted thinking content part."""
    part = ContentPart.redacted_thinking(
        text="[REDACTED]",
        signature="sig_xyz789"
    )
    assert part.kind == ContentKind.REDACTED_THINKING
    assert part.thinking.redacted is True

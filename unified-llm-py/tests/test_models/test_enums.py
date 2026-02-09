"""Test enum definitions."""

import pytest
from unified_llm.models._enums import Role, ContentKind


def test_role_values():
    """Test Role enum has all required values."""
    assert Role.SYSTEM.value == "system"
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.TOOL.value == "tool"
    assert Role.DEVELOPER.value == "developer"


def test_role_from_string():
    """Test Role can be created from string."""
    assert Role("system") == Role.SYSTEM
    assert Role("user") == Role.USER


def test_content_kind_values():
    """Test ContentKind enum has all required values."""
    assert ContentKind.TEXT.value == "text"
    assert ContentKind.IMAGE.value == "image"
    assert ContentKind.AUDIO.value == "audio"
    assert ContentKind.DOCUMENT.value == "document"
    assert ContentKind.TOOL_CALL.value == "tool_call"
    assert ContentKind.TOOL_RESULT.value == "tool_result"
    assert ContentKind.THINKING.value == "thinking"
    assert ContentKind.REDACTED_THINKING.value == "redacted_thinking"


def test_content_kind_from_string():
    """Test ContentKind can be created from string."""
    assert ContentKind("text") == ContentKind.TEXT
    assert ContentKind("tool_call") == ContentKind.TOOL_CALL

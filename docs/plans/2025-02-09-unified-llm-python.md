# Python Unified LLM Client Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified LLM client library in Python that provides a single interface for multiple LLM providers (OpenAI, Anthropic, Google Gemini) with provider-agnostic APIs for generation, streaming, tool calling, and structured output.

**Architecture:** Four-layer architecture:
- Layer 1 (Provider Spec): ProviderAdapter interface and shared data models
- Layer 2 (Provider Utils): HTTP client, SSE parser, retry logic, normalization utilities
- Layer 3 (Core Client): Client with provider routing, middleware, configuration
- Layer 4 (High-level API): `generate()`, `stream()`, `generate_object()` with tool execution loops

**Tech Stack:** Python 3.11+, httpx (HTTP), pydantic (validation), pytest (tests), anyio (async)

---

## Project Setup

### Task 1: Create Project Structure

**Files:**
- Create: `pyproject.toml` (Project configuration with dependencies)
- Create: `src/unified_llm/__init__.py` (Package init)
- Create: `src/unified_llm/_exceptions.py` (Error hierarchy)
- Create: `tests/__init__.py` (Test package init)
- Create: `tests/conftest.py` (Pytest fixtures)
- Create: `.env.example` (Example environment variables)
- Create: `.gitignore` (Git ignore patterns)
- Create: `README.md` (Project documentation)

**Step 1: Initialize project with uv**

```bash
# Create library project with src layout
uv init --lib unified-llm

# Add core dependencies
uv add httpx pydantic anyio

# Add development dependencies
uv add --dev pytest pytest-asyncio pytest-cov ruff mypy
```

**Step 2: Create .env.example**

```
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Gemini
GEMINI_API_KEY=...
```

**Step 4: Create README.md**

```markdown
# Unified LLM Client

A unified Python client for OpenAI, Anthropic, and Google Gemini LLM providers.

## Installation

```bash
pip install unified-llm
```

## Quick Start

```python
import asyncio
from unified_llm import generate

async def main():
    result = await generate(
        model="claude-opus-4-6",
        prompt="Explain quantum computing in one paragraph"
    )
    print(result.text)

asyncio.run(main())
```

## Configuration

Set environment variables for your providers:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

## Features

- Unified API across OpenAI, Anthropic, and Gemini
- Streaming support with consistent events
- Tool calling with parallel execution
- Structured output generation
- Automatic retries with exponential backoff
- Provider-specific options via escape hatch
```

**Step 5: Create package __init__.py**

```python
"""
Unified LLM Client - A single interface for OpenAI, Anthropic, and Gemini.

This package provides a unified API for interacting with multiple LLM providers.
"""

__version__ = "0.1.0"

# High-level API (Layer 4)
from unified_llm.high_level import generate, stream, generate_object

# Core client (Layer 3)
from unified_llm.client import Client

# Provider adapters
from unified_llm.providers import (
    AnthropicAdapter,
    OpenAIAdapter,
    GeminiAdapter,
)

# Data models
from unified_llm.models import (
    Message,
    Request,
    Response,
    Usage,
    Tool,
    ToolChoice,
)

# Exceptions
from unified_llm._exceptions import (
    SDKError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
)

__all__ = [
    # Version
    "__version__",
    # High-level API
    "generate",
    "stream",
    "generate_object",
    # Core client
    "Client",
    # Providers
    "AnthropicAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    # Models
    "Message",
    "Request",
    "Response",
    "Usage",
    "Tool",
    "ToolChoice",
    # Exceptions
    "SDKError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
]
```

**Step 6: Run shell commands to verify structure**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python
ls -la src/unified_llm/
```

Expected: Directory listing shows `__init__.py` file

**Step 7: Commit**

```bash
git add pyproject.toml .gitignore .env.example README.md src/unified_llm/__init__.py
git commit -m "feat: create project structure and configuration"
```

---

## Layer 1: Provider Specification (Data Models)

### Task 2: Define Role Enum and ContentKind Enum

**Files:**
- Create: `src/unified_llm/models/_enums.py`
- Create: `tests/test_models/test_enums.py`

**Step 1: Write the failing test**

```python
# tests/test_models/test_enums.py
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
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_enums.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.models._enums'"

**Step 3: Write minimal implementation**

```python
# src/unified_llm/models/_enums.py
"""Enum definitions for the unified LLM client."""

from enum import Enum


class Role(str, Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class ContentKind(str, Enum):
    """Type of content in a message part."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_enums.py -v
```

Expected: PASS (5 tests passed)

**Step 5: Commit**

```bash
git add src/unified_llm/models/_enums.py tests/test_models/test_enums.py
git commit -m "feat: define Role and ContentKind enums"
```

---

### Task 3: Define ContentPart Data Classes

**Files:**
- Create: `src/unified_llm/models/_content.py`
- Create: `tests/test_models/test_content.py`

**Step 1: Write the failing test**

```python
# tests/test_models/test_content.py
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
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_content.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.models._content'"

**Step 3: Write minimal implementation**

```python
# src/unified_llm/models/_content.py
"""Content part data models for messages."""

from dataclasses import dataclass, field
from typing import Literal
from unified_llm.models._enums import ContentKind


@dataclass(frozen=True)
class ImageData:
    """Image data for multi-modal content."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    detail: Literal["auto", "low", "high"] | None = None


@dataclass(frozen=True)
class AudioData:
    """Audio data for multi-modal content."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None


@dataclass(frozen=True)
class DocumentData:
    """Document data for multi-modal content."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    file_name: str | None = None


@dataclass(frozen=True)
class ToolCallData:
    """Tool call data from assistant messages."""

    id: str
    name: str
    arguments: dict | str
    type: Literal["function", "custom"] = "function"


@dataclass(frozen=True)
class ToolResultData:
    """Tool result data for tool messages."""

    tool_call_id: str
    content: str | dict
    is_error: bool = False
    image_data: bytes | None = None
    image_media_type: str | None = None


@dataclass(frozen=True)
class ThinkingData:
    """Thinking/reasoning content from models."""

    text: str
    signature: str | None = None
    redacted: bool = False


@dataclass(frozen=True)
class ContentPart:
    """A tagged union for different content types in messages."""

    kind: ContentKind
    text: str | None = None
    image: ImageData | None = None
    audio: AudioData | None = None
    document: DocumentData | None = None
    tool_call: ToolCallData | None = None
    tool_result: ToolResultData | None = None
    thinking: ThinkingData | None = None

    @classmethod
    def text(cls, text: str) -> "ContentPart":
        """Create a text content part."""
        return cls(kind=ContentKind.TEXT, text=text)

    @classmethod
    def image(
        cls,
        url: str | None = None,
        data: bytes | None = None,
        media_type: str | None = None,
        detail: Literal["auto", "low", "high"] | None = None,
    ) -> "ContentPart":
        """Create an image content part."""
        return cls(
            kind=ContentKind.IMAGE,
            image=ImageData(url=url, data=data, media_type=media_type, detail=detail),
        )

    @classmethod
    def tool_call(
        cls,
        id: str,
        name: str,
        arguments: dict | str,
        type: Literal["function", "custom"] = "function",
    ) -> "ContentPart":
        """Create a tool call content part."""
        return cls(
            kind=ContentKind.TOOL_CALL,
            tool_call=ToolCallData(id=id, name=name, arguments=arguments, type=type),
        )

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        content: str | dict,
        is_error: bool = False,
    ) -> "ContentPart":
        """Create a tool result content part."""
        return cls(
            kind=ContentKind.TOOL_RESULT,
            tool_result=ToolResultData(
                tool_call_id=tool_call_id, content=content, is_error=is_error
            ),
        )

    @classmethod
    def thinking(cls, text: str, signature: str | None = None) -> "ContentPart":
        """Create a thinking content part."""
        return cls(
            kind=ContentKind.THINKING, thinking=ThinkingData(text=text, signature=signature)
        )

    @classmethod
    def redacted_thinking(cls, text: str, signature: str) -> "ContentPart":
        """Create a redacted thinking content part."""
        return cls(
            kind=ContentKind.REDACTED_THINKING,
            thinking=ThinkingData(text=text, signature=signature, redacted=True),
        )
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_content.py -v
```

Expected: PASS (8 tests passed)

**Step 5: Commit**

```bash
git add src/unified_llm/models/_content.py tests/test_models/test_content.py
git commit -m "feat: define ContentPart and related data classes"
```

---

### Task 4: Define Message, Request, Response, Usage Models

**Files:**
- Create: `src/unified_llm/models/_message.py`
- Create: `src/unified_llm/models/_request.py`
- Create: `src/unified_llm/models/_response.py`
- Create: `tests/test_models/test_message.py`
- Create: `tests/test_models/test_request.py`
- Create: `tests/test_models/test_response.py`

**Step 1: Write the failing test for Message**

```python
# tests/test_models/test_message.py
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
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_message.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.models._message'"

**Step 3: Write minimal implementation for Message**

```python
# src/unified_llm/models/_message.py
"""Message data model for conversations."""

from dataclasses import dataclass, field
from typing import Literal
from unified_llm.models._enums import Role
from unified_llm.models._content import ContentPart


@dataclass(frozen=True)
class Message:
    """A message in a conversation."""

    role: Role
    content: list[ContentPart]
    name: str | None = None
    tool_call_id: str | None = None

    @classmethod
    def system(cls, text: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=[ContentPart.text(text)])

    @classmethod
    def user(cls, text: str) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=[ContentPart.text(text)])

    @classmethod
    def assistant(cls, text: str) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=[ContentPart.text(text)])

    @classmethod
    def tool_result(
        cls, tool_call_id: str, content: str | dict, is_error: bool = False
    ) -> "Message":
        """Create a tool result message."""
        return cls(
            role=Role.TOOL,
            content=[ContentPart.tool_result(tool_call_id, content, is_error)],
            tool_call_id=tool_call_id,
        )

    @property
    def text(self) -> str:
        """Concatenate all text content parts."""
        return "".join(part.text for part in self.content if part.text is not None)
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_message.py -v
```

Expected: PASS (8 tests passed)

**Step 5: Commit**

```bash
git add src/unified_llm/models/_message.py tests/test_models/test_message.py
git commit -m "feat: define Message data model"
```

---

### Task 5: Define Request, Response, Usage, FinishReason Models

**Files:**
- Create: `src/unified_llm/models/_request.py`
- Create: `src/unified_llm/models/_response.py`
- Create: `tests/test_models/test_request.py`
- Create: `tests/test_models/test_response.py`

**Step 1: Write the failing test**

```python
# tests/test_models/test_request.py
import pytest
from unified_llm.models._request import Request, ToolChoice, ResponseFormat


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
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_request.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.models._request'"

**Step 3: Write minimal implementation**

```python
# src/unified_llm/models/_request.py
"""Request data model for LLM API calls."""

from dataclasses import dataclass, field
from typing import Literal, override
from unified_llm.models._message import Message


@dataclass(frozen=True)
class ToolChoice:
    """Controls tool calling behavior."""

    mode: Literal["auto", "none", "required", "named"]
    tool_name: str | None = None

    @classmethod
    def auto(cls) -> "ToolChoice":
        """Let the model decide whether to call tools."""
        return cls(mode="auto")

    @classmethod
    def none(cls) -> "ToolChoice":
        """Prevent the model from calling tools."""
        return cls(mode="none")

    @classmethod
    def required(cls) -> "ToolChoice":
        """Force the model to call at least one tool."""
        return cls(mode="required")

    @classmethod
    def named(cls, tool_name: str) -> "ToolChoice":
        """Force the model to call a specific tool."""
        return cls(mode="named", tool_name=tool_name)


@dataclass(frozen=True)
class ResponseFormat:
    """Controls structured output format."""

    type: Literal["text", "json", "json_schema"]
    json_schema: dict | None = None
    strict: bool = False

    @classmethod
    def text(cls) -> "ResponseFormat":
        """Return plain text (default)."""
        return cls(type="text")

    @classmethod
    def json(cls) -> "ResponseFormat":
        """Return JSON (not schema-validated)."""
        return cls(type="json")

    @classmethod
    def json_schema(cls, schema: dict, strict: bool = False) -> "ResponseFormat":
        """Return JSON matching the schema."""
        return cls(type="json_schema", json_schema=schema, strict=strict)


@dataclass(frozen=True)
class Request:
    """Request for LLM generation."""

    model: str
    messages: list[Message]
    provider: str | None = None
    tools: list | None = None
    tool_choice: ToolChoice | None = None
    response_format: ResponseFormat | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    metadata: dict[str, str] | None = None
    provider_options: dict | None = None
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_request.py -v
```

Expected: PASS (9 tests passed)

**Step 5: Write tests for Response/Usage/FinishReason**

```python
# tests/test_models/test_response.py
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
```

**Step 6: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_response.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.models._response'"

**Step 7: Write minimal implementation**

```python
# src/unified_llm/models/_response.py
"""Response data model for LLM API results."""

from dataclasses import dataclass, field
from datetime import datetime
from unified_llm.models._message import Message
from unified_llm.models._content import ContentPart, ToolCallData


@dataclass(frozen=True)
class Usage:
    """Token usage information."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    raw: dict | None = None

    def __add__(self, other: "Usage") -> "Usage":
        """Add two Usage objects together."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=_sum_optional(self.reasoning_tokens, other.reasoning_tokens),
            cache_read_tokens=_sum_optional(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_sum_optional(
                self.cache_write_tokens, other.cache_write_tokens
            ),
        )


def _sum_optional(a: int | None, b: int | None) -> int | None:
    """Sum two optional integers."""
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


@dataclass(frozen=True)
class FinishReason:
    """Why the generation stopped."""

    reason: str  # "stop", "length", "tool_calls", "content_filter", "error", "other"
    raw: str | None = None  # Provider-specific reason


@dataclass(frozen=True)
class RateLimitInfo:
    """Rate limit information from response headers."""

    requests_remaining: int | None = None
    requests_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_limit: int | None = None
    reset_at: datetime | None = None


@dataclass(frozen=True)
class Warning:
    """Non-fatal warning from the provider."""

    message: str
    code: str | None = None


@dataclass(frozen=True)
class Response:
    """Response from an LLM provider."""

    id: str
    model: str
    provider: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    raw: dict | None = None
    warnings: list[Warning] = field(default_factory=list)
    rate_limit: RateLimitInfo | None = None

    @property
    def text(self) -> str:
        """Concatenate all text content."""
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCallData]:
        """Extract tool calls from the message."""
        return [
            part.tool_call
            for part in self.message.content
            if part.tool_call is not None
        ]

    @property
    def reasoning(self) -> str | None:
        """Extract reasoning/thinking content."""
        thinking_parts = [
            part.thinking
            for part in self.message.content
            if part.thinking is not None
        ]
        if not thinking_parts:
            return None
        return "\n\n".join(t.text for t in thinking_parts)
```

**Step 8: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_response.py -v
```

Expected: PASS (14 tests passed)

**Step 9: Commit**

```bash
git add src/unified_llm/models/_request.py src/unified_llm/models/_response.py tests/test_models/test_request.py tests/test_models/test_response.py
git commit -m "feat: define Request, Response, Usage, and related models"
```

---

### Task 6: Define StreamEvent Models

**Files:**
- Create: `src/unified_llm/models/_stream.py`
- Create: `tests/test_models/test_stream.py`

**Step 1: Write the failing test**

```python
# tests/test_models/test_stream.py
import pytest
from unified_llm.models._stream import StreamEvent, StreamEventType


def test_stream_event_text_delta():
    """Test TEXT_DELTA event."""
    event = StreamEvent.text_delta("Hello, ")
    assert event.type == StreamEventType.TEXT_DELTA
    assert event.delta == "Hello, "


def test_stream_event_text_start():
    """Test TEXT_START event."""
    event = StreamEvent.text_start(text_id="txt_123")
    assert event.type == StreamEventType.TEXT_START
    assert event.text_id == "txt_123"


def test_stream_event_text_end():
    """Test TEXT_END event."""
    event = StreamEvent.text_end(text_id="txt_123")
    assert event.type == StreamEventType.TEXT_END
    assert event.text_id == "txt_123"


def test_stream_event_reasoning_delta():
    """Test REASONING_DELTA event."""
    event = StreamEvent.reasoning_delta("Let me think...")
    assert event.type == StreamEventType.REASONING_DELTA
    assert event.reasoning_delta == "Let me think..."


def test_stream_event_tool_call_start():
    """Test TOOL_CALL_START event."""
    event = StreamEvent.tool_call_start(
        tool_call_id="call_123",
        tool_name="get_weather",
    )
    assert event.type == StreamEventType.TOOL_CALL_START
    assert event.tool_call is not None
    assert event.tool_call.id == "call_123"


def test_stream_event_finish():
    """Test FINISH event."""
    from unified_llm.models._response import Usage, FinishReason

    event = StreamEvent.finish(
        finish_reason=FinishReason(reason="stop"),
        usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
    )
    assert event.type == StreamEventType.FINISH
    assert event.usage.total_tokens == 30


def test_stream_event_error():
    """Test ERROR event."""
    from unified_llm._exceptions import SDKError

    event = StreamEvent.error(SDKError("Something went wrong"))
    assert event.type == StreamEventType.ERROR
    assert event.error is not None
    assert event.error.message == "Something went wrong"
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_stream.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.models._stream'"

**Step 3: Write minimal implementation**

```python
# src/unified_llm/models/_stream.py
"""Stream event data models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from unified_llm.models._response import Usage, FinishReason, Response
from unified_llm.models._content import ToolCallData


class StreamEventType(str, Enum):
    """Types of stream events."""

    STREAM_START = "stream_start"
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"
    REASONING_START = "reasoning_start"
    REASONING_DELTA = "reasoning_delta"
    REASONING_END = "reasoning_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    STEP_FINISH = "step_finish"
    FINISH = "finish"
    ERROR = "error"
    PROVIDER_EVENT = "provider_event"


@dataclass(frozen=True)
class StreamEvent:
    """A streaming event from the LLM provider."""

    type: StreamEventType | str

    # Text events
    delta: str | None = None
    text_id: str | None = None

    # Reasoning events
    reasoning_delta: str | None = None

    # Tool call events
    tool_call: ToolCallData | None = None

    # Finish events
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    response: Response | None = None

    # Error events
    error: Exception | None = None

    # Provider passthrough
    raw: dict | None = None

    @classmethod
    def text_start(cls, text_id: str) -> "StreamEvent":
        """Create a TEXT_START event."""
        return cls(type=StreamEventType.TEXT_START, text_id=text_id)

    @classmethod
    def text_delta(cls, delta: str) -> "StreamEvent":
        """Create a TEXT_DELTA event."""
        return cls(type=StreamEventType.TEXT_DELTA, delta=delta)

    @classmethod
    def text_end(cls, text_id: str) -> "StreamEvent":
        """Create a TEXT_END event."""
        return cls(type=StreamEventType.TEXT_END, text_id=text_id)

    @classmethod
    def reasoning_delta(cls, delta: str) -> "StreamEvent":
        """Create a REASONING_DELTA event."""
        return cls(type=StreamEventType.REASONING_DELTA, reasoning_delta=delta)

    @classmethod
    def reasoning_start(cls) -> "StreamEvent":
        """Create a REASONING_START event."""
        return cls(type=StreamEventType.REASONING_START)

    @classmethod
    def reasoning_end(cls) -> "StreamEvent":
        """Create a REASONING_END event."""
        return cls(type=StreamEventType.REASONING_END)

    @classmethod
    def tool_call_start(cls, tool_call_id: str, tool_name: str) -> "StreamEvent":
        """Create a TOOL_CALL_START event."""
        return cls(
            type=StreamEventType.TOOL_CALL_START,
            tool_call=ToolCallData(id=tool_call_id, name=tool_name, arguments={}),
        )

    @classmethod
    def tool_call_delta(cls, tool_call_id: str, delta_args: str) -> "StreamEvent":
        """Create a TOOL_CALL_DELTA event."""
        return cls(
            type=StreamEventType.TOOL_CALL_DELTA,
            tool_call=ToolCallData(id=tool_call_id, name="", arguments=delta_args),
        )

    @classmethod
    def tool_call_end(cls, tool_call: ToolCallData) -> "StreamEvent":
        """Create a TOOL_CALL_END event."""
        return cls(type=StreamEventType.TOOL_CALL_END, tool_call=tool_call)

    @classmethod
    def finish(
        cls, finish_reason: FinishReason, usage: Usage, response: Response | None = None
    ) -> "StreamEvent":
        """Create a FINISH event."""
        return cls(
            type=StreamEventType.FINISH,
            finish_reason=finish_reason,
            usage=usage,
            response=response,
        )

    @classmethod
    def error(cls, error: Exception) -> "StreamEvent":
        """Create an ERROR event."""
        return cls(type=StreamEventType.ERROR, error=error)

    @classmethod
    def stream_start(cls) -> "StreamEvent":
        """Create a STREAM_START event."""
        return cls(type=StreamEventType.STREAM_START)

    @classmethod
    def step_finish(cls) -> "StreamEvent":
        """Create a STEP_FINISH event."""
        return cls(type=StreamEventType.STEP_FINISH)
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_models/test_stream.py -v
```

Expected: PASS (8 tests passed)

**Step 5: Commit**

```bash
git add src/unified_llm/models/_stream.py tests/test_models/test_stream.py
git commit -m "feat: define StreamEvent and StreamEventType models"
```

---

### Task 7: Create Models Package Exports

**Files:**
- Create: `src/unified_llm/models/__init__.py`

**Step 1: Create the models package init**

```python
# src/unified_llm/models/__init__.py
"""Data models for the unified LLM client."""

# Enums
from unified_llm.models._enums import Role, ContentKind

# Content parts
from unified_llm.models._content import (
    ContentPart,
    ImageData,
    AudioData,
    DocumentData,
    ToolCallData,
    ToolResultData,
    ThinkingData,
)

# Messages
from unified_llm.models._message import Message

# Requests
from unified_llm.models._request import Request, ToolChoice, ResponseFormat

# Responses
from unified_llm.models._response import (
    Response,
    Usage,
    FinishReason,
    RateLimitInfo,
    Warning,
)

# Streaming
from unified_llm.models._stream import StreamEvent, StreamEventType

__all__ = [
    # Enums
    "Role",
    "ContentKind",
    # Content
    "ContentPart",
    "ImageData",
    "AudioData",
    "DocumentData",
    "ToolCallData",
    "ToolResultData",
    "ThinkingData",
    # Messages
    "Message",
    # Requests
    "Request",
    "ToolChoice",
    "ResponseFormat",
    # Responses
    "Response",
    "Usage",
    "FinishReason",
    "RateLimitInfo",
    "Warning",
    # Streaming
    "StreamEvent",
    "StreamEventType",
]
```

**Step 2: Verify imports work**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -c "from unified_llm.models import Message, Request, Response; print('Imports OK')"
```

Expected: "Imports OK"

**Step 3: Commit**

```bash
git add src/unified_llm/models/__init__.py
git commit -m "feat: create models package with exports"
```

---

## Layer 1: Exception Hierarchy

### Task 8: Define Exception Classes

**Files:**
- Modify: `src/unified_llm/_exceptions.py`
- Create: `tests/test_exceptions.py`

**Step 1: Write the failing test**

```python
# tests/test_exceptions.py
import pytest
from unified_llm._exceptions import (
    SDKError,
    ProviderError,
    AuthenticationError,
    AccessDeniedError,
    NotFoundError,
    InvalidRequestError,
    RateLimitError,
    ServerError,
    ContentFilterError,
    ContextLengthError,
    QuotaExceededError,
    RequestTimeoutError,
    AbortError,
    NetworkError,
    StreamError,
    InvalidToolCallError,
    NoObjectGeneratedError,
    ConfigurationError,
)


def test_sdk_error():
    """Test base SDKError."""
    err = SDKError("Something went wrong")
    assert err.message == "Something went wrong"
    assert err.cause is None


def test_sdk_error_with_cause():
    """Test SDKError with cause."""
    original = ValueError("Original error")
    err = SDKError("Wrapped error", cause=original)
    assert err.cause is original


def test_provider_error():
    """Test ProviderError with all fields."""
    err = ProviderError(
        provider="openai",
        message="Rate limit exceeded",
        status_code=429,
        error_code="rate_limit_exceeded",
        retryable=True,
        retry_after=1.0,
    )
    assert err.provider == "openai"
    assert err.status_code == 429
    assert err.retryable is True
    assert err.retry_after == 1.0


def test_authentication_error_not_retryable():
    """Test AuthenticationError is not retryable."""
    err = AuthenticationError("Invalid API key", provider="anthropic")
    assert err.retryable is False


def test_rate_limit_error_is_retryable():
    """Test RateLimitError is retryable."""
    err = RateLimitError("Rate limited", provider="openai", retry_after=2.0)
    assert err.retryable is True
    assert err.retry_after == 2.0


def test_server_error_is_retryable():
    """Test ServerError is retryable."""
    err = ServerError("Internal server error", provider="gemini", status_code=500)
    assert err.retryable is True


def test_configuration_error_not_retryable():
    """Test ConfigurationError is not retryable."""
    err = ConfigurationError("No provider configured")
    assert err.retryable is False


def test_inheritance_hierarchy():
    """Test all errors inherit from SDKError."""
    errors = [
        AuthenticationError("", provider=""),
        RateLimitError("", provider=""),
        InvalidRequestError("", provider=""),
        ConfigurationError(""),
        NoObjectGeneratedError(""),
    ]
    for err in errors:
        assert isinstance(err, SDKError)
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_exceptions.py -v
```

Expected: FAIL with current empty or incomplete _exceptions.py

**Step 3: Write minimal implementation**

```python
# src/unified_llm/_exceptions.py
"""Exception hierarchy for the unified LLM client."""


class SDKError(Exception):
    """Base exception for all SDK errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


class ProviderError(SDKError):
    """Error from an LLM provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        raw: dict | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw = raw


class AuthenticationError(ProviderError):
    """401: Invalid API key or expired token."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class AccessDeniedError(ProviderError):
    """403: Insufficient permissions."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class NotFoundError(ProviderError):
    """404: Model or endpoint not found."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class InvalidRequestError(ProviderError):
    """400/422: Malformed request or invalid parameters."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class RateLimitError(ProviderError):
    """429: Rate limit exceeded."""

    def __init__(self, message: str, *, provider: str, retry_after: float | None = None, **kwargs):
        super().__init__(message, provider=provider, retryable=True, retry_after=retry_after, **kwargs)


class ServerError(ProviderError):
    """500-599: Provider internal error."""

    def __init__(self, message: str, *, provider: str, status_code: int, **kwargs):
        super().__init__(message, provider=provider, status_code=status_code, retryable=True, **kwargs)


class ContentFilterError(ProviderError):
    """Response blocked by safety/content filter."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class ContextLengthError(ProviderError):
    """413: Input + output exceeds context window."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class QuotaExceededError(ProviderError):
    """Billing/quota limit exceeded."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


# Non-provider errors


class RequestTimeoutError(SDKError):
    """Request or stream timeout."""

    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = True


class AbortError(SDKError):
    """Request cancelled via abort signal."""

    def __init__(self, message: str = "Request was aborted"):
        super().__init__(message)
        self.retryable = False


class NetworkError(SDKError):
    """Network-level failure."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, cause=cause)
        self.retryable = True


class StreamError(SDKError):
    """Error during stream consumption."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, cause=cause)
        self.retryable = True


class InvalidToolCallError(SDKError):
    """Tool call parameter validation failed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = False


class NoObjectGeneratedError(SDKError):
    """Structured output parsing/validation failed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = False


class ConfigurationError(SDKError):
    """SDK configuration error."""

    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = False
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_exceptions.py -v
```

Expected: PASS (12 tests passed)

**Step 5: Commit**

```bash
git add src/unified_llm/_exceptions.py tests/test_exceptions.py
git commit -m "feat: define complete exception hierarchy"
```

---

## Layer 1: Provider Adapter Interface

### Task 9: Define ProviderAdapter Interface

**Files:**
- Create: `src/unified_llm/providers/_base.py`
- Create: `tests/test_providers/test_base.py`

**Step 1: Write the failing test**

```python
# tests/test_providers/test_base.py
import pytest
from unified_llm.providers._base import ProviderAdapter
from unified_llm.models import Request


def test_provider_adapter_is_abstract():
    """Test ProviderAdapter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ProviderAdapter()


def test_provider_adapter_requires_complete():
    """Test ProviderAdapter requires complete() implementation."""
    class MockAdapter(ProviderAdapter):
        name = "mock"

        async def stream(self, request: Request):
            async def gen():
                yield
            return gen()

    with pytest.raises(TypeError):
        MockAdapter().complete(Request(model="test", messages=[]))


def test_provider_adapter_requires_stream():
    """Test ProviderAdapter requires stream() implementation."""
    class MockAdapter(ProviderAdapter):
        name = "mock"

        async def complete(self, request: Request):
            from unified_llm.models import Response, Usage, FinishReason, Message
            return Response(
                id="test",
                model="test",
                provider="mock",
                message=Message.assistant(""),
                finish_reason=FinishReason(reason="stop"),
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

    with pytest.raises(TypeError):
        MockAdapter().stream(Request(model="test", messages=[]))
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_providers/test_base.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.providers._base'"

**Step 3: Write minimal implementation**

```python
# src/unified_llm/providers/_base.py
"""Base provider adapter interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from unified_llm.models import Request, Response, StreamEvent


class ProviderAdapter(ABC):
    """Abstract base for provider adapters.

    Each provider (OpenAI, Anthropic, Gemini) implements this interface.
    """

    #: Provider name (e.g., "openai", "anthropic", "gemini")
    name: str

    @abstractmethod
    async def complete(self, request: Request) -> Response:
        """Send request, block until complete, return full response.

        Args:
            request: The unified request object.

        Returns:
            Response from the provider.

        Raises:
            ProviderError: On provider errors.
        """
        raise NotImplementedError

    @abstractmethod
    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send request, return async iterator of stream events.

        Args:
            request: The unified request object.

        Yields:
            StreamEvent objects as they arrive.

        Raises:
            ProviderError: On provider errors.
        """
        raise NotImplementedError

    async def initialize(self) -> None:
        """Validate configuration at startup.

        Called when adapter is registered with Client.
        Raise ConfigurationError if configuration is invalid.
        """
        pass

    async def close(self) -> None:
        """Release resources (HTTP connections, etc.).

        Called by Client.close().
        """
        pass

    def supports_tool_choice(self, mode: str) -> bool:
        """Check if a specific tool choice mode is supported.

        Args:
            mode: "auto", "none", "required", "named"

        Returns:
            True if the mode is supported.
        """
        return mode in ("auto", "none", "required", "named")
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_providers/test_base.py -v
```

Expected: PASS (3 tests passed)

**Step 5: Create providers package init**

```python
# src/unified_llm/providers/__init__.py
"""Provider adapters for LLM providers."""

from unified_llm.providers._base import ProviderAdapter

__all__ = [
    "ProviderAdapter",
]
```

**Step 6: Commit**

```bash
git add src/unified_llm/providers/_base.py src/unified_llm/providers/__init__.py tests/test_providers/test_base.py
git commit -m "feat: define ProviderAdapter interface"
```

---

## Layer 2: Provider Utilities (HTTP Client, SSE Parser)

### Task 10: Create HTTP Client Utility

**Files:**
- Create: `src/unified_llm/utils/_http.py`
- Create: `tests/test_utils/test_http.py`

**Step 1: Write the failing test**

```python
# tests/test_utils/test_http.py
import pytest
from unified_llm.utils._http import HttpClient, HttpMethod


@pytest.mark.asyncio
async def test_http_client_get_request():
    """Test HTTP client can make GET requests."""
    client = HttpClient()
    response = await client.request(
        method=HttpMethod.GET,
        url="https://httpbin.org/get",
        headers={"User-Agent": "test"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "headers" in data


@pytest.mark.asyncio
async def test_http_client_post_request():
    """Test HTTP client can make POST requests."""
    client = HttpClient()
    response = await client.request(
        method=HttpMethod.POST,
        url="https://httpbin.org/post",
        json={"test": "data"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["json"] == {"test": "data"}


@pytest.mark.asyncio
async def test_http_client_streaming():
    """Test HTTP client can handle streaming responses."""
    client = HttpClient()
    chunks = []
    async for chunk in client.stream(
        method=HttpMethod.GET,
        url="https://httpbin.org/stream/3",
    ):
        chunks.append(chunk)

    assert len(chunks) == 3


@pytest.mark.asyncio
async def test_http_client_timeout():
    """Test HTTP client respects timeout."""
    import pytest
    from unified_llm._exceptions import RequestTimeoutError

    client = HttpClient(timeout=0.001)  # 1ms timeout
    with pytest.raises(RequestTimeoutError):
        await client.request(
            method=HttpMethod.GET,
            url="https://httpbin.org/delay/1",
        )
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_utils/test_http.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.utils._http'"

**Step 3: Write minimal implementation**

```python
# src/unified_llm/utils/_http.py
"""HTTP client utility for provider adapters."""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Any

import httpx

from unified_llm._exceptions import (
    RequestTimeoutError,
    NetworkError,
)


class HttpMethod(str, Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass(frozen=True)
class HttpResponse:
    """HTTP response wrapper."""

    status_code: int
    headers: dict[str, str]
    content: bytes
    _client_response: httpx.Response | None = None

    def text(self) -> str:
        """Get response text."""
        return self.content.decode("utf-8")

    def json(self) -> Any:
        """Parse response as JSON."""
        import json
        return json.loads(self.content)


class HttpClient:
    """Async HTTP client wrapper around httpx."""

    def __init__(
        self,
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
        limits: httpx.Limits | None = None,
    ):
        """Initialize HTTP client.

        Args:
            timeout: Request timeout in seconds.
            connect_timeout: Connection timeout in seconds.
            limits: Connection pool limits.
        """
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=timeout,
            write=timeout,
            pool=timeout,
        )
        self._limits = limits or httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        )
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            limits=self._limits,
            http2=True,  # Enable HTTP/2
        )

    async def request(
        self,
        method: HttpMethod,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        content: bytes | None = None,
    ) -> HttpResponse:
        """Make an HTTP request.

        Args:
            method: HTTP method.
            url: Request URL.
            headers: Request headers.
            params: Query parameters.
            json: JSON body.
            content: Raw body bytes.

        Returns:
            HttpResponse wrapper.

        Raises:
            NetworkError: On network errors.
            RequestTimeoutError: On timeout.
        """
        try:
            response = await self._client.request(
                method=method.value,
                url=url,
                headers=headers,
                params=params,
                json=json,
                content=content,
            )
            return HttpResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=response.content,
                _client_response=response,
            )

        except httpx.TimeoutException as e:
            raise RequestTimeoutError(f"Request timeout: {e}") from e

        except httpx.NetworkError as e:
            raise NetworkError(f"Network error: {e}", cause=e) from e

    async def stream(
        self,
        method: HttpMethod,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream an HTTP response.

        Args:
            method: HTTP method.
            url: Request URL.
            headers: Request headers.
            params: Query parameters.
            json: JSON body.

        Yields:
            Response chunks as bytes.

        Raises:
            NetworkError: On network errors.
            RequestTimeoutError: On timeout.
        """
        try:
            async with self._client.stream(
                method=method.value,
                url=url,
                headers=headers,
                params=params,
                json=json,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

        except httpx.TimeoutException as e:
            raise RequestTimeoutError(f"Stream timeout: {e}") from e

        except httpx.NetworkError as e:
            raise NetworkError(f"Network error: {e}", cause=e) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_utils/test_http.py -v
```

Expected: PASS (4 tests passed)

**Step 5: Create utils package init**

```python
# src/unified_llm/utils/__init__.py
"""Utility functions for provider adapters."""

from unified_llm.utils._http import HttpClient, HttpMethod, HttpResponse

__all__ = [
    "HttpClient",
    "HttpMethod",
    "HttpResponse",
]
```

**Step 6: Commit**

```bash
git add src/unified_llm/utils/_http.py src/unified_llm/utils/__init__.py tests/test_utils/test_http.py
git commit -m "feat: add HttpClient utility"
```

---

### Task 11: Create SSE Parser

**Files:**
- Create: `src/unified_llm/utils/_sse.py`
- Create: `tests/test_utils/test_sse.py`

**Step 1: Write the failing test**

```python
# tests/test_utils/test_sse.py
import pytest
from unified_llm.utils._sse import parse_sse


@pytest.mark.asyncio
async def test_parse_sse_basic_events():
    """Test parsing basic SSE events."""
    data = b"event: message\ndata: Hello\n\nevent: done\ndata: World\n\n"
    events = []
    async for event_type, data in parse_sse(data):
        events.append((event_type, data))

    assert events == [
        ("message", "Hello"),
        ("done", "World"),
    ]


@pytest.mark.asyncio
async def test_parse_sse_multiline_data():
    """Test parsing SSE with multi-line data."""
    data = b"data: line1\ndata: line2\n\n"
    events = []
    async for event_type, data in parse_sse(data):
        events.append((event_type, data))

    assert len(events) == 1
    assert events[0][1] == "line1\nline2"


@pytest.mark.asyncio
async def test_parse_sse_with_retry():
    """Test parsing SSE with retry field."""
    data = b"retry: 1000\n\ndata: test\n\n"
    events = []
    async for event_type, data in parse_sse(data):
        events.append((event_type, data))

    assert len(events) == 1
    assert events[0] == ("message", "test")  # Default event type


@pytest.mark.asyncio
async def test_parse_sse_ignores_comments():
    """Test that comment lines are ignored."""
    data = b": this is a comment\ndata: actual data\n\n"
    events = []
    async for event_type, data in parse_sse(data):
        events.append((event_type, data))

    assert len(events) == 1
    assert events[0][1] == "actual data"


@pytest.mark.asyncio
async def test_parse_sse_from_async_iterator():
    """Test parsing SSE from async iterator."""
    async def chunks():
        yield b"event: start\n"
        yield b"data: part1\n"
        yield b"data: part2\n\n"
        yield b"event: end\n"
        yield b"data: final\n\n"

    events = []
    async for event_type, data in parse_sse(chunks()):
        events.append((event_type, data))

    assert events == [
        ("start", "part1\npart2"),
        ("end", "final"),
    ]
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_utils/test_sse.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.utils._sse'"

**Step 3: Write minimal implementation**

```python
# src/unified_llm/utils/_sse.py
"""Server-Sent Events (SSE) parser."""

from typing import AsyncIterator, Tuple


async def parse_sse(
    source: bytes | AsyncIterator[bytes],
) -> AsyncIterator[Tuple[str, str]]:
    """Parse Server-Sent Events.

    Args:
        source: Either bytes or an async iterator of bytes chunks.

    Yields:
        Tuples of (event_type, data). Event type defaults to "message"
        if not specified.

    Example:
        >>> async for event_type, data in parse_sse(response):
        ...     print(f"{event_type}: {data}")
    """
    buffer = b""
    event_type = "message"
    data_lines = []

    if isinstance(source, bytes):
        chunks = _iter_bytes([source])
    else:
        chunks = source

    async for chunk in chunks:
        buffer += chunk

        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line_str = line.decode("utf-8").rstrip("\r")

            # Empty line = end of event
            if not line_str:
                if data_lines:
                    yield (event_type, "\n".join(data_lines))
                    event_type = "message"
                    data_lines = []
                continue

            # Comment line
            if line_str.startswith(":"):
                continue

            # Parse field
            if ":" in line_str:
                field, value = line_str.split(":", 1)
                value = value.lstrip()

                if field == "event":
                    event_type = value
                elif field == "data":
                    data_lines.append(value)
                elif field == "retry":
                    # Could be used to set reconnection delay
                    pass
            else:
                # Treat line as data field with no value
                data_lines.append(line_str)

    # Emit any remaining event
    if data_lines:
        yield (event_type, "\n".join(data_lines))


async def _iter_bytes(chunks: list[bytes]) -> AsyncIterator[bytes]:
    """Convert list of bytes to async iterator."""
    for chunk in chunks:
        yield chunk
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_utils/test_sse.py -v
```

Expected: PASS (5 tests passed)

**Step 5: Update utils package exports**

```python
# src/unified_llm/utils/__init__.py
"""Utility functions for provider adapters."""

from unified_llm.utils._http import HttpClient, HttpMethod, HttpResponse
from unified_llm.utils._sse import parse_sse

__all__ = [
    "HttpClient",
    "HttpMethod",
    "HttpResponse",
    "parse_sse",
]
```

**Step 6: Commit**

```bash
git add src/unified_llm/utils/_sse.py src/unified_llm/utils/__init__.py tests/test_utils/test_sse.py
git commit -m "feat: add SSE parser utility"
```

---

### Task 12: Create Retry Utility

**Files:**
- Create: `src/unified_llm/utils/_retry.py`
- Create: `tests/test_utils/test_retry.py`

**Step 1: Write the failing test**

```python
# tests/test_utils/test_retry.py
import pytest
from unified_llm.utils._retry import RetryPolicy, retry
from unified_llm._exceptions import RateLimitError, ServerError, AuthenticationError
import asyncio


@pytest.mark.asyncio
async def test_retry_success_on_second_attempt():
    """Test retry succeeds after transient failure."""
    attempts = []

    async def failing_func():
        attempts.append(1)
        if len(attempts) < 2:
            raise ServerError("Temporary error", provider="test", status_code=503)
        return "success"

    result = await retry(failing_func, policy=RetryPolicy(max_retries=3))
    assert result == "success"
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_retry_exhausted():
    """Test retry gives up after max attempts."""
    attempts = 0

    async def always_failing_func():
        nonlocal attempts
        attempts += 1
        raise ServerError("Always failing", provider="test", status_code=500)

    with pytest.raises(ServerError):
        await retry(always_failing_func, policy=RetryPolicy(max_retries=2))

    assert attempts == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_retry_no_retry_for_non_retryable():
    """Test non-retryable errors fail immediately."""
    attempts = 0

    async def auth_error_func():
        nonlocal attempts
        attempts += 1
        raise AuthenticationError("Bad auth", provider="test")

    with pytest.raises(AuthenticationError):
        await retry(auth_error_func, policy=RetryPolicy(max_retries=3))

    assert attempts == 1  # No retries for non-retryable errors


@pytest.mark.asyncio
async def test_retry_with_backoff():
    """Test exponential backoff timing."""
    import time

    attempt_times = []

    async def failing_func():
        attempt_times.append(time.time())
        raise RateLimitError("Rate limited", provider="test")

    policy = RetryPolicy(max_retries=2, base_delay=0.1, backoff_multiplier=2.0, jitter=False)

    with pytest.raises(RateLimitError):
        await retry(failing_func, policy=policy)

    # Check delays: first retry ~0.1s, second retry ~0.2s
    assert len(attempt_times) == 3
    delay1 = attempt_times[1] - attempt_times[0]
    delay2 = attempt_times[2] - attempt_times[1]
    assert 0.08 < delay1 < 0.15  # ~0.1s
    assert 0.18 < delay2 < 0.25  # ~0.2s


@pytest.mark.asyncio
async def test_retry_with_retry_after():
    """Test Retry-After header overrides backoff."""
    import time

    attempt_times = []

    async def rate_limited_func():
        attempt_times.append(time.time())
        # Provider asked to wait 0.2 seconds
        raise RateLimitError("Rate limited", provider="test", retry_after=0.2)

    policy = RetryPolicy(max_retries=1, base_delay=1.0, jitter=False)

    with pytest.raises(RateLimitError):
        await retry(rate_limited_func, policy=policy)

    # Should use retry_after (0.2s) not base_delay (1.0s)
    delay = attempt_times[1] - attempt_times[0]
    assert 0.18 < delay < 0.25  # ~0.2s


@pytest.mark.asyncio
async def test_retry_callback():
    """Test on_retry callback is called."""
    callbacks = []

    async def record_callback(error, attempt, delay):
        callbacks.append((attempt, delay))

    async def failing_func():
        raise ServerError("Error", provider="test", status_code=500)

    policy = RetryPolicy(max_retries=2, base_delay=0.1, jitter=False, on_retry=record_callback)

    with pytest.raises(ServerError):
        await retry(failing_func, policy=policy)

    assert len(callbacks) == 2
    assert callbacks[0][0] == 1  # First retry
    assert callbacks[1][0] == 2  # Second retry
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_utils/test_retry.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.utils._retry'"

**Step 3: Write minimal implementation**

```python
# src/unified_llm/utils/_retry.py
"""Retry logic with exponential backoff."""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from unified_llm._exceptions import SDKError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy configuration."""

    max_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    on_retry: Callable[[SDKError, int, float], Awaitable[None]] | None = None


async def retry(
    func: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy = RetryPolicy(),
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry.
        policy: Retry policy configuration.

    Returns:
        Result from successful function call.

    Raises:
        The last error if all retries exhausted.
    """
    last_error: SDKError | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await func()

        except SDKError as e:
            last_error = e

            # Check if error is retryable
            retryable = getattr(e, "retryable", False)
            if not retryable:
                raise

            # Check if we should retry
            if attempt >= policy.max_retries:
                raise

            # Calculate delay
            retry_after = getattr(e, "retry_after", None)
            if retry_after is not None:
                # Provider specified retry delay
                if retry_after > policy.max_delay:
                    # Provider asked to wait too long
                    raise
                delay = retry_after
            else:
                # Exponential backoff
                delay = min(
                    policy.base_delay * (policy.backoff_multiplier ** attempt),
                    policy.max_delay,
                )

                # Add jitter
                if policy.jitter:
                    delay = delay * random.uniform(0.5, 1.5)

            # Call callback if provided
            if policy.on_retry:
                await policy.on_retry(e, attempt + 1, delay)

            # Wait before retry
            await asyncio.sleep(delay)

    # Should never reach here, but for type safety
    raise last_error or SDKError("Retry failed with no error")
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_utils/test_retry.py -v
```

Expected: PASS (7 tests passed)

**Step 5: Update utils package exports**

```python
# src/unified_llm/utils/__init__.py
"""Utility functions for provider adapters."""

from unified_llm.utils._http import HttpClient, HttpMethod, HttpResponse
from unified_llm.utils._sse import parse_sse
from unified_llm.utils._retry import RetryPolicy, retry

__all__ = [
    "HttpClient",
    "HttpMethod",
    "HttpResponse",
    "parse_sse",
    "RetryPolicy",
    "retry",
]
```

**Step 6: Commit**

```bash
git add src/unified_llm/utils/_retry.py src/unified_llm/utils/__init__.py tests/test_utils/test_retry.py
git commit -m "feat: add retry utility with exponential backoff"
```

---

## Layer 3: Core Client

### Task 13: Implement Core Client with Provider Routing

**Files:**
- Create: `src/unified_llm/client.py`
- Create: `tests/test_client.py`

**Step 1: Write the failing test**

```python
# tests/test_client.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from unified_llm.client import Client
from unified_llm.models import Request, Response, Usage, FinishReason, Message, StreamEvent
from unified_llm.providers._base import ProviderAdapter


class MockAdapter(ProviderAdapter):
    """Mock provider adapter for testing."""

    name = "mock"

    def __init__(self):
        self.complete_called = False
        self.stream_called = False

    async def complete(self, request: Request) -> Response:
        self.complete_called = True
        return Response(
            id="test-123",
            model=request.model,
            provider=self.name,
            message=Message.assistant("Mock response"),
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    async def stream(self, request: Request):
        self.stream_called = True
        yield StreamEvent.stream_start()
        yield StreamEvent.text_delta("Mock")
        yield StreamEvent.text_delta(" response")
        yield StreamEvent.finish(
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=10, output_tokens=2, total_tokens=12),
        )


def test_client_construction_with_providers():
    """Test Client can be constructed with explicit providers."""
    adapter = MockAdapter()
    client = Client(providers={"mock": adapter}, default_provider="mock")

    assert client.default_provider == "mock"


def test_client_from_env():
    """Test Client can be constructed from environment."""
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    client = Client.from_env()

    # Should have registered Anthropic provider
    assert "anthropic" in client._providers


def test_client_complete_routes_to_provider():
    """Test complete() routes to correct provider."""
    import asyncio

    adapter = MockAdapter()
    client = Client(providers={"mock": adapter}, default_provider="mock")

    async def test():
        request = Request(model="test-model", messages=[Message.user("Hello")])
        response = await client.complete(request)

        assert adapter.complete_called is True
        assert response.text == "Mock response"

    asyncio.run(test())


def test_client_complete_explicit_provider():
    """Test complete() with explicit provider parameter."""
    import asyncio

    adapter1 = MockAdapter()
    adapter1.name = "mock1"
    adapter2 = MockAdapter()
    adapter2.name = "mock2"

    client = Client(providers={"mock1": adapter1, "mock2": adapter2}, default_provider="mock1")

    async def test():
        request = Request(model="test-model", messages=[Message.user("Hello")], provider="mock2")
        await client.complete(request)

        assert adapter2.complete_called is True
        assert adapter1.complete_called is False

    asyncio.run(test())


def test_client_stream_routes_to_provider():
    """Test stream() routes to correct provider."""
    import asyncio

    adapter = MockAdapter()
    client = Client(providers={"mock": adapter}, default_provider="mock")

    async def test():
        request = Request(model="test-model", messages=[Message.user("Hello")])
        events = []
        async for event in client.stream(request):
            events.append(event)

        assert adapter.stream_called is True
        assert len(events) == 4
        assert events[1].delta == "Mock"

    asyncio.run(test())


def test_client_no_provider_error():
    """Test error when no provider configured."""
    import asyncio

    client = Client(providers={}, default_provider=None)

    async def test():
        request = Request(model="test-model", messages=[Message.user("Hello")])
        with pytest.raises(Exception):  # ConfigurationError
            await client.complete(request)

    asyncio.run(test())


def test_client_unknown_provider_error():
    """Test error when requesting unknown provider."""
    import asyncio

    adapter = MockAdapter()
    client = Client(providers={"mock": adapter}, default_provider="mock")

    async def test():
        request = Request(model="test-model", messages=[Message.user("Hello")], provider="unknown")
        with pytest.raises(Exception):  # ConfigurationError
            await client.complete(request)

    asyncio.run(test())


def test_client_close():
    """Test Client.close() calls close on all providers."""
    import asyncio

    adapter = MockAdapter()
    adapter.close = AsyncMock()

    client = Client(providers={"mock": adapter}, default_provider="mock")

    async def test():
        await client.close()
        adapter.close.assert_called_once()

    asyncio.run(test())
```

**Step 2: Run test to verify it fails**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_client.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'unified_llm.client'"

**Step 3: Write minimal implementation**

```python
# src/unified_llm/client.py
"""Core client with provider routing and middleware."""

import os
from typing import AsyncIterator, Callable

from unified_llm._exceptions import ConfigurationError
from unified_llm.models import Request, Response, StreamEvent
from unified_llm.providers._base import ProviderAdapter


MiddlewareFn = Callable[[Request, "NextFn"], Callable[[], Response]]
NextFn = Callable[[Request], Response]


class Client:
    """Core LLM client with provider routing.

    The Client manages provider adapters, routes requests to the correct
    provider, and applies middleware.
    """

    def __init__(
        self,
        *,
        providers: dict[str, ProviderAdapter],
        default_provider: str | None = None,
        middleware: list | None = None,
    ):
        """Initialize Client.

        Args:
            providers: Mapping of provider names to adapters.
            default_provider: Default provider when not specified in request.
            middleware: List of middleware functions.
        """
        self._providers = providers
        self._default_provider = default_provider
        self._middleware = middleware or []

        if default_provider and default_provider not in providers:
            raise ConfigurationError(
                f"Default provider '{default_provider}' not found in registered providers"
            )

    @classmethod
    def from_env(cls) -> "Client":
        """Create Client from environment variables.

        Checks for ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY.

        Returns:
            Configured Client instance.
        """
        providers = {}
        default_provider = None

        # Import adapters here to avoid circular imports
        from unified_llm.providers import AnthropicAdapter, OpenAIAdapter, GeminiAdapter

        # Anthropic
        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            providers["anthropic"] = AnthropicAdapter(api_key=api_key)
            if default_provider is None:
                default_provider = "anthropic"

        # OpenAI
        if api_key := os.environ.get("OPENAI_API_KEY"):
            providers["openai"] = OpenAIAdapter(api_key=api_key)
            if default_provider is None:
                default_provider = "openai"

        # Gemini
        if api_key := os.environ.get("GEMINI_API_KEY"):
            providers["gemini"] = GeminiAdapter(api_key=api_key)
            if default_provider is None:
                default_provider = "gemini"

        return cls(providers=providers, default_provider=default_provider)

    async def complete(self, request: Request) -> Response:
        """Complete a request (blocking, non-streaming).

        Args:
            request: Unified request object.

        Returns:
            Response from the provider.

        Raises:
            ConfigurationError: If no provider is configured.
            ProviderError: On provider errors.
        """
        adapter = self._get_adapter(request)
        return await adapter.complete(request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Stream a request.

        Args:
            request: Unified request object.

        Yields:
            StreamEvent objects as they arrive.

        Raises:
            ConfigurationError: If no provider is configured.
            ProviderError: On provider errors.
        """
        adapter = self._get_adapter(request)
        async for event in adapter.stream(request):
            yield event

    def _get_adapter(self, request: Request) -> ProviderAdapter:
        """Get the adapter for a request.

        Args:
            request: Request with optional provider field.

        Returns:
            Provider adapter.

        Raises:
            ConfigurationError: If provider not found.
        """
        provider_name = request.provider or self._default_provider

        if provider_name is None:
            raise ConfigurationError(
                "No provider specified and no default provider configured"
            )

        if provider_name not in self._providers:
            raise ConfigurationError(
                f"Provider '{provider_name}' not registered. "
                f"Available providers: {list(self._providers.keys())}"
            )

        return self._providers[provider_name]

    async def close(self) -> None:
        """Close all provider adapters."""
        for adapter in self._providers.values():
            await adapter.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()
```

**Step 4: Run test to verify it passes**

```bash
cd /Volumes/MOVESPEED/ai-tools/attractor/unified-llm-python && python -m pytest tests/test_client.py -v
```

Expected: PASS (10 tests passed)

**Step 5: Commit**

```bash
git add src/unified_llm/client.py tests/test_client.py
git commit -m "feat: implement core Client with provider routing"
```

---

## Provider Adapters Implementation

### Task 14-20: Anthropic Adapter (Messages API)

*Due to length constraints, the following tasks are summarized. Each follows the TDD pattern.*

### Task 14: Anthropic Adapter - Basic Structure and Authentication

**Files:**
- Create: `src/unified_llm/providers/anthropic.py`
- Create: `tests/test_providers/test_anthropic_basic.py`

**Tests:**
- Test adapter can be initialized with API key
- Test adapter has correct name
- Test adapter reads base_url from env or constructor
- Test authentication header is set correctly

### Task 15: Anthropic Adapter - Request Transformation

**Files:**
- Create: `src/unified_llm/providers/anthropic/_transform.py`
- Create: `tests/test_providers/test_anthropic_transform.py`

**Tests:**
- Test system message extraction
- Test message role mapping
- Test content part transformation (text, image, tool_call)
- Test tool definition transformation
- Test tool_choice mapping
- Test provider_options handling

### Task 16: Anthropic Adapter - Response Transformation

**Files:**
- Create: `src/unified_llm/providers/anthropic/_response.py`
- Create: `tests/test_providers/test_anthropic_response.py`

**Tests:**
- Test text content extraction
- Test tool_call content extraction
- Test thinking block handling
- Test usage mapping
- Test finish_reason mapping
- Test rate limit header parsing

### Task 17: Anthropic Adapter - Streaming

**Files:**
- Create: `src/unified_llm/providers/anthropic/_stream.py`
- Create: `tests/test_providers/test_anthropic_stream.py`

**Tests:**
- Test SSE event parsing
- Test content_block_start events
- Test content_block_delta events
- Test message_stop events
- Test thinking block streaming

### Task 18: Anthropic Adapter - Error Handling

**Files:**
- Create: `src/unified_llm/providers/anthropic/_errors.py`
- Create: `tests/test_providers/test_anthropic_errors.py`

**Tests:**
- Test 401 -> AuthenticationError
- Test 429 -> RateLimitError
- Test 400 -> InvalidRequestError
- Test 5xx -> ServerError
- Test Retry-After header parsing

### Task 19: Anthropic Adapter - Prompt Caching

**Files:**
- Modify: `src/unified_llm/providers/anthropic/_transform.py`
- Create: `tests/test_providers/test_anthropic_caching.py`

**Tests:**
- Test cache_control injected on system message
- Test cache_control injected on tool definitions
- Test cache_control injected on conversation prefix
- Test cache_read_tokens reported in usage
- Test cache_write_tokens reported in usage
- Test auto_cache can be disabled via provider_options

### Task 20: Anthropic Adapter - Integration Tests

**Files:**
- Create: `tests/test_providers/integration_anthropic.py`

**Tests:**
- Test actual API call with real API key (skip if no key)
- Test simple text generation
- Test streaming text generation
- Test image input
- Test tool calling
- Test multi-turn conversation with caching

---

### Task 21-27: OpenAI Adapter (Responses API)

*Similar structure to Anthropic adapter*

### Task 21: OpenAI Adapter - Basic Structure
### Task 22: OpenAI Adapter - Request Transformation (Responses API format)
### Task 23: OpenAI Adapter - Response Transformation
### Task 24: OpenAI Adapter - Streaming (Responses API format)
### Task 25: OpenAI Adapter - Error Handling
### Task 26: OpenAI Adapter - Reasoning Tokens
### Task 27: OpenAI Adapter - Integration Tests

---

### Task 28-34: Gemini Adapter (Gemini API)

*Similar structure to Anthropic adapter*

### Task 28: Gemini Adapter - Basic Structure
### Task 29: Gemini Adapter - Request Transformation
### Task 30: Gemini Adapter - Response Transformation
### Task 31: Gemini Adapter - Streaming (SSE/JSON blocks)
### Task 32: Gemini Adapter - Error Handling
### Task 33: Gemini Adapter - Synthetic Tool Call IDs
### Task 34: Gemini Adapter - Integration Tests

---

## Layer 4: High-Level API

### Task 35: Implement generate() Function

**Files:**
- Create: `src/unified_llm/high_level/_generate.py`
- Create: `tests/test_high_level/test_generate.py`

**Tests:**
- Test generate with simple prompt
- Test generate with messages
- Test generate error when both prompt and messages provided
- Test generate with system parameter
- Test generate with temperature
- Test generate with max_tokens
- Test generate returns GenerateResult with correct fields

**Implementation highlights:**
- Prompt standardization (prompt -> single user message)
- Message construction from prompt
- Client.complete() wrapper
- No tool execution yet (that's next)

### Task 36: Implement Tool Execution Loop

**Files:**
- Create: `src/unified_llm/high_level/_tools.py`
- Create: `tests/test_high_level/test_tools.py`

**Tests:**
- Test tool with execute handler is executed
- Test tool without execute returns tool_calls to caller
- Test max_tool_rounds limits iterations
- Test max_tool_rounds=0 disables execution
- Test parallel tool execution (multiple tools run concurrently)
- Test tool errors sent as error results
- Test unknown tool calls return error

### Task 37: Implement stream() Function

**Files:**
- Create: `src/unified_llm/high_level/_stream.py`
- Create: `tests/test_high_level/test_stream.py`

**Tests:**
- Test stream yields TEXT_DELTA events
- Test stream() returns StreamResult
- Test StreamResult.response() after iteration
- Test StreamResult.text_stream convenience
- Test stream with tool execution (STEP_FINISH events)

### Task 38: Implement generate_object() Function

**Files:**
- Create: `src/unified_llm/high_level/_structured.py`
- Create: `tests/test_high_level/test_structured.py`

**Tests:**
- Test generate_object returns parsed object
- Test generate_object with OpenAI (native json_schema)
- Test generate_object with Gemini (native schema)
- Test generate_object with Anthropic (prompt injection fallback)
- Test generate_object raises NoObjectGeneratedError on parse failure
- Test generate_object validates against schema

### Task 39: Implement Retry and Timeout in High-Level API

**Files:**
- Modify: `src/unified_llm/high_level/_generate.py`, `_stream.py`, `_structured.py`
- Create: `tests/test_high_level/test_retry_timeout.py`

**Tests:**
- Test generate with max_retries retries transient errors
- Test generate with max_retries=0 doesn't retry
- Test stream doesn't retry after data starts
- Test timeout with simple timeout value
- Test timeout with TimeoutConfig (total and per_step)

### Task 40: Implement Abort Signal Support

**Files:**
- Create: `src/unified_llm/high_level/_abort.py`
- Create: `tests/test_high_level/test_abort.py`

**Tests:**
- Test AbortController.abort() cancels generate()
- Test abort raises AbortError
- Test abort cancels stream mid-stream

---

## Model Directory

### Task 41: Create Model Catalog

**Files:**
- Create: `src/unified_llm/models/_catalog.py`
- Create: `tests/test_models/test_catalog.py`

**Tests:**
- Test get_model_info returns correct model
- Test get_model_info returns None for unknown model
- Test list_models returns all models
- Test list_models filters by provider
- Test get_latest_model returns newest model
- Test get_latest_model filters by capability

**Implementation:**
```python
# src/unified_llm/models/_catalog.py
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ModelInfo:
    id: str
    provider: str
    display_name: str
    context_window: int
    max_output: int | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    aliases: list[str] | None = None

MODELS = [
    # Anthropic
    ModelInfo(id="claude-opus-4-6", provider="anthropic", display_name="Claude Opus 4.6",
              context_window=200000, supports_tools=True, supports_vision=True, supports_reasoning=True),
    ModelInfo(id="claude-sonnet-4-5", provider="anthropic", display_name="Claude Sonnet 4.5",
              context_window=200000, supports_tools=True, supports_vision=True, supports_reasoning=True),

    # OpenAI
    ModelInfo(id="gpt-5.2", provider="openai", display_name="GPT-5.2",
              context_window=1047576, supports_tools=True, supports_vision=True, supports_reasoning=True),
    # ... more models
]

def get_model_info(model_id: str) -> ModelInfo | None:
    for model in MODELS:
        if model.id == model_id or (model.aliases and model_id in model.aliases):
            return model
    return None

def list_models(provider: str | None = None) -> list[ModelInfo]:
    if provider:
        return [m for m in MODELS if m.provider == provider]
    return MODELS.copy()

def get_latest_model(provider: str, capability: str | None = None) -> ModelInfo | None:
    models = list_models(provider)
    if capability:
        models = [m for m in models if getattr(m, f"supports_{capability}", False)]
    return models[0] if models else None
```

---

## Final Tasks

### Task 42: Update Main Package Exports

**Files:**
- Modify: `src/unified_llm/__init__.py`

**Step:** Ensure all public APIs are exported from main package.

### Task 43: Add Comprehensive Documentation

**Files:**
- Create: `docs/usage.md`
- Create: `docs/providers.md`
- Create: `docs/api.md`

### Task 44: Add Type Checking Configuration

**Files:**
- Create: `pyproject.toml` mypy configuration
- Run: `mypy src/unified_llm`

### Task 45: Add Code Coverage

**Files:**
- Run: `pytest --cov=unified_llm --cov-report=html`
- Target: >90% coverage

### Task 46: Integration Test Suite

**Files:**
- Create: `tests/integration/test_providers_e2e.py`

**Tests:**
- Run Definition of Done checklist (Section 8 of spec)
- Test all providers with real API keys
- Verify cross-provider parity matrix

---

## Summary

This plan implements the complete unified LLM spec in Python:

1. **Project Setup** (Task 1): Configuration and structure
2. **Layer 1 - Data Models** (Tasks 2-7): All data types
3. **Layer 1 - Exceptions** (Task 8): Complete error hierarchy
4. **Layer 1 - Provider Interface** (Task 9): Adapter contract
5. **Layer 2 - Utilities** (Tasks 10-12): HTTP, SSE, retry
6. **Layer 3 - Core Client** (Task 13): Routing and middleware
7. **Provider Adapters** (Tasks 14-34): Anthropic, OpenAI, Gemini
8. **Layer 4 - High-Level API** (Tasks 35-40): generate, stream, tools
9. **Model Catalog** (Task 41): Model directory
10. **Final Tasks** (Tasks 42-46): Exports, docs, type checking

Total: **46 tasks**, each broken into 5 steps (test, fail, implement, pass, commit).

Estimated timeline: 40-60 hours of development time.

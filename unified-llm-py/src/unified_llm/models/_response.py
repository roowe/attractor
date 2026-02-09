"""Response data model for LLM API results."""

from dataclasses import dataclass, field
from datetime import datetime
from unified_llm.models._message import Message
from unified_llm.models._content import ToolCallData


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

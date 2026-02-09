"""Request data model for LLM API calls."""

from dataclasses import dataclass
from typing import Literal
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

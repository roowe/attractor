"""Content part data models for messages."""

from dataclasses import dataclass
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

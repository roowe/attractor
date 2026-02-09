"""Stream event data models."""

from dataclasses import dataclass
from enum import Enum
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

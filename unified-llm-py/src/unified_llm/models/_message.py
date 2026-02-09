"""Message data model for conversations."""

from dataclasses import dataclass
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
        from unified_llm.models._enums import ContentKind
        return "".join(
            part.text for part in self.content
            if part.kind == ContentKind.TEXT and part.text is not None
        )

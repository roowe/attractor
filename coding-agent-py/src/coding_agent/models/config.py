# coding-agent-py/src/coding_agent/models/config.py
from typing import Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator


ReasoningEffort = Literal["low", "medium", "high", None]


class SessionConfig(BaseModel):
    """Configuration for a coding agent session."""

    max_turns: int = Field(default=0, ge=0)
    max_tool_rounds_per_input: int = Field(default=200, ge=1)
    default_command_timeout_ms: int = Field(default=10000, ge=1000)
    max_command_timeout_ms: int = Field(default=600000, ge=1000)
    reasoning_effort: Optional[ReasoningEffort] = None
    tool_output_limits: Dict[str, int] = Field(default_factory=dict)
    tool_line_limits: Dict[str, Optional[int]] = Field(default_factory=dict)
    enable_loop_detection: bool = True
    loop_detection_window: int = Field(default=10, ge=2)
    max_subagent_depth: int = Field(default=1, ge=0)

    @field_validator("max_command_timeout_ms")
    @classmethod
    def max_timeout_must_be_greater_than_default(cls, v, info):
        if "default_command_timeout_ms" in info.data:
            if v < info.data["default_command_timeout_ms"]:
                raise ValueError("max_command_timeout_ms must be >= default_command_timeout_ms")
        return v

# coding-agent-py/src/coding_agent/models/tool.py
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


class ToolResult(BaseModel):
    """Result of executing a tool call."""
    tool_call_id: str
    content: str
    is_error: bool = False


class ToolDefinition(BaseModel):
    """Definition of a tool for LLM consumption."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema

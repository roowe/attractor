# coding-agent-py/src/coding_agent/tools/registry.py
from typing import Dict, Callable, Any, Awaitable, List
from pydantic import BaseModel

from coding_agent.models.tool import ToolDefinition
from coding_agent.exec.environment import ExecutionEnvironment


ToolExecutor = Callable[[Dict[str, Any], ExecutionEnvironment], Awaitable[str]]


class RegisteredTool(BaseModel):
    """A registered tool with its definition and executor."""

    definition: ToolDefinition
    executor: ToolExecutor

    class Config:
        arbitrary_types_allowed = True


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """Register a new tool or replace an existing one."""
        self._tools[tool.definition.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions."""
        return [tool.definition for tool in self._tools.values()]

    def names(self) -> List[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())

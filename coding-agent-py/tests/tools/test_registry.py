# coding-agent-py/tests/tools/test_registry.py
import pytest
from typing import Dict, Any
from coding_agent.tools.registry import ToolRegistry, RegisteredTool
from coding_agent.models.tool import ToolDefinition
from coding_agent.exec.environment import LocalExecutionEnvironment


async def dummy_tool_executor(arguments: Dict[str, Any], env: LocalExecutionEnvironment) -> str:
    """A dummy tool executor for testing."""
    return f"Executed with: {arguments}"


def test_tool_registry_register_and_get():
    registry = ToolRegistry()

    tool = RegisteredTool(
        definition=ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        ),
        executor=dummy_tool_executor,
    )

    registry.register(tool)

    retrieved = registry.get("test_tool")
    assert retrieved is not None
    assert retrieved.definition.name == "test_tool"


def test_tool_registry_get_nonexistent():
    registry = ToolRegistry()

    retrieved = registry.get("nonexistent")
    assert retrieved is None


def test_tool_registry_unregister():
    registry = ToolRegistry()

    tool = RegisteredTool(
        definition=ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        ),
        executor=dummy_tool_executor,
    )

    registry.register(tool)
    assert registry.get("test_tool") is not None

    registry.unregister("test_tool")
    assert registry.get("test_tool") is None


def test_tool_registry_definitions():
    registry = ToolRegistry()

    tool1 = RegisteredTool(
        definition=ToolDefinition(
            name="tool1",
            description="Tool 1",
            parameters={"type": "object"},
        ),
        executor=dummy_tool_executor,
    )

    tool2 = RegisteredTool(
        definition=ToolDefinition(
            name="tool2",
            description="Tool 2",
            parameters={"type": "object"},
        ),
        executor=dummy_tool_executor,
    )

    registry.register(tool1)
    registry.register(tool2)

    definitions = registry.definitions()
    assert len(definitions) == 2
    assert definitions[0].name == "tool1"
    assert definitions[1].name == "tool2"


def test_tool_registry_names():
    registry = ToolRegistry()

    tool1 = RegisteredTool(
        definition=ToolDefinition(
            name="tool1",
            description="Tool 1",
            parameters={"type": "object"},
        ),
        executor=dummy_tool_executor,
    )

    tool2 = RegisteredTool(
        definition=ToolDefinition(
            name="tool2",
            description="Tool 2",
            parameters={"type": "object"},
        ),
        executor=dummy_tool_executor,
    )

    registry.register(tool1)
    registry.register(tool2)

    names = registry.names()
    assert "tool1" in names
    assert "tool2" in names

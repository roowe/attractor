# coding-agent-py/src/coding_agent/tools/core/file_tools.py
from typing import Dict, Any

from coding_agent.tools.registry import ToolRegistry, RegisteredTool
from coding_agent.models.tool import ToolDefinition
from coding_agent.exec.environment import ExecutionEnvironment


async def read_file_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for read_file tool."""
    file_path = arguments.get("file_path")
    offset = arguments.get("offset")
    limit = arguments.get("limit", 2000)

    if not file_path:
        return "Error: file_path is required"

    try:
        content = await env.read_file(file_path, offset=offset, limit=limit)

        # Add line numbers
        lines = content.split("\n")
        start_line = offset if offset else 1
        numbered_lines = [f"{i + start_line}|{line}" for i, line in enumerate(lines)]

        return "\n".join(numbered_lines)

    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def register_read_file_tool(registry: ToolRegistry) -> None:
    """Register the read_file tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="read_file",
            description=(
                "Read a file from the filesystem. Returns content with line numbers."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start reading from"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 2000)"
                    }
                },
                "required": ["file_path"],
            },
        ),
        executor=read_file_executor,
    )
    registry.register(tool)


async def write_file_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for write_file tool."""
    file_path = arguments.get("file_path")
    content = arguments.get("content", "")

    if not file_path:
        return "Error: file_path is required"

    try:
        await env.write_file(file_path, content)
        bytes_written = len(content.encode("utf-8"))
        return f"Successfully wrote {bytes_written} bytes to {file_path}"

    except Exception as e:
        return f"Error writing file: {str(e)}"


def register_write_file_tool(registry: ToolRegistry) -> None:
    """Register the write_file tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="write_file",
            description="Write content to a file. Creates the file and parent directories if needed.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content to write"
                    }
                },
                "required": ["file_path", "content"],
            },
        ),
        executor=write_file_executor,
    )
    registry.register(tool)


async def edit_file_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for edit_file tool."""
    file_path = arguments.get("file_path")
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string", "")
    replace_all = arguments.get("replace_all", False)

    if not file_path or not old_string:
        return "Error: file_path and old_string are required"

    try:
        # Read current content
        content = await env.read_file(file_path)

        # Check if old_string exists
        if old_string not in content:
            return f"Error: old_string not found in file. Current content may have changed."

        # Count occurrences if not replace_all
        if not replace_all:
            count = content.count(old_string)
            if count > 1:
                return f"Error: old_string appears {count} times. Use replace_all=true or provide more context."

        # Replace
        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)

        # Write back
        await env.write_file(file_path, new_content)
        replacements = new_content.count(new_string) if new_string else 0
        return f"Successfully made {replacements} replacement(s) in {file_path}"

    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error editing file: {str(e)}"


def register_edit_file_tool(registry: ToolRegistry) -> None:
    """Register the edit_file tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="edit_file",
            description="Replace an exact string occurrence in a file.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false)"
                    }
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        ),
        executor=edit_file_executor,
    )
    registry.register(tool)

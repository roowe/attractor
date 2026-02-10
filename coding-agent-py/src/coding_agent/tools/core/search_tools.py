# coding-agent-py/src/coding_agent/tools/core/search_tools.py
from typing import Dict, Any

from coding_agent.tools.registry import ToolRegistry, RegisteredTool
from coding_agent.models.tool import ToolDefinition
from coding_agent.exec.environment import ExecutionEnvironment
from coding_agent.exec.grep_options import GrepOptions


async def shell_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for shell tool."""
    command = arguments.get("command")
    timeout_ms = arguments.get("timeout_ms")
    description = arguments.get("description", "")

    if not command:
        return "Error: command is required"

    # Get timeout from config or use default
    if timeout_ms is None:
        timeout_ms = 10000  # Default 10 seconds

    try:
        result = await env.exec_command(command, timeout_ms=timeout_ms)

        output = ""
        if description:
            output += f"[{description}]\n"

        output += f"Exit code: {result.exit_code}"

        if result.stdout:
            output += f"\n{result.stdout}"

        if result.stderr:
            output += f"\n{result.stderr}"

        if result.timed_out:
            output += "\n[Command timed out]"

        return output

    except Exception as e:
        return f"Error executing command: {str(e)}"


def register_shell_tool(registry: ToolRegistry) -> None:
    """Register the shell tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="shell",
            description="Execute a shell command. Returns stdout, stderr, and exit code.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Override default timeout in milliseconds"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of what this command does"
                    }
                },
                "required": ["command"],
            },
        ),
        executor=shell_executor,
    )
    registry.register(tool)


async def grep_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for grep tool."""
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")
    glob_filter = arguments.get("glob_filter")
    case_insensitive = arguments.get("case_insensitive", False)
    max_results = arguments.get("max_results", 100)

    if not pattern:
        return "Error: pattern is required"

    try:
        options = GrepOptions(
            pattern=pattern,
            path=path,
            glob_filter=glob_filter,
            case_insensitive=case_insensitive,
            max_results=max_results,
        )

        return await env.grep(options)

    except Exception as e:
        return f"Error searching files: {str(e)}"


def register_grep_tool(registry: ToolRegistry) -> None:
    """Register the grep tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="grep",
            description="Search file contents using a regex pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search (default: working directory)"
                    },
                    "glob_filter": {
                        "type": "string",
                        "description": "File pattern filter (e.g., '*.py')"
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default: false)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 100)"
                    }
                },
                "required": ["pattern"],
            },
        ),
        executor=grep_executor,
    )
    registry.register(tool)


async def glob_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for glob tool."""
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")

    if not pattern:
        return "Error: pattern is required"

    try:
        results = await env.glob(pattern, path)

        if not results:
            return f"No files found matching pattern: {pattern}"

        return "\n".join(results)

    except Exception as e:
        return f"Error finding files: {str(e)}"


def register_glob_tool(registry: ToolRegistry) -> None:
    """Register the glob tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="glob",
            description="Find files matching a glob pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.ts')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory (default: working directory)"
                    }
                },
                "required": ["pattern"],
            },
        ),
        executor=glob_executor,
    )
    registry.register(tool)

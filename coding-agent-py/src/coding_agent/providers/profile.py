# coding-agent-py/src/coding_agent/providers/profile.py
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from coding_agent.tools.registry import ToolRegistry
from coding_agent.models.tool import ToolDefinition
from coding_agent.exec.environment import ExecutionEnvironment
from coding_agent.tools.core.file_tools import (
    register_read_file_tool,
    register_write_file_tool,
    register_edit_file_tool,
)
from coding_agent.tools.core.search_tools import (
    register_shell_tool,
    register_grep_tool,
    register_glob_tool,
)


class ProviderProfile(ABC):
    """Base class for provider-specific profiles."""

    def __init__(self, id: str, model: str) -> None:
        self.id = id
        self.model = model
        self.tool_registry = ToolRegistry()

    @abstractmethod
    def build_system_prompt(self, env: ExecutionEnvironment, project_docs: Dict[str, str]) -> str:
        """Build the system prompt for this provider."""
        pass

    @abstractmethod
    def tools(self) -> list[ToolDefinition]:
        """Get the tool definitions for this provider."""
        pass

    @abstractmethod
    def provider_options(self) -> Optional[Dict[str, Any]]:
        """Get provider-specific options for LLM requests."""
        pass

    # Capability flags (can be overridden)
    supports_reasoning: bool = False
    supports_streaming: bool = True
    supports_parallel_tool_calls: bool = False
    context_window_size: int = 200000


class AnthropicProfile(ProviderProfile):
    """Profile aligned with Claude Code's toolset and prompts."""

    def __init__(self, model: str = "claude-opus-4-6") -> None:
        super().__init__(id="anthropic", model=model)
        self.supports_reasoning = True
        self.supports_streaming = True
        self.supports_parallel_tool_calls = True
        self.context_window_size = 200000

        # Register core tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register Claude Code-aligned tools."""
        register_read_file_tool(self.tool_registry)
        register_write_file_tool(self.tool_registry)
        register_edit_file_tool(self.tool_registry)
        register_shell_tool(self.tool_registry)
        register_grep_tool(self.tool_registry)
        register_glob_tool(self.tool_registry)

    def tools(self) -> list[ToolDefinition]:
        """Get all tool definitions."""
        return self.tool_registry.definitions()

    def build_system_prompt(self, env: ExecutionEnvironment, project_docs: Dict[str, str]) -> str:
        """Build Claude Code-aligned system prompt."""
        prompt = """You are Claude, an AI programming assistant.

## File Operations

You can read, write, and edit files:
- Use read_file to view file contents with line numbers
- Use edit_file to make precise string replacements (old_string must be unique)
- Use write_file to create new files or completely replace contents
- Prefer editing existing files over creating new ones

## Command Execution

You can execute shell commands using the shell tool. Commands timeout after 120 seconds by default.

## Search

- Use grep to search file contents with regex patterns
- Use glob to find files by name patterns

## Guidelines

- Always read a file before editing it
- When edit_file fails because old_string is not unique, read the file and provide more context
- Use exact string matching for edits
- Check command output before proceeding

"""

        # Add environment context
        prompt += f"\n<environment>\n"
        prompt += f"Working directory: {env.working_directory()}\n"
        prompt += f"Platform: {env.platform()}\n"
        prompt += f"OS version: {env.os_version()}\n"
        prompt += f"</environment>\n"

        # Add project docs
        if project_docs:
            prompt += "\n## Project Instructions\n\n"
            for name, content in project_docs.items():
                prompt += f"### {name}\n{content}\n\n"

        return prompt

    def provider_options(self) -> Optional[Dict[str, Any]]:
        """Get Anthropic-specific provider options."""
        return {
            "anthropic": {
                "beta_headers": {
                    "max-tokens": 8192,
                }
            }
        }


class OpenAIProfile(ProviderProfile):
    """Profile aligned with codex-rs toolset and prompts."""

    def __init__(self, model: str = "gpt-5.2-codex") -> None:
        super().__init__(id="openai", model=model)
        self.supports_reasoning = True
        self.supports_streaming = True
        self.supports_parallel_tool_calls = True
        self.context_window_size = 128000

        self._register_tools()

    def _register_tools(self) -> None:
        """Register codex-rs-aligned tools."""
        # OpenAI uses apply_patch instead of edit_file
        register_read_file_tool(self.tool_registry)
        register_write_file_tool(self.tool_registry)
        register_shell_tool(self.tool_registry)
        register_grep_tool(self.tool_registry)
        register_glob_tool(self.tool_registry)
        # TODO: Register apply_patch tool

    def tools(self) -> list[ToolDefinition]:
        return self.tool_registry.definitions()

    def build_system_prompt(self, env: ExecutionEnvironment, project_docs: Dict[str, str]) -> str:
        """Build codex-rs-aligned system prompt."""
        prompt = """You are Codex, an AI programming assistant.

## File Operations

You can read and write files, and apply patches:
- Use read_file to view file contents
- Use write_file to create new files
- Use apply_patch to make efficient edits (supports create, update, delete, rename)

## Command Execution

You can execute shell commands. Commands timeout after 10 seconds by default.

## Guidelines

- Use apply_patch for all file modifications when possible
- Apply patches use v4a format with context-aware hunks
- Check command output before proceeding

"""

        prompt += f"\n<environment>\n"
        prompt += f"Working directory: {env.working_directory()}\n"
        prompt += f"Platform: {env.platform()}\n"
        prompt += f"</environment>\n"

        return prompt

    def provider_options(self) -> Optional[Dict[str, Any]]:
        """Get OpenAI-specific provider options."""
        return {
            "openai": {
                "reasoning": {
                    "effort": "medium",
                }
            }
        }


class GeminiProfile(ProviderProfile):
    """Profile aligned with gemini-cli toolset and prompts."""

    def __init__(self, model: str = "gemini-2.5-pro") -> None:
        super().__init__(id="gemini", model=model)
        self.supports_reasoning = True
        self.supports_streaming = True
        self.supports_parallel_tool_calls = False
        self.context_window_size = 1000000

        self._register_tools()

    def _register_tools(self) -> None:
        """Register gemini-cli-aligned tools."""
        register_read_file_tool(self.tool_registry)
        register_write_file_tool(self.tool_registry)
        register_edit_file_tool(self.tool_registry)
        register_shell_tool(self.tool_registry)
        register_grep_tool(self.tool_registry)
        register_glob_tool(self.tool_registry)

    def tools(self) -> list[ToolDefinition]:
        return self.tool_registry.definitions()

    def build_system_prompt(self, env: ExecutionEnvironment, project_docs: Dict[str, str]) -> str:
        """Build gemini-cli-aligned system prompt."""
        prompt = """You are Gemini, an AI programming assistant.

## File Operations

You can read, write, and edit files:
- Use read_file to view file contents
- Use write_file to create new files
- Use edit_file to make changes

## Command Execution

You can execute shell commands with a 10 second default timeout.

## Project Instructions

Look for GEMINI.md files for project-specific instructions.

"""

        prompt += f"\n<environment>\n"
        prompt += f"Working directory: {env.working_directory()}\n"
        prompt += f"</environment>\n"

        return prompt

    def provider_options(self) -> Optional[Dict[str, Any]]:
        """Get Gemini-specific provider options."""
        return {
            "gemini": {
                "safety_settings": [],
            }
        }

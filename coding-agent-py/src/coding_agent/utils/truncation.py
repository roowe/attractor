# coding-agent-py/src/coding_agent/utils/truncation.py
from typing import Literal

# Default tool output limits (from spec)
DEFAULT_TOOL_LIMITS = {
    "read_file": 50000,
    "shell": 30000,
    "grep": 20000,
    "glob": 20000,
    "edit_file": 10000,
    "write_file": 1000,
    "spawn_agent": 20000,
}

DEFAULT_TRUNCATION_MODES = {
    "read_file": "head_tail",
    "shell": "head_tail",
    "grep": "tail",
    "glob": "tail",
    "edit_file": "tail",
    "write_file": "tail",
    "spawn_agent": "head_tail",
}

# Default line limits
DEFAULT_LINE_LIMITS = {
    "shell": 256,
    "grep": 200,
    "glob": 500,
}


def truncate_output(
    output: str,
    max_chars: int,
    mode: Literal["head_tail", "tail"] = "head_tail",
) -> str:
    """Truncate output based on character limit."""
    if len(output) <= max_chars:
        return output

    removed = len(output) - max_chars

    if mode == "head_tail":
        half = max_chars // 2
        head = output[:half]
        tail = output[-half:]

        warning = (
            f"\n\n[Warning: Tool output has been truncated. {removed} characters "
            f"have been removed from the middle. Full output is available in the event stream. "
            f"If you need to see specific parts, please re-run the tool with more targeted arguments.]\n\n"
        )
        return head + warning + tail

    else:  # tail mode
        tail = output[-max_chars:]

        warning = (
            f"[Warning: Tool output has been truncated. The first {removed} characters "
            f"have been removed. Full output is available in the event stream.]\n\n"
        )
        return warning + tail


def truncate_lines(output: str, max_lines: int) -> str:
    """Truncate output based on line limit."""
    lines = output.split("\n")

    if len(lines) <= max_lines:
        return output

    head_count = max_lines // 2
    tail_count = max_lines - head_count
    omitted = len(lines) - head_count - tail_count

    head = "\n".join(lines[:head_count])
    tail = "\n".join(lines[-tail_count:])

    return f"{head}\n[... {omitted} lines omitted ...]\n{tail}"


def truncate_tool_output(output: str, tool_name: str, config) -> str:
    """Truncate tool output using configured limits."""
    # Get character limit
    max_chars = config.tool_output_limits.get(
        tool_name,
        DEFAULT_TOOL_LIMITS.get(tool_name, 50000)
    )

    # Get truncation mode
    mode = DEFAULT_TRUNCATION_MODES.get(tool_name, "head_tail")

    # Step 1: Character-based truncation (always runs)
    result = truncate_output(output, max_chars, mode)

    # Step 2: Line-based truncation (if configured)
    max_lines = config.tool_line_limits.get(
        tool_name,
        DEFAULT_LINE_LIMITS.get(tool_name)
    )

    if max_lines is not None:
        result = truncate_lines(result, max_lines)

    return result

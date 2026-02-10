# coding-agent-py/src/coding_agent/exec/grep_options.py
from typing import Optional
from pydantic import BaseModel, Field


class GrepOptions(BaseModel):
    """Options for grep tool."""
    pattern: str
    path: str = "."
    glob_filter: Optional[str] = None
    case_insensitive: bool = False
    max_results: int = 100
    output_mode: str = "content"  # "content", "files_with_matches", "count"
    context_lines: int = 0

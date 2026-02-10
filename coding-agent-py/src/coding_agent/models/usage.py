# coding-agent-py/src/coding_agent/models/usage.py
from pydantic import BaseModel


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __init__(self, **data):
        if "total_tokens" not in data:
            data["total_tokens"] = data.get("prompt_tokens", 0) + data.get("completion_tokens", 0)
        super().__init__(**data)

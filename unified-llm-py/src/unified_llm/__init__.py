"""
Unified LLM Client - A single interface for OpenAI, Anthropic, and Gemini.

This package provides a unified API for interacting with multiple LLM providers.
"""

__version__ = "0.1.0"

# High-level API (Layer 4)
# from unified_llm.high_level import generate, stream, generate_object

# Core client (Layer 3)
from unified_llm.client import Client

# Provider adapters
from unified_llm.providers import (
    AnthropicAdapter,
    OpenAIAdapter,
    GeminiAdapter,
)

# Data models
from unified_llm.models import (
    Message,
    Request,
    Response,
    Usage,
    ToolChoice,
    ResponseFormat,
    StreamEvent,
    Role,
    ContentKind,
    ContentPart,
)

# Exceptions
from unified_llm._exceptions import (
    SDKError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
)

__all__ = [
    # Version
    "__version__",
    # Core client
    "Client",
    # Providers
    "AnthropicAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    # Models
    "Message",
    "Request",
    "Response",
    "Usage",
    "ToolChoice",
    "ResponseFormat",
    "StreamEvent",
    "Role",
    "ContentKind",
    "ContentPart",
    # Exceptions
    "SDKError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
]

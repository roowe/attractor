"""Base provider adapter interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from unified_llm.models import Request, Response, StreamEvent


class ProviderAdapter(ABC):
    """Abstract base for provider adapters.

    Each provider (OpenAI, Anthropic, Gemini) implements this interface.
    """

    #: Provider name (e.g., "openai", "anthropic", "gemini")
    name: str

    @abstractmethod
    async def complete(self, request: Request) -> Response:
        """Send request, block until complete, return full response.

        Args:
            request: The unified request object.

        Returns:
            Response from the provider.

        Raises:
            ProviderError: On provider errors.
        """
        raise NotImplementedError

    @abstractmethod
    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send request, return async iterator of stream events.

        Args:
            request: The unified request object.

        Yields:
            StreamEvent objects as they arrive.

        Raises:
            ProviderError: On provider errors.
        """
        raise NotImplementedError

    async def initialize(self) -> None:
        """Validate configuration at startup.

        Called when adapter is registered with Client.
        Raise ConfigurationError if configuration is invalid.
        """
        pass

    async def close(self) -> None:
        """Release resources (HTTP connections, etc.).

        Called by Client.close().
        """
        pass

    def supports_tool_choice(self, mode: str) -> bool:
        """Check if a specific tool choice mode is supported.

        Args:
            mode: "auto", "none", "required", "named"

        Returns:
            True if the mode is supported.
        """
        return mode in ("auto", "none", "required", "named")

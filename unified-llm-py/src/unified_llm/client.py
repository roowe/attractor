"""Core client with provider routing and middleware."""

import os
from typing import AsyncIterator

from unified_llm._exceptions import ConfigurationError
from unified_llm.models import Request, Response, StreamEvent
from unified_llm.providers._base import ProviderAdapter


class Client:
    """Core LLM client with provider routing.

    The Client manages provider adapters, routes requests to the correct
    provider, and applies middleware.
    """

    def __init__(
        self,
        *,
        providers: dict[str, ProviderAdapter],
        default_provider: str | None = None,
        middleware: list | None = None,
    ):
        """Initialize Client.

        Args:
            providers: Mapping of provider names to adapters.
            default_provider: Default provider when not specified in request.
            middleware: List of middleware functions.
        """
        self._providers = providers
        self._default_provider = default_provider
        self._middleware = middleware or []

        if default_provider and default_provider not in providers:
            raise ConfigurationError(
                f"Default provider '{default_provider}' not found in registered providers"
            )

    @classmethod
    def from_env(cls) -> "Client":
        """Create Client from environment variables.

        Checks for ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY.

        Returns:
            Configured Client instance.
        """
        providers = {}
        default_provider = None

        # Import adapters here to avoid circular imports
        from unified_llm.providers import AnthropicAdapter, OpenAIAdapter, GeminiAdapter

        # Anthropic
        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            providers["anthropic"] = AnthropicAdapter(api_key=api_key)
            if default_provider is None:
                default_provider = "anthropic"

        # OpenAI
        if api_key := os.environ.get("OPENAI_API_KEY"):
            providers["openai"] = OpenAIAdapter(api_key=api_key)
            if default_provider is None:
                default_provider = "openai"

        # Gemini
        if api_key := os.environ.get("GEMINI_API_KEY"):
            providers["gemini"] = GeminiAdapter(api_key=api_key)
            if default_provider is None:
                default_provider = "gemini"

        return cls(providers=providers, default_provider=default_provider)

    async def complete(self, request: Request) -> Response:
        """Complete a request (blocking, non-streaming).

        Args:
            request: Unified request object.

        Returns:
            Response from the provider.

        Raises:
            ConfigurationError: If no provider is configured.
            ProviderError: On provider errors.
        """
        adapter = self._get_adapter(request)
        return await adapter.complete(request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Stream a request.

        Args:
            request: Unified request object.

        Yields:
            StreamEvent objects as they arrive.

        Raises:
            ConfigurationError: If no provider is configured.
            ProviderError: On provider errors.
        """
        adapter = self._get_adapter(request)
        async for event in adapter.stream(request):
            yield event

    def _get_adapter(self, request: Request) -> ProviderAdapter:
        """Get the adapter for a request.

        Args:
            request: Request with optional provider field.

        Returns:
            Provider adapter.

        Raises:
            ConfigurationError: If provider not found.
        """
        provider_name = request.provider or self._default_provider

        if provider_name is None:
            raise ConfigurationError(
                "No provider specified and no default provider configured"
            )

        if provider_name not in self._providers:
            raise ConfigurationError(
                f"Provider '{provider_name}' not registered. "
                f"Available providers: {list(self._providers.keys())}"
            )

        return self._providers[provider_name]

    async def close(self) -> None:
        """Close all provider adapters."""
        for adapter in self._providers.values():
            await adapter.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()

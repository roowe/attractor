"""Test Client functionality."""

import pytest
from unified_llm.client import Client
from unified_llm.models import Request, Response, Usage, FinishReason, Message, StreamEvent
from unified_llm.providers._base import ProviderAdapter


class MockAdapter(ProviderAdapter):
    """Mock provider adapter for testing."""

    name = "mock"

    def __init__(self):
        self.complete_called = False
        self.stream_called = False

    async def complete(self, request: Request) -> Response:
        self.complete_called = True
        return Response(
            id="test-123",
            model=request.model,
            provider=self.name,
            message=Message.assistant("Mock response"),
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    async def stream(self, request: Request):
        self.stream_called = True
        yield StreamEvent.stream_start()
        yield StreamEvent.text_delta("Mock")
        yield StreamEvent.text_delta(" response")
        yield StreamEvent.finish(
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=10, output_tokens=2, total_tokens=12),
        )


def test_client_construction_with_providers():
    """Test Client can be constructed with explicit providers."""
    adapter = MockAdapter()
    client = Client(providers={"mock": adapter}, default_provider="mock")

    assert client._default_provider == "mock"  # type: ignore


def test_client_complete_routes_to_provider():
    """Test complete() routes to correct provider."""
    import asyncio

    adapter = MockAdapter()
    client = Client(providers={"mock": adapter}, default_provider="mock")

    async def test():
        request = Request(model="test-model", messages=[Message.user("Hello")])
        response = await client.complete(request)

        assert adapter.complete_called is True
        assert response.text == "Mock response"

    asyncio.run(test())


def test_client_complete_explicit_provider():
    """Test complete() with explicit provider parameter."""
    import asyncio

    adapter1 = MockAdapter()
    adapter1.name = "mock1"
    adapter2 = MockAdapter()
    adapter2.name = "mock2"

    client = Client(providers={"mock1": adapter1, "mock2": adapter2}, default_provider="mock1")

    async def test():
        request = Request(model="test-model", messages=[Message.user("Hello")], provider="mock2")
        await client.complete(request)

        assert adapter2.complete_called is True
        assert adapter1.complete_called is False

    asyncio.run(test())


def test_client_stream_routes_to_provider():
    """Test stream() routes to correct provider."""
    import asyncio

    adapter = MockAdapter()
    client = Client(providers={"mock": adapter}, default_provider="mock")

    async def test():
        request = Request(model="test-model", messages=[Message.user("Hello")])
        events = []
        async for event in client.stream(request):
            events.append(event)

        assert adapter.stream_called is True
        assert len(events) == 4
        assert events[1].delta == "Mock"

    asyncio.run(test())


def test_client_no_provider_error():
    """Test error when no provider configured."""
    import asyncio

    client = Client(providers={}, default_provider=None)

    async def test():
        request = Request(model="test-model", messages=[Message.user("Hello")])
        with pytest.raises(Exception):  # ConfigurationError
            await client.complete(request)

    asyncio.run(test())


def test_client_unknown_provider_error():
    """Test error when requesting unknown provider."""
    import asyncio

    adapter = MockAdapter()
    client = Client(providers={"mock": adapter}, default_provider="mock")

    async def test():
        request = Request(model="test-model", messages=[Message.user("Hello")], provider="unknown")
        with pytest.raises(Exception):  # ConfigurationError
            await client.complete(request)

    asyncio.run(test())

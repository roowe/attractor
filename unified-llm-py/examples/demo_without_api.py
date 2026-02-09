"""
Demo script that works without API keys using mock adapters.
"""

import asyncio
from unified_llm import Client, Message, Request
from unified_llm.models import Response, Usage, FinishReason, StreamEvent, StreamEventType
from unified_llm.providers._base import ProviderAdapter


class MockAdapter(ProviderAdapter):
    """Mock provider adapter for demonstration."""

    name = "mock"

    async def complete(self, request: Request) -> Response:
        # Simulate response based on input
        user_input = request.messages[-1].text if request.messages else "Hello"

        responses = {
            "quantum": "Quantum computing uses quantum bits (qubits) that can exist in superposition, enabling parallel processing and exponential computational power for certain problems.",
            "haiku": "Code flows fast,\nBugs hide in the night,\nMorning light reveals.",
            "capital": "The capital of France is Paris, with an estimated population of 2.1 million people in the city center.",
        }

        # Find matching response
        response_text = "I understand you said: " + user_input
        for keyword, response in responses.items():
            if keyword.lower() in user_input.lower():
                response_text = response
                break

        return Response(
            id="mock-123",
            model=request.model,
            provider=self.name,
            message=Message.assistant(response_text),
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
        )

    async def stream(self, request: Request):
        yield StreamEvent.stream_start()

        user_input = request.messages[-1].text if request.messages else "Hello"

        # Simulate streaming response
        response_text = f"Streaming response to: {user_input}"

        # Stream word by word
        for word in response_text.split():
            yield StreamEvent.text_delta(word + " ")
            await asyncio.sleep(0.05)  # Simulate network delay

        yield StreamEvent.finish(
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        )


async def main():
    """Demonstrate the unified LLM client with mock adapter."""

    # Create client with mock adapter
    mock = MockAdapter()
    client = Client(
        providers={"mock": mock},
        default_provider="mock",
    )

    print("=== Unified LLM Client Demo ===\n")

    # Example 1: Simple text generation
    print("Example 1: Simple text generation")
    print("-" * 40)

    request = Request(
        model="mock-model",
        messages=[Message.user("Explain quantum computing in one paragraph.")],
    )

    response = await client.complete(request)
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"Response: {response.text}")
    print(f"Tokens: {response.usage.total_tokens}\n")

    # Example 2: Streaming generation
    print("Example 2: Streaming generation")
    print("-" * 40)
    print("Response: ", end="", flush=True)

    request = Request(
        model="mock-model",
        messages=[Message.user("Write a haiku about coding.")],
    )

    async for event in client.stream(request):
        if event.type == StreamEventType.TEXT_DELTA:
            print(event.delta, end="", flush=True)
    print("\n")

    # Example 3: Multi-turn conversation
    print("Example 3: Multi-turn conversation")
    print("-" * 40)

    messages = [
        Message.system("You are a helpful assistant."),
        Message.user("What is the capital of France?"),
    ]

    request = Request(model="mock-model", messages=messages)
    response = await client.complete(request)
    print(f"Q: What is the capital of France?")
    print(f"A: {response.text}\n")

    # Example 4: Model catalog
    print("Example 4: Model catalog")
    print("-" * 40)

    from unified_llm.models import list_models, get_model_info, get_latest_model

    # List Anthropic models
    anthropic_models = list_models("anthropic")
    print(f"Anthropic models: {[m.id for m in anthropic_models]}")

    # Get model info
    info = get_model_info("claude-opus-4-6")
    if info:
        print(f"Claude Opus 4.6:")
        print(f"  Display name: {info.display_name}")
        print(f"  Context window: {info.context_window}")
        print(f"  Supports tools: {info.supports_tools}")
        print(f"  Supports reasoning: {info.supports_reasoning}")
        print(f"  Cost: ${info.input_cost_per_million}/M input, ${info.output_cost_per_million}/M output")

    # Get latest model
    latest = get_latest_model("anthropic")
    print(f"\nLatest Anthropic model: {latest.id if latest else 'None'}")

    print("\n=== Demo complete ===")

    # Close the client
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

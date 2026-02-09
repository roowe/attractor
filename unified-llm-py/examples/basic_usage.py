"""
Example usage of the unified LLM client.

This demonstrates how to use the unified LLM client with different providers.
"""

import asyncio
import os

from unified_llm import Client, Message, Request


async def main():
    """Demonstrate basic usage of the unified LLM client."""

    # Create client from environment variables
    # Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY
    client = Client.from_env()

    # Example 1: Simple text generation
    print("=== Example 1: Simple text generation ===")
    request = Request(
        model="claude-opus-4-6",  # or "gpt-5.2", "gemini-2.5-flash"
        messages=[Message.user("What is quantum computing? Explain in one paragraph.")],
        max_tokens=100,
    )

    response = await client.complete(request)
    print(f"Response: {response.text}")
    print(f"Usage: {response.usage.total_tokens} tokens")
    print(f"Provider: {response.provider}")
    print()

    # Example 2: Streaming generation
    print("=== Example 2: Streaming generation ===")
    request = Request(
        model="gpt-5.2",
        messages=[Message.user("Write a haiku about coding.")],
        provider="openai",  # Explicit provider
    )

    print("Response: ", end="", flush=True)
    async for event in client.stream(request):
        from unified_llm.models import StreamEventType
        if event.type == StreamEventType.TEXT_DELTA:
            print(event.delta, end="", flush=True)
    print("\n")

    # Example 3: Multi-turn conversation
    print("=== Example 3: Multi-turn conversation ===")
    messages = [
        Message.system("You are a helpful assistant."),
        Message.user("What is the capital of France?"),
        Message.assistant("The capital of France is Paris."),
        Message.user("And what is its population?"),
    ]

    request = Request(
        model="claude-opus-4-6",
        messages=messages,
        max_tokens=50,
    )

    response = await client.complete(request)
    print(f"Response: {response.text}")
    print()

    # Example 4: Using model catalog
    print("=== Example 4: Model catalog ===")
    from unified_llm.models import list_models, get_model_info, get_latest_model

    # List all Anthropic models
    anthropic_models = list_models("anthropic")
    print(f"Anthropic models: {[m.id for m in anthropic_models]}")

    # Get model info
    info = get_model_info("claude-opus-4-6")
    if info:
        print(f"Claude Opus 4.6: {info.display_name}")
        print(f"  Context window: {info.context_window}")
        print(f"  Supports tools: {info.supports_tools}")

    # Get latest model by provider
    latest = get_latest_model("anthropic", "reasoning")
    if latest:
        print(f"Latest Anthropic reasoning model: {latest.id}")

    # Close the client
    await client.close()


if __name__ == "__main__":
    # Check for API keys
    if not any(os.environ.get(k) for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]):
        print("Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY")
        print("\nExample:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  python examples/basic_usage.py")
        exit(1)

    asyncio.run(main())

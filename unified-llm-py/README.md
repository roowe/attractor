# Unified LLM Client

A unified Python client for OpenAI, Anthropic, and Google Gemini LLM providers.

## Status

This is an implementation of the [Unified LLM Spec](https://github.com/yourusername/attractor). Currently implements:

- ✅ **Layer 1**: Data models (Message, Request, Response, Usage, StreamEvent)
- ✅ **Layer 1**: Exception hierarchy with retryable flags
- ✅ **Layer 1**: ProviderAdapter interface
- ✅ **Layer 2**: Utilities (HttpClient, SSE parser, retry logic)
- ✅ **Layer 3**: Core Client with provider routing
- ✅ **Provider Adapters**: Anthropic (Messages API), OpenAI (Responses API), Gemini (Gemini API)
- ✅ **Model Catalog**: Known models with capabilities and pricing
- ⏳ **Layer 4**: High-level API (`generate()`, `stream()`, `generate_object()`) - TODO

## Installation

```bash
pip install unified-llm
```

## Quick Start

```python
import asyncio
from unified_llm import Client, Message, Request

async def main():
    # Create client from environment variables
    client = Client.from_env()

    # Simple text generation
    request = Request(
        model="claude-opus-4-6",
        messages=[Message.user("Explain quantum computing in one paragraph")],
        max_tokens=100,
    )

    response = await client.complete(request)
    print(response.text)
    print(f"Tokens: {response.usage.total_tokens}")

asyncio.run(main())
```

## Configuration

Set environment variables for your providers:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

## Usage Examples

### Streaming Generation

```python
import asyncio
from unified_llm import Client, Request, Message
from unified_llm.models import StreamEventType

async def stream_example():
    client = Client.from_env()

    request = Request(
        model="gpt-5.2",
        messages=[Message.user("Write a haiku about coding")],
        provider="openai",  # Explicit provider selection
    )

    async for event in client.stream(request):
        if event.type == StreamEventType.TEXT_DELTA:
            print(event.delta, end="", flush=True)
    print()

asyncio.run(stream_example())
```

### Multi-turn Conversation

```python
from unified_llm import Client, Request, Message

async def conversation():
    client = Client.from_env()

    messages = [
        Message.system("You are a helpful assistant."),
        Message.user("What is the capital of France?"),
        Message.assistant("The capital of France is Paris."),
        Message.user("And what is its population?"),
    ]

    request = Request(model="claude-opus-4-6", messages=messages)
    response = await client.complete(request)
    print(response.text)

asyncio.run(conversation())
```

### Model Catalog

```python
from unified_llm.models import list_models, get_model_info, get_latest_model

# List all models for a provider
anthropic_models = list_models("anthropic")
for model in anthropic_models:
    print(f"{model.id}: {model.display_name}")

# Get detailed model info
info = get_model_info("claude-opus-4-6")
print(f"Context window: {info.context_window}")
print(f"Supports tools: {info.supports_tools}")

# Get latest model by capability
latest = get_latest_model("anthropic", "reasoning")
print(f"Latest reasoning model: {latest.id}")
```

## Architecture

Four-layer architecture:

1. **Layer 1 (Provider Spec)**: ProviderAdapter interface and data models
2. **Layer 2 (Provider Utils)**: HTTP client, SSE parser, retry logic
3. **Layer 3 (Core Client)**: Provider routing, middleware, configuration
4. **Layer 4 (High-level API)**: `generate()`, `stream()`, `generate_object()` (TODO)

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=unified_llm --cov-report=html

# Type check
uv run mypy src/unified_llm

# Run examples
export ANTHROPIC_API_KEY=sk-ant-...
python examples/basic_usage.py
```

## License

Apache License 2.0

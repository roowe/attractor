"""Model catalog with known LLM models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Information about an LLM model."""

    id: str
    provider: str
    display_name: str
    context_window: int
    max_output: int | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    aliases: list[str] | None = None


# Model catalog as of 2026
MODELS = [
    # ==========================================================
    # Anthropic -- Top quality: Claude Opus 4.6
    # ==========================================================
    ModelInfo(
        id="claude-opus-4-6",
        provider="anthropic",
        display_name="Claude Opus 4.6",
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
    ),
    ModelInfo(
        id="claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        aliases=["sonnet", "claude-sonnet"],
    ),

    # ==========================================================
    # OpenAI -- Top quality: GPT-5.2 series
    # ==========================================================
    ModelInfo(
        id="gpt-5.2",
        provider="openai",
        display_name="GPT-5.2",
        context_window=1047576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=150.0,
        output_cost_per_million=600.0,
    ),
    ModelInfo(
        id="gpt-5.2-mini",
        provider="openai",
        display_name="GPT-5.2 Mini",
        context_window=1047576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    ModelInfo(
        id="gpt-5.2-codex",
        provider="openai",
        display_name="GPT-5.2 Codex",
        context_window=1047576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),

    # ==========================================================
    # Gemini -- Latest: Gemini 3 Flash Preview
    # ==========================================================
    ModelInfo(
        id="gemini-2.5-pro",
        provider="gemini",
        display_name="Gemini 2.5 Pro",
        context_window=1048576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=1.25,
        output_cost_per_million=5.0,
    ),
    ModelInfo(
        id="gemini-2.5-flash",
        provider="gemini",
        display_name="Gemini 2.5 Flash",
        context_window=1048576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
        aliases=["flash", "gemini-flash"],
    ),

    # ==========================================================
    # Legacy models
    # ==========================================================
    ModelInfo(
        id="gpt-4.1",
        provider="openai",
        display_name="GPT-4.1",
        context_window=128000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=False,
        input_cost_per_million=2.50,
        output_cost_per_million=10.0,
    ),
    ModelInfo(
        id="claude-3.5-sonnet",
        provider="anthropic",
        display_name="Claude 3.5 Sonnet",
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=False,
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
    ),
]


def get_model_info(model_id: str) -> ModelInfo | None:
    """Get information about a model by ID or alias.

    Args:
        model_id: Model ID or alias.

    Returns:
        ModelInfo if found, None otherwise.
    """
    for model in MODELS:
        if model.id == model_id:
            return model
        if model.aliases and model_id in model.aliases:
            return model
    return None


def list_models(provider: str | None = None) -> list[ModelInfo]:
    """List all known models, optionally filtered by provider.

    Args:
        provider: Optional provider name to filter by.

    Returns:
        List of ModelInfo objects.
    """
    if provider:
        return [m for m in MODELS if m.provider == provider]
    return list(MODELS)


def get_latest_model(provider: str, capability: str | None = None) -> ModelInfo | None:
    """Get the latest/best model for a provider.

    Args:
        provider: Provider name ("anthropic", "openai", "gemini").
        capability: Optional capability filter ("reasoning", "vision", "tools").

    Returns:
        ModelInfo of the latest model, or None if no match.
    """
    models = list_models(provider)

    if capability:
        attr = f"supports_{capability}"
        models = [m for m in models if getattr(m, attr, False)]

    return models[0] if models else None

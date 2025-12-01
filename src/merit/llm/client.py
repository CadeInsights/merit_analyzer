"""LLM client factory."""

from merit.llm.base import LLMProvider
from merit.llm.config import SUPPORTED_COMBINATIONS, LLMConfig


def build_client(config: LLMConfig) -> LLMProvider:
    """Build an LLM provider from configuration.

    Args:
        config: LLMConfig with vendor settings

    Returns:
        Configured LLMProvider instance

    Raises:
        ValueError: If vendor combination is not supported
    """
    mv = config.model_vendor
    iv = config.inference_vendor

    if mv not in SUPPORTED_COMBINATIONS:
        raise ValueError(f"{mv} not supported. Available: {list(SUPPORTED_COMBINATIONS.keys())}")
    if iv not in SUPPORTED_COMBINATIONS[mv]:
        raise ValueError(f"{iv} not supported for {mv}. Available: {SUPPORTED_COMBINATIONS[mv]}")

    match mv, iv:
        case "openai", "openai":
            from openai import OpenAI

            from merit.llm.providers.openai import OpenAIProvider

            return OpenAIProvider(OpenAI())

        case "anthropic", "anthropic":
            from anthropic import Anthropic

            from merit.llm.providers.anthropic import AnthropicProvider

            return AnthropicProvider(Anthropic())

        case "anthropic", "aws":
            from anthropic import AnthropicBedrock

            from merit.llm.providers.anthropic import AnthropicProvider

            return AnthropicProvider(
                AnthropicBedrock(),
                default_model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                default_small_model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            )

        case "anthropic", "gcp":
            from anthropic import AnthropicVertex

            from merit.llm.providers.anthropic import AnthropicProvider

            if not config.project_id:
                raise ValueError("project_id required for GCP Vertex AI")

            region = config.region or "us-east5"
            return AnthropicProvider(
                AnthropicVertex(region=region, project_id=config.project_id),
                default_model="claude-sonnet-4-5@20250929",
                default_small_model="claude-haiku-4-5@20251001",
            )

        case _:
            raise ValueError(f"Unsupported combination: {mv}/{iv}")

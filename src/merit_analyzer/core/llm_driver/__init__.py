"""Handle LLM clients"""

import os
import asyncio
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from .openai_handler import LLMOpenAI
from .anthropic_handler import LLMClaude
from .abstract_provider_handler import LLMAbstractHandler
from .policies import AGENT, TOOL, FILE_ACCESS_POLICY

load_dotenv()

cached_client: Optional[LLMAbstractHandler] = None
cached_key: Optional[tuple[str, str]] = None
client_lock = asyncio.Lock()
validated_once = False

SUPPORTED = {
    "openai": ["openai"],
    "anthropic": ["anthropic", "gcp", "aws"],
}


async def build_llm_client(model_vendor: str, inference_vendor: str) -> LLMAbstractHandler:
    """Get the right LLM client based on model and vendor"""

    mv = model_vendor.lower().strip()
    ip = inference_vendor.lower().strip()

    if not mv:
        raise ValueError("MODEL_VENDOR has not been provided in ENV.")
    if not ip:
        raise ValueError("INFERENCE_VENDOR has not been provided in ENV.")

    match mv, ip:
        case "openai", "openai":
            from openai import OpenAI

            client = LLMOpenAI(OpenAI())

        case "anthropic", "aws":
            os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
            from anthropic import AnthropicBedrock

            client = LLMClaude(AnthropicBedrock())
            client.default_big_model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            client.default_small_model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

        case "anthropic", "gcp":
            os.environ["CLAUDE_CODE_USE_VERTEX"] = "1"
            from anthropic import AnthropicVertex

            region = os.getenv("CLOUD_ML_REGION", "us-east5")
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT or ANTHROPIC_VERTEX_PROJECT_ID must be set for Vertex AI")

            client = LLMClaude(AnthropicVertex(region=region, project_id=project_id))
            client.default_big_model = "claude-sonnet-4-5@20250929"
            client.default_small_model = "claude-haiku-4-5@20251001"

        case "anthropic", "anthropic":
            from anthropic import Anthropic

            client = LLMClaude(Anthropic())

        case _, _:
            if mv not in SUPPORTED:
                raise ValueError(f"{mv} is not supported yet. Available model families: {list(SUPPORTED.keys())}")
            if ip not in SUPPORTED[mv]:
                raise ValueError(f"{ip} is not supported for {mv}. Supported providers for {mv}: {SUPPORTED[mv]}")

    return client


async def validate_client(client: LLMAbstractHandler) -> None:
    """Health check."""

    class TestSchema(BaseModel):
        response: str

    await client.create_object(
        prompt='Return JSON: {"response": "True"}',
        schema=TestSchema,
    )


async def get_llm_client() -> LLMAbstractHandler:
    """
    Return a cached LLM client built from MODEL_VENDOR and INFERENCE_VENDOR.
    Rebuilds if envs changed.
    """
    global cached_client, cached_key, validated_once

    model_vendor = os.getenv("MODEL_VENDOR") or ""
    inference_vendor = os.getenv("INFERENCE_VENDOR") or ""
    key = (model_vendor.lower().strip(), inference_vendor.lower().strip())

    if cached_client is not None and cached_key == key:
        return cached_client

    async with client_lock:
        if cached_client is not None and cached_key == key:  # yes async is cursed
            return cached_client

        client = await build_llm_client(*key)

        if not validated_once:
            await validate_client(client)
            validated_once = True

        cached_client = client
        cached_key = key
        return cached_client


__all__ = ["AGENT", "TOOL", "FILE_ACCESS_POLICY", "get_llm_client", "LLMAbstractHandler"]

import asyncio
import inspect

import pytest

from merit_analyzer.core import llm_driver
from merit_analyzer.core.llm_driver import get_llm_client


def test_get_llm_client_is_exposed():
    assert inspect.iscoroutinefunction(get_llm_client)
    assert llm_driver.get_llm_client is get_llm_client


@pytest.mark.asyncio
async def test_get_llm_client_caches_single_configuration(monkeypatch):
    first_client = object()
    build_calls = []
    validations = []

    async def fake_build(model_vendor, inference_vendor):
        build_calls.append((model_vendor, inference_vendor))
        return first_client

    async def fake_validate(client):
        validations.append(client)

    monkeypatch.setenv("MODEL_VENDOR", "OpenAI")
    monkeypatch.setenv("INFERENCE_VENDOR", "OpenAI")
    monkeypatch.setattr(llm_driver, "build_llm_client", fake_build)
    monkeypatch.setattr(llm_driver, "validate_client", fake_validate)
    monkeypatch.setattr(llm_driver, "cached_client", None)
    monkeypatch.setattr(llm_driver, "cached_key", None)
    monkeypatch.setattr(llm_driver, "validated_once", False)
    monkeypatch.setattr(llm_driver, "client_lock", asyncio.Lock())

    first = await get_llm_client()
    second = await get_llm_client()

    assert first is second is first_client
    assert build_calls == [("openai", "openai")]
    assert validations == [first_client]


@pytest.mark.asyncio
async def test_get_llm_client_rebuilds_when_env_changes(monkeypatch):
    first_client = object()
    second_client = object()
    build_calls = []
    validations = []

    async def fake_build(model_vendor, inference_vendor):
        build_calls.append((model_vendor, inference_vendor))
        return [first_client, second_client][len(build_calls) - 1]

    async def fake_validate(client):
        validations.append(client)

    monkeypatch.setattr(llm_driver, "build_llm_client", fake_build)
    monkeypatch.setattr(llm_driver, "validate_client", fake_validate)
    monkeypatch.setattr(llm_driver, "cached_client", None)
    monkeypatch.setattr(llm_driver, "cached_key", None)
    monkeypatch.setattr(llm_driver, "validated_once", False)
    monkeypatch.setattr(llm_driver, "client_lock", asyncio.Lock())

    monkeypatch.setenv("MODEL_VENDOR", "OpenAI")
    monkeypatch.setenv("INFERENCE_VENDOR", "OpenAI")
    first = await get_llm_client()

    monkeypatch.setenv("MODEL_VENDOR", "Anthropic")
    monkeypatch.setenv("INFERENCE_VENDOR", "AWS")
    second = await get_llm_client()

    assert first is first_client
    assert second is second_client
    assert build_calls == [("openai", "openai"), ("anthropic", "aws")]
    assert validations == [first_client]

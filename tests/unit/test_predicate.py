import json
from pathlib import Path

import httpx
import pytest

from merit.context import TestContext as MeritTestContext, test_context_scope as context_scope
from merit.predicates.ai_predicates import (
    follows_policy,
    has_conflicting_facts,
    has_facts,
    has_topics,
    has_unsupported_facts,
    matches_facts,
    matches_writing_layout,
    matches_writing_style,
)
from merit.predicates.base import PredicateResult, predicate
from merit.predicates.client import (
    PredicateAPIClient,
    PredicateAPIFactory,
    PredicateAPISettings,
    PredicateType,
    close_predicate_api_client,
    create_predicate_api_client,
    get_predicate_api_client,
)
from merit.testing.discovery import TestItem


def _make_item(name: str = "merit_fn", id_suffix: str | None = None) -> TestItem:
    """Create a minimal TestItem for testing."""
    return TestItem(
        name=name,
        fn=lambda: None,
        module_path=Path("test.py"),
        is_async=False,
        id_suffix=id_suffix,
    )


def test_predicate_result_and_metadata_auto_filled():
    @predicate
    def simple_predicate(actual: str, reference: str, strict: bool = True):
        return actual == reference

    def merit_with_simple_predicate():
        result = simple_predicate("test", "test")

        # Basic properties
        assert result
        assert result.value is True
        assert result.confidence == 1.0

        predicate_metadata = result.predicate_metadata

        # Predicate metadata
        assert predicate_metadata.actual == "test"
        assert predicate_metadata.reference == "test"

        # Auto-filled identifiers
        assert predicate_metadata.predicate_name == "simple_predicate"
        assert predicate_metadata.merit_name == "merit_with_simple_predicate"

    with context_scope(MeritTestContext(item=_make_item("merit_with_simple_predicate"))):
        merit_with_simple_predicate()


@pytest.mark.asyncio
async def test_factory_get_reuses_client_and_aclose_resets() -> None:
    settings = PredicateAPISettings.model_validate(
        {
            "MERIT_API_BASE_URL": "https://example.com",
            "MERIT_API_KEY": "secret",
        }
    )
    factory = PredicateAPIFactory(settings=settings)

    client1 = await factory.get()
    client2 = await factory.get()

    assert client1 is client2
    assert factory._http is not None
    assert factory._http.is_closed is False

    await factory.aclose()

    assert factory._http is None
    assert factory._client is None

    client3 = await factory.get()
    assert client3 is not client1

    await factory.aclose()


@pytest.mark.asyncio
async def test_remote_predicate_client_check_posts_payload_and_parses_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            status_code=200,
            json={"passed": False, "confidence": 0.25, "reasoning": "nope"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://example.com/", transport=transport) as http:
        settings = PredicateAPISettings.model_validate(
            {
                "MERIT_API_BASE_URL": "https://example.com",
                "MERIT_API_KEY": "secret",
            }
        )
        client = PredicateAPIClient(http=http, settings=settings)

        from merit.predicates.client import PredicateAPIRequest, PredicateType

        result = await client.request_predicate(
            PredicateAPIRequest(
                actual="actual",
                reference="reference",
                assertion_type=PredicateType.FACTS_NOT_CONTRADICT,
                strict=False,
            )
        )

    assert captured["method"] == "POST"
    assert captured["path"] == "/assertions/evaluate"
    assert captured["json"] == {
        "actual": "actual",
        "reference": "reference",
        "assertion_type": "facts_not_contradict",
        "strict": False,
        "enable_reasoning": False,
        "request_id": None,
    }

    assert result.passed is False
    assert result.confidence == 0.25
    assert result.reasoning == "nope"


@pytest.mark.asyncio
async def test_module_level_get_and_close_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERIT_API_BASE_URL", "https://example.com")
    monkeypatch.setenv("MERIT_API_KEY", "secret")

    # Reset to ensure clean state (previous tests may have initialized)
    close_predicate_api_client()

    # Should raise error before initialization
    with pytest.raises(RuntimeError):
        await get_predicate_api_client()

    create_predicate_api_client()

    client1 = await get_predicate_api_client()
    client2 = await get_predicate_api_client()
    assert client1 is client2

    await close_predicate_api_client()

    with pytest.raises(RuntimeError):
        client3 = await get_predicate_api_client()
        assert client3 is not client1

    await close_predicate_api_client()


def test_predicate_decorator_supports_optional_kwargs():
    @predicate
    def equals(actual: str, reference: str):
        return actual == reference

    result = equals("test", "test")

    assert isinstance(result, PredicateResult)
    assert result.predicate_metadata.predicate_name == "equals"
    assert result.predicate_metadata.actual == "test"
    assert result.predicate_metadata.reference == "test"


@pytest.mark.asyncio
async def test_ai_predicates_call_api_correctly(monkeypatch) -> None:
    captured_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(payload)

        # Default response: passed=True for most, but we can customize logic if needed
        return httpx.Response(
            status_code=200,
            json={"passed": True, "confidence": 0.9, "reasoning": "all good"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://example.com/", transport=transport) as http:
        settings = PredicateAPISettings.model_validate(
            {"MERIT_API_BASE_URL": "https://example.com", "MERIT_API_KEY": "secret"}
        )
        client = PredicateAPIClient(http=http, settings=settings)

        # Use monkeypatch to return our mocked client
        import merit.predicates.ai_predicates as ai_p

        async def mock_get_client():
            return client

        monkeypatch.setattr(ai_p, "get_predicate_api_client", mock_get_client)

        # 1. has_conflicting_facts (value = not resp.passed)
        # If passed=True (no contradictions), value should be False
        res = await has_conflicting_facts("actual text", "reference text", strict=False)
        assert res.value is False
        assert captured_payloads[-1]["assertion_type"] == PredicateType.FACTS_NOT_CONTRADICT
        assert captured_payloads[-1]["actual"] == "actual text"
        assert captured_payloads[-1]["reference"] == "reference text"
        assert captured_payloads[-1]["strict"] is False

        # 2. has_unsupported_facts (value = not resp.passed)
        res = await has_unsupported_facts("a", "r")
        assert res.value is False
        assert captured_payloads[-1]["assertion_type"] == PredicateType.FACTS_SUPPORTED

        # 3. matches_facts (value = resp.passed)
        res = await matches_facts("a", "r")
        assert res.value is True
        assert captured_payloads[-1]["assertion_type"] == PredicateType.FACTS_FULL_MATCH

        # 4. follows_policy (value = resp.passed)
        res = await follows_policy("a", "r")
        assert res.value is True
        assert captured_payloads[-1]["assertion_type"] == PredicateType.CONDITIONS_MET

        # 5. matches_writing_layout (value = resp.passed)
        res = await matches_writing_layout("a", "r")
        assert res.value is True
        assert captured_payloads[-1]["assertion_type"] == PredicateType.STRUCTURE_MATCH

        # 6. matches_writing_style (value = resp.passed)
        res = await matches_writing_style("a", "r")
        assert res.value is True
        assert captured_payloads[-1]["assertion_type"] == PredicateType.STYLE_MATCH

        # 7. has_facts (value = resp.passed)
        res = await has_facts("a", "r")
        assert res.value is True
        assert captured_payloads[-1]["assertion_type"] == PredicateType.FACTS_NOT_MISSING

        # 8. has_topics (value = resp.passed)
        res = await has_topics("a", "r")
        assert res.value is True
        assert captured_payloads[-1]["assertion_type"] == PredicateType.HAS_TOPICS

import pytest
import json
import httpx
from uuid import UUID

from merit.predicates.base import PredicateResult, PredicateMetadata, predicate
from merit.predicates.client import (
    PredicateAPIClient,
    PredicateAPIFactory,
    PredicateAPISettings,
    close_predicate_api_client,
    get_predicate_api_client,
)


def simple_predicate(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> PredicateResult:
    return PredicateResult(
        value=actual == reference,
        message=None,
        predicate_metadata=PredicateMetadata(actual=actual, reference=reference, context=context, strict=strict),
        confidence=1.0,
    )


def test_predicate_result_and_metadata_auto_filled():
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

    merit_with_simple_predicate()


def test_predicate_metadata_auto_filled_no_merit_name():
    metadata = PredicateMetadata(actual="test", reference="test")

    # Auto-filled predicate_name parsed test function name
    assert metadata.predicate_name == "test_predicate_metadata_auto_filled_no_merit_name"
    assert metadata.merit_name is None


def test_predicate_actual_and_reference_truncated_in_repr():
    long_string_actual = "a" * 100
    long_string_reference = "b" * 100

    result = simple_predicate(long_string_actual, long_string_reference)
    parsed_result_back_to_json = json.loads(repr(result))

    # Truncated actual and reference in repr
    assert parsed_result_back_to_json["predicate_metadata"]["actual"] == long_string_actual[:50] + "..."
    assert parsed_result_back_to_json["predicate_metadata"]["reference"] == long_string_reference[:50] + "..."

    # Original actual and reference are not truncated
    assert result.predicate_metadata.actual == long_string_actual
    assert result.predicate_metadata.reference == long_string_reference


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
            json={"passed": False, "confidence": 0.25, "message": "nope"},
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

        result = await client.check(
            actual="actual",
            reference="reference",
            check="some-check",
            strict=False,
            context=None,
        )

    assert captured["method"] == "POST"
    assert captured["path"] == "/check"
    assert captured["json"] == {
        "actual": "actual",
        "reference": "reference",
        "check": "some-check",
        "strict": False,
    }

    assert result.passed is False
    assert result.confidence == 0.25
    assert result.message == "nope"


@pytest.mark.asyncio
async def test_module_level_get_and_close_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERIT_API_BASE_URL", "https://example.com")
    monkeypatch.setenv("MERIT_API_KEY", "secret")

    client1 = await get_predicate_api_client()
    client2 = await get_predicate_api_client()
    assert client1 is client2

    await close_predicate_api_client()

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


def test_predicate_decorator_supports_optional_kwargs_with_case_id():
    @predicate
    def equals(actual: str, reference: str):
        return actual == reference

    result = equals("test", "test", case_id=UUID("123e4567-e89b-12d3-a456-426614174000"))

    assert result.case_id == UUID("123e4567-e89b-12d3-a456-426614174000")

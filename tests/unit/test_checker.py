import pytest
import json
import httpx

from merit.checkers.base import CheckerResult, CheckerMetadata
from merit.checkers.client import (
    CheckerAPIClient,
    CheckerAPIFactory,
    CheckerAPISettings,
    close_checker_api_client,
    get_checker_api_client,
)


def simple_checker(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> CheckerResult:
    return CheckerResult(
        value=actual == reference,
        message=None,
        checker_metadata=CheckerMetadata(actual=actual, reference=reference, context=context, strict=strict),
        confidence=1.0,
    )


def test_checker_result_and_metadata_auto_filled():
    def merit_with_simple_checker():
        result = simple_checker("test", "test")

        # Basic properties
        assert result
        assert result.value is True
        assert result.confidence == 1.0

        checker_metadata = result.checker_metadata

        # Checker metadata
        assert checker_metadata.actual == "test"
        assert checker_metadata.reference == "test"

        # Auto-filled identifiers
        assert checker_metadata.checker_name == "simple_checker"
        assert checker_metadata.merit_name == "merit_with_simple_checker"

    merit_with_simple_checker()


def test_checker_metadata_auto_filled_no_merit_name():
    metadata = CheckerMetadata(actual="test", reference="test")

    # Auto-filled checker_name parsed test function name
    assert metadata.checker_name is "test_checker_metadata_auto_filled_no_merit_name"
    assert metadata.merit_name is None


def test_checker_actual_and_reference_truncated_in_repr():
    long_string_actual = "a" * 100
    long_string_reference = "b" * 100

    result = simple_checker(long_string_actual, long_string_reference)
    parsed_result_back_to_json = json.loads(repr(result))

    # Truncated actual and reference in repr
    assert parsed_result_back_to_json["checker_metadata"]["actual"] == long_string_actual[:50] + "..."
    assert parsed_result_back_to_json["checker_metadata"]["reference"] == long_string_reference[:50] + "..."

    # Original actual and reference are not truncated
    assert result.checker_metadata.actual == long_string_actual
    assert result.checker_metadata.reference == long_string_reference


@pytest.mark.asyncio
async def test_factory_get_reuses_client_and_aclose_resets() -> None:
    settings = CheckerAPISettings.model_validate(
        {
            "MERIT_API_BASE_URL": "https://example.com",
            "MERIT_API_KEY": "secret",
        }
    )
    factory = CheckerAPIFactory(settings=settings)

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
async def test_remote_checker_client_check_posts_payload_and_parses_response() -> None:
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
        settings = CheckerAPISettings.model_validate(
            {
                "MERIT_API_BASE_URL": "https://example.com",
                "MERIT_API_KEY": "secret",
            }
        )
        client = CheckerAPIClient(http=http, settings=settings)

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

    client1 = await get_checker_api_client()
    client2 = await get_checker_api_client()
    assert client1 is client2

    await close_checker_api_client()

    client3 = await get_checker_api_client()
    assert client3 is not client1

    await close_checker_api_client()
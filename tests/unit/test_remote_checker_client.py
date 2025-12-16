import json

import httpx
import pytest

from merit.checks.client import (
    RemoteCheckerClient,
    RemoteCheckerClientFactory,
    RemoteCheckerSettings,
    close_remote_checks_client,
    get_remote_checks_client,
)


@pytest.mark.asyncio
async def test_factory_get_reuses_client_and_aclose_resets() -> None:
    settings = RemoteCheckerSettings.model_validate(
        {
            "MERIT_API_BASE_URL": "https://example.com",
            "MERIT_API_KEY": "secret",
        }
    )
    factory = RemoteCheckerClientFactory(settings=settings)

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
        settings = RemoteCheckerSettings.model_validate(
            {
                "MERIT_API_BASE_URL": "https://example.com",
                "MERIT_API_KEY": "secret",
            }
        )
        client = RemoteCheckerClient(http=http, settings=settings)

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

    client1 = await get_remote_checks_client()
    client2 = await get_remote_checks_client()
    assert client1 is client2

    await close_remote_checks_client()

    client3 = await get_remote_checks_client()
    assert client3 is not client1

    await close_remote_checks_client()


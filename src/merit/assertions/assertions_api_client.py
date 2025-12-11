import httpx

from typing import Literal
from pydantic import BaseModel
from enum import Enum


class AssertionAPIRequest(BaseModel):
    actual: str
    reference: str
    check: str
    strict: bool = True
    with_context: str | None = None


class FactsCheckRequest(AssertionAPIRequest):
    check: Literal["contradictions", "supported", "full_match"]


class StyleCheckRequest(AssertionAPIRequest):
    check: Literal["match"]


class ConditionsCheckRequest(AssertionAPIRequest):
    check: Literal["met"]


class AssertionAPIResponse(BaseModel):
    passed: bool
    confidence: float
    message: str | None = None


class AssertionAPIRoute(Enum):
        FACTS_CHECK = "/facts"
        STYLE_CHECK = "/style"
        CONDITIONS_CHECK = "/conditions"


class AssertionAPIClient:
    """Client for asserting values using Merit Assertions API endpoints."""

    def __init__(self, base_url: str, token: str):
        self._base_url = base_url
        self._token = token
        self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=httpx.Timeout(5.0, connect=5.0),
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def get_assertion_result(self, path: AssertionAPIRoute, payload: FactsCheckRequest | StyleCheckRequest | ConditionsCheckRequest) -> AssertionAPIResponse:

        assert self._http_client is not None, "Client used outside context manager"

        response = await self._http_client.post(path.value, json=payload.model_dump_json())
        response.raise_for_status()

        return AssertionAPIResponse.model_validate(response.json())
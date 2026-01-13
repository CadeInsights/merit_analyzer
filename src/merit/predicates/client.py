from __future__ import annotations


"""HTTP client for calling the Merit remote predicate API."""

import asyncio
import logging
import random
from enum import Enum

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)

load_dotenv()


class PredicateType(str, Enum):
    """Types of assertions that can be evaluated."""

    FACTS_NOT_CONTRADICT = "facts_not_contradict"
    FACTS_SUPPORTED = "facts_supported"
    FACTS_FULL_MATCH = "facts_full_match"
    FACTS_NOT_MISSING = "facts_not_missing"
    CONDITIONS_MET = "conditions_met"
    STYLE_MATCH = "style_match"
    STRUCTURE_MATCH = "structure_match"
    HAS_TOPICS = "has_topics"


class PredicateAPIRequest(BaseModel):
    """Request payload sent to the remote predicate service."""

    assertion_type: PredicateType
    actual: str
    reference: str
    strict: bool = True
    enable_reasoning: bool = False
    request_id: str | None = None


class PredicateAPIResponse(BaseModel):
    """Response payload returned by the remote predicate service."""

    passed: bool
    confidence: float
    reasoning: str | None = None


class PredicateAPISettings(BaseSettings):
    """Configuration for `RemotePredicateClient`.

    Environment variables are read without a prefix.

    Attributes:
    ----------
    base_url
        Service base URL (from ``MERIT_API_BASE_URL``).
    api_key
        Bearer token (from ``MERIT_API_KEY``).
    connect_timeout, read_timeout, write_timeout, pool_timeout
        httpx timeouts in seconds.
    max_connections, max_keepalive_connections
        Connection pool limits for the underlying `httpx.AsyncClient`.
    keepalive_expiry
        Close idle keep-alive connections after this many seconds
        (from ``MERIT_API_KEEPALIVE_EXPIRY``).
    retry_max_attempts
        Maximum number of attempts for a single request.
    retry_base_delay_s, retry_max_delay_s, retry_jitter_s
        Exponential backoff parameters (seconds).
    retry_status_codes
        Status codes that trigger a retry.
    retry_on_server_errors
        Whether 5xx responses should be retried.
    """

    base_url: HttpUrl = Field(validation_alias="MERIT_API_BASE_URL")
    api_key: SecretStr = Field(validation_alias="MERIT_API_KEY")
    debugging_mode: bool = Field(default=False, validation_alias="MERIT_DEBUGGING_MODE")
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    pool_timeout: float = 5.0
    max_connections: int = 200
    max_keepalive_connections: int = 50
    keepalive_expiry: float = Field(default=30.0, validation_alias="MERIT_API_KEEPALIVE_EXPIRY")
    retry_max_attempts: int = 4
    retry_base_delay_s: float = 0.05
    retry_max_delay_s: float = 1.0
    retry_jitter_s: float = 0.05
    retry_status_codes: list[int] = Field(default_factory=lambda: [408, 429])
    retry_on_server_errors: bool = True

    model_config = SettingsConfigDict(
        extra="forbid",
        env_prefix="",
    )


class PredicateAPIClient:
    """Thin wrapper around an httpx.AsyncClient."""

    def __init__(self, http: httpx.AsyncClient, settings: PredicateAPISettings) -> None:
        """Initialize the client.

        Parameters
        ----------
        http
            Pre-configured async HTTP client used to issue requests.
        settings
            Retry and timeout configuration.
        """
        self._http = http
        self._settings = settings

    async def request_predicate(self, request: PredicateAPIRequest) -> PredicateAPIResponse:
        """Run a remote check against the configured service.

        Parameters
        ----------
        actual
            Model output to validate.
        reference
            Reference output / expected content.
        check
            Check identifier understood by the remote API.
        strict
            Whether to enforce strict checking semantics.

        Returns:
        -------
        PredicateAPIResponse
            Parsed response returned by the service.

        Raises:
        ------
        httpx.HTTPError
            If the request ultimately fails or returns a non-success status.
        """
        s = self._settings

        if s.debugging_mode:
            request.enable_reasoning = True

        payload = request.model_dump()

        for attempt in range(s.retry_max_attempts):
            try:
                resp = await self._http.post("assertions/evaluate", json=payload)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == s.retry_max_attempts - 1:
                    raise
                delay_s = min(
                    s.retry_max_delay_s, s.retry_base_delay_s * (2**attempt)
                ) + random.uniform(0, s.retry_jitter_s)
                await asyncio.sleep(delay_s)
                continue

            should_retry = resp.status_code in s.retry_status_codes or (
                s.retry_on_server_errors and resp.status_code >= 500
            )
            if should_retry:
                if attempt == s.retry_max_attempts - 1:
                    await resp.aread()
                    resp.raise_for_status()
                await resp.aread()
                delay_s = min(
                    s.retry_max_delay_s, s.retry_base_delay_s * (2**attempt)
                ) + random.uniform(0, s.retry_jitter_s)
                await asyncio.sleep(delay_s)
                continue

            resp.raise_for_status()
            final_response = PredicateAPIResponse.model_validate(resp.json())

            if s.debugging_mode:
                logger.info(f"Predicate response: {resp.json()}")

            return final_response

        raise RuntimeError("PredicateAPIClient.check exhausted retries")


class PredicateAPIFactory:
    """Lazy, reusable factory for `PredicateAPIClient`.

    The factory owns a single underlying `httpx.AsyncClient` and returns a
    shared `PredicateAPIClient` instance while the HTTP client remains open.
    """

    def __init__(self, settings: PredicateAPISettings | None = None) -> None:
        """Create a factory.

        Parameters
        ----------
        settings
            Optional settings override. If omitted, settings are loaded from the
            environment via `PredicateAPISettings`.
        """
        self._settings = settings or PredicateAPISettings()  # type: ignore[call-arg]
        self._lock = asyncio.Lock()
        self._http: httpx.AsyncClient | None = None
        self._client: PredicateAPIClient | None = None

    async def aclose(self) -> None:
        """Close the underlying `httpx.AsyncClient` (if any) and reset state."""
        async with self._lock:
            if self._http and not self._http.is_closed:
                await self._http.aclose()
            self._http = None
            self._client = None

    async def get(self) -> PredicateAPIClient:
        """Return a shared `PredicateAPIClient`, creating it if needed.

        Returns:
        -------
        PredicateAPIClient
            A client backed by a shared `httpx.AsyncClient` connection pool.
        """
        http = self._http
        client = self._client
        if client is not None and http is not None and not http.is_closed:
            return client

        async with self._lock:
            http = self._http
            client = self._client
            if client is not None and http is not None and not http.is_closed:
                return client

            if self._http is None or self._http.is_closed:
                s = self._settings
                base_url = str(s.base_url).rstrip("/") + "/"
                self._http = httpx.AsyncClient(
                    base_url=base_url,
                    headers={"Authorization": f"Bearer {s.api_key.get_secret_value()}"},
                    timeout=httpx.Timeout(
                        connect=s.connect_timeout,
                        read=s.read_timeout,
                        write=s.write_timeout,
                        pool=s.pool_timeout,
                    ),
                    limits=httpx.Limits(
                        max_connections=s.max_connections,
                        max_keepalive_connections=s.max_keepalive_connections,
                        keepalive_expiry=s.keepalive_expiry,
                    ),
                )
                self._client = PredicateAPIClient(self._http, settings=s)

            if self._client is None:
                raise RuntimeError("PredicateAPIFactory failed to initialize")

            return self._client


# Module-level default client helpers
_default_factory: PredicateAPIFactory | None = None


def create_predicate_api_client(settings: PredicateAPISettings | None = None) -> None:
    """Initialize the global PredicateAPIClient factory."""
    global _default_factory
    _default_factory = PredicateAPIFactory(settings=settings)


async def get_predicate_api_client() -> PredicateAPIClient:
    """Return a process-wide shared PredicateAPIClient."""
    if _default_factory is None:
        raise RuntimeError(
            "Predicate API client not initialized. Call create_predicate_api_client() first."
        )
    return await _default_factory.get()


async def close_predicate_api_client() -> None:
    """Close the shared client pool."""
    global _default_factory
    if _default_factory is None:
        return
    await _default_factory.aclose()

    _default_factory = None

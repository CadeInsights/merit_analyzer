from __future__ import annotations

"""HTTP client for calling the Merit remote checker API."""

import asyncio
import random

import httpx
from pydantic import BaseModel, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class CheckerAPIRequest(BaseModel):
    """Request payload sent to the remote checker service.

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
    context
        Optional extra context provided to the checker.
    """

    actual: str
    reference: str
    check: str  # TODO: build enum when API design is finalized
    strict: bool = True
    context: str | None = None


class CheckerAPIResponse(BaseModel):
    """Response payload returned by the remote checker service.

    Attributes
    ----------
    passed
        Whether the check succeeded.
    confidence
        Confidence score in the range provided by the service.
    message
        Optional human-readable explanation.
    """

    passed: bool
    confidence: float
    message: str | None = None


class CheckerAPISettings(BaseSettings):
    """Configuration for `RemoteCheckerClient`.

    Environment variables are read without a prefix.

    Attributes
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


class CheckerAPIClient:
    """Thin wrapper around an httpx.AsyncClient."""

    def __init__(self, http: httpx.AsyncClient, settings: CheckerAPISettings) -> None:
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

    async def check(
        self,
        actual: str,
        reference: str,
        check: str,
        strict: bool = True,
        context: str | None = None,
    ) -> CheckerAPIResponse:
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
        context
            Optional extra context provided to the checker.

        Returns
        -------
        CheckerAPIResponse
            Parsed response returned by the service.

        Raises
        ------
        httpx.HTTPError
            If the request ultimately fails or returns a non-success status.
        """

        payload = CheckerAPIRequest(
            actual=actual,
            reference=reference,
            check=check,
            strict=strict,
            context=context,
        ).model_dump(exclude_none=True)

        s = self._settings

        for attempt in range(s.retry_max_attempts):
            try:
                resp = await self._http.post("check", json=payload)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == s.retry_max_attempts - 1:
                    raise
                delay_s = min(s.retry_max_delay_s, s.retry_base_delay_s * (2**attempt)) + random.uniform(
                    0, s.retry_jitter_s
                )
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
                delay_s = min(s.retry_max_delay_s, s.retry_base_delay_s * (2**attempt)) + random.uniform(
                    0, s.retry_jitter_s
                )
                await asyncio.sleep(delay_s)
                continue

            resp.raise_for_status()
            return CheckerAPIResponse.model_validate(resp.json())

        raise RuntimeError("CheckerAPIClient.check exhausted retries")


class CheckerAPIFactory:
    """Lazy, reusable factory for `CheckerAPIClient`.

    The factory owns a single underlying `httpx.AsyncClient` and returns a
    shared `CheckerAPIClient` instance while the HTTP client remains open.
    """

    def __init__(self, settings: CheckerAPISettings | None = None) -> None:
        """Create a factory.

        Parameters
        ----------
        settings
            Optional settings override. If omitted, settings are loaded from the
            environment via `CheckerAPISettings`.
        """

        self._settings = settings or CheckerAPISettings()  # type: ignore[call-arg]
        self._lock = asyncio.Lock()
        self._http: httpx.AsyncClient | None = None
        self._client: CheckerAPIClient | None = None

    async def aclose(self) -> None:
        """Close the underlying `httpx.AsyncClient` (if any) and reset state."""

        async with self._lock:
            if self._http and not self._http.is_closed:
                await self._http.aclose()
            self._http = None
            self._client = None

    async def get(self) -> CheckerAPIClient:
        """Return a shared `CheckerAPIClient`, creating it if needed.

        Returns
        -------
        CheckerAPIClient
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
                self._client = CheckerAPIClient(self._http, settings=s)

            if self._client is None:
                raise RuntimeError("CheckerAPIFactory failed to initialize")

            return self._client


# Module-level default client helpers
_default_factory: CheckerAPIFactory | None = None


async def get_checker_api_client() -> CheckerAPIClient:
    """Return a process-wide shared CheckerAPIClient (lazy init)."""

    global _default_factory
    if _default_factory is None:
        _default_factory = CheckerAPIFactory()
    return await _default_factory.get()


# TODO: fix leaking client. httpx closes TCP on idle, but still not cool
async def close_checker_api_client() -> None:
    """Close the shared client pool and reset the default factory."""

    global _default_factory
    if _default_factory is None:
        return
    await _default_factory.aclose()
    _default_factory = None

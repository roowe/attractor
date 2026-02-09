"""HTTP client utility for provider adapters."""

from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Any

import httpx

from unified_llm._exceptions import (
    RequestTimeoutError,
    NetworkError,
)


class HttpMethod(str, Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass(frozen=True)
class HttpResponse:
    """HTTP response wrapper."""

    status_code: int
    headers: dict[str, str]
    content: bytes
    _client_response: httpx.Response | None = None

    def text(self) -> str:
        """Get response text."""
        return self.content.decode("utf-8")

    def json(self) -> Any:
        """Parse response as JSON.

        Raises:
            ValueError: If response is not valid JSON.
        """
        import json
        try:
            return json.loads(self.content)
        except json.JSONDecodeError as e:
            # Provide more helpful error message
            text_preview = self.text()[:200] if self.content else "(empty response)"
            raise ValueError(
                f"Failed to parse JSON response (status {self.status_code}). "
                f"Response preview: {text_preview!r}"
            ) from e

    @property
    def is_success(self) -> bool:
        """Check if response status code indicates success (2xx)."""
        return 200 <= self.status_code < 300

    @property
    def is_error(self) -> bool:
        """Check if response status code indicates error (4xx or 5xx)."""
        return self.status_code >= 400


class HttpClient:
    """Async HTTP client wrapper around httpx."""

    def __init__(
        self,
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
        limits: httpx.Limits | None = None,
    ):
        """Initialize HTTP client.

        Args:
            timeout: Request timeout in seconds.
            connect_timeout: Connection timeout in seconds.
            limits: Connection pool limits.
        """
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=timeout,
            write=timeout,
            pool=timeout,
        )
        self._limits = limits or httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        )
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            limits=self._limits,
        )

    async def request(
        self,
        method: HttpMethod,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        content: bytes | None = None,
    ) -> HttpResponse:
        """Make an HTTP request.

        Args:
            method: HTTP method.
            url: Request URL.
            headers: Request headers.
            params: Query parameters.
            json: JSON body.
            content: Raw body bytes.

        Returns:
            HttpResponse wrapper.

        Raises:
            NetworkError: On network errors.
            RequestTimeoutError: On timeout.
        """
        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json,
                content=content,
            )
            return HttpResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=response.content,
                _client_response=response,
            )

        except httpx.TimeoutException as e:
            raise RequestTimeoutError(f"Request timeout: {e}") from e

        except httpx.NetworkError as e:
            raise NetworkError(f"Network error: {e}", cause=e) from e

    async def stream(
        self,
        method: HttpMethod,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream an HTTP response.

        Args:
            method: HTTP method.
            url: Request URL.
            headers: Request headers.
            params: Query parameters.
            json: JSON body.

        Yields:
            Response chunks as bytes.

        Raises:
            NetworkError: On network errors.
            RequestTimeoutError: On timeout.
        """
        try:
            async with self._client.stream(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

        except httpx.TimeoutException as e:
            raise RequestTimeoutError(f"Stream timeout: {e}") from e

        except httpx.NetworkError as e:
            raise NetworkError(f"Network error: {e}", cause=e) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()

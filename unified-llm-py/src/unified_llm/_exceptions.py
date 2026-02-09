"""Exception hierarchy for the unified LLM client."""


class SDKError(Exception):
    """Base exception for all SDK errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


class ProviderError(SDKError):
    """Error from an LLM provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        raw: dict | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw = raw


class AuthenticationError(ProviderError):
    """401: Invalid API key or expired token."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class AccessDeniedError(ProviderError):
    """403: Insufficient permissions."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class NotFoundError(ProviderError):
    """404: Model or endpoint not found."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class InvalidRequestError(ProviderError):
    """400/422: Malformed request or invalid parameters."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class RateLimitError(ProviderError):
    """429: Rate limit exceeded."""

    def __init__(self, message: str, *, provider: str, retry_after: float | None = None, **kwargs):
        super().__init__(message, provider=provider, retryable=True, retry_after=retry_after, **kwargs)


class ServerError(ProviderError):
    """500-599: Provider internal error."""

    def __init__(self, message: str, *, provider: str, status_code: int, **kwargs):
        super().__init__(message, provider=provider, status_code=status_code, retryable=True, **kwargs)


class ContentFilterError(ProviderError):
    """Response blocked by safety/content filter."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class ContextLengthError(ProviderError):
    """413: Input + output exceeds context window."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class QuotaExceededError(ProviderError):
    """Billing/quota limit exceeded."""

    def __init__(self, message: str, *, provider: str, **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


# Non-provider errors


class RequestTimeoutError(SDKError):
    """Request or stream timeout."""

    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = True


class AbortError(SDKError):
    """Request cancelled via abort signal."""

    def __init__(self, message: str = "Request was aborted"):
        super().__init__(message)
        self.retryable = False


class NetworkError(SDKError):
    """Network-level failure."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, cause=cause)
        self.retryable = True


class StreamError(SDKError):
    """Error during stream consumption."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, cause=cause)
        self.retryable = True


class InvalidToolCallError(SDKError):
    """Tool call parameter validation failed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = False


class NoObjectGeneratedError(SDKError):
    """Structured output parsing/validation failed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = False


class ConfigurationError(SDKError):
    """SDK configuration error."""

    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = False

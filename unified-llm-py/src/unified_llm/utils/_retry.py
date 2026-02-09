"""Retry logic with exponential backoff."""

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from unified_llm._exceptions import SDKError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy configuration."""

    max_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    on_retry: Callable[[SDKError, int, float], Awaitable[None]] | None = None


async def retry(
    func: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy = RetryPolicy(),
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry.
        policy: Retry policy configuration.

    Returns:
        Result from successful function call.

    Raises:
        The last error if all retries exhausted.
    """
    last_error: SDKError | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await func()

        except SDKError as e:
            last_error = e

            # Check if error is retryable
            retryable = getattr(e, "retryable", False)
            if not retryable:
                raise

            # Check if we should retry
            if attempt >= policy.max_retries:
                raise

            # Calculate delay
            retry_after = getattr(e, "retry_after", None)
            if retry_after is not None:
                # Provider specified retry delay
                if retry_after > policy.max_delay:
                    # Provider asked to wait too long
                    raise
                delay = retry_after
            else:
                # Exponential backoff
                delay = min(
                    policy.base_delay * (policy.backoff_multiplier ** attempt),
                    policy.max_delay,
                )

                # Add jitter
                if policy.jitter:
                    delay = delay * random.uniform(0.5, 1.5)

            # Call callback if provided
            if policy.on_retry:
                await policy.on_retry(e, attempt + 1, delay)

            # Wait before retry
            await asyncio.sleep(delay)

    # Should never reach here, but for type safety
    raise last_error or SDKError("Retry failed with no error")

"""Server-Sent Events (SSE) parser."""

from typing import AsyncIterator, Tuple


async def parse_sse(
    source: bytes | AsyncIterator[bytes],
) -> AsyncIterator[Tuple[str, str]]:
    """Parse Server-Sent Events.

    Args:
        source: Either bytes or an async iterator of bytes chunks.

    Yields:
        Tuples of (event_type, data). Event type defaults to "message"
        if not specified.

    Example:
        >>> async for event_type, data in parse_sse(response):
        ...     print(f"{event_type}: {data}")
    """
    buffer = b""
    event_type = "message"
    data_lines = []

    if isinstance(source, bytes):
        chunks = _iter_bytes([source])
    else:
        chunks = source

    async for chunk in chunks:
        buffer += chunk

        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line_str = line.decode("utf-8").rstrip("\r")

            # Empty line = end of event
            if not line_str:
                if data_lines:
                    yield (event_type, "\n".join(data_lines))
                    event_type = "message"
                    data_lines = []
                continue

            # Comment line
            if line_str.startswith(":"):
                continue

            # Parse field
            if ":" in line_str:
                field, value = line_str.split(":", 1)
                value = value.lstrip()

                if field == "event":
                    event_type = value
                elif field == "data":
                    data_lines.append(value)
                elif field == "retry":
                    # Could be used to set reconnection delay
                    pass
            else:
                # Treat line as data field with no value
                data_lines.append(line_str)

    # Emit any remaining event
    if data_lines:
        yield (event_type, "\n".join(data_lines))


async def _iter_bytes(chunks: list[bytes]) -> AsyncIterator[bytes]:
    """Convert list of bytes to async iterator."""
    for chunk in chunks:
        yield chunk

"""Anthropic provider adapter using the Messages API."""

import os
from typing import AsyncIterator

from unified_llm._exceptions import (
    AuthenticationError,
    AccessDeniedError,
    NotFoundError,
    InvalidRequestError,
    RateLimitError,
    ServerError,
    ContextLengthError,
)
from unified_llm.models import (
    Request,
    Response,
    StreamEvent,
    Message,
    Usage,
    FinishReason,
    ContentPart,
    Role,
    ContentKind,
)
from unified_llm.models._content import ToolCallData
from unified_llm.models._response import RateLimitInfo
from unified_llm.providers._base import ProviderAdapter
from unified_llm.utils._http import HttpClient


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic's Claude API using the Messages API.

    Uses Anthropic's native Messages API (/v1/messages) for full feature support:
    - Prompt caching with cache_control blocks
    - Extended thinking with thinking blocks
    - Tool calling with tool_use and tool_result
    - Vision/image support
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = 120.0,
    ):
        """Initialize Anthropic adapter.

        Args:
            api_key: Anthropic API key.
            base_url: Optional custom base URL.
            default_headers: Optional default HTTP headers.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or
                         "https://api.anthropic.com/v1")
        self._default_headers = default_headers or {}
        self._timeout = timeout
        self._client = HttpClient(timeout=timeout)

    async def complete(self, request: Request) -> Response:
        """Send request, block until complete, return full response."""
        # Transform unified request to Anthropic format
        anthropic_request = self._transform_request(request)

        headers = self._get_headers(request)

        try:
            http_response = await self._client.request(
                method="POST",
                url=f"{self._base_url}/messages",  # Note: some providers might need /v1/messages
                headers=headers,
                json=anthropic_request,
            )

            # Check for HTTP errors
            self._check_http_error(http_response)

            # Transform Anthropic response to unified format
            return self._transform_response(http_response, request)

        except Exception as e:
            if isinstance(e, (AuthenticationError, RateLimitError, ServerError, InvalidRequestError)):
                raise
            # Wrap unknown errors
            raise ServerError(
                f"Anthropic API error: {e}",
                provider=self.name,
                status_code=getattr(e, "status_code", 500),
            ) from e

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send request, return async iterator of stream events."""
        anthropic_request = self._transform_request(request, stream=True)
        headers = self._get_headers(request)

        yield StreamEvent.stream_start()

        try:
            async for chunk in self._client.stream(
                method="POST",
                url=f"{self._base_url}/messages",
                headers=headers,
                json=anthropic_request,
            ):
                # Parse SSE chunk
                async for event_type, data in self._parse_sse_chunk(chunk):
                    # Transform Anthropic event to unified format
                    unified_event = self._transform_stream_event(event_type, data)
                    if unified_event:
                        yield unified_event

        except Exception as e:
            yield StreamEvent.error(e)

    def _get_headers(self, request: Request) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            **self._default_headers,
        }

        # Add beta headers for features
        if provider_opts := request.provider_options:
            if anthropic_opts := provider_opts.get("anthropic"):
                if beta_headers := anthropic_opts.get("beta_headers"):
                    headers["anthropic-beta"] = ",".join(beta_headers)

        return headers

    def _transform_request(self, request: Request, stream: bool = False) -> dict:
        """Transform unified request to Anthropic format."""
        # Extract system message
        system_message = ""
        messages = []

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                # Append to system message
                system_message += msg.text
            elif msg.role == Role.DEVELOPER:
                # Merge with system
                system_message += msg.text
            elif msg.role == Role.TOOL:
                # Tool result - add to next user message
                tool_result_content = self._transform_tool_result(msg.content[0].tool_result)
                messages.append({
                    "role": "user",
                    "content": [tool_result_content],
                })
            elif msg.role == Role.ASSISTANT:
                # Assistant message - may contain tool_use or text
                messages.append({
                    "role": "assistant",
                    "content": self._transform_assistant_content(msg.content),
                })
            else:  # USER
                messages.append({
                    "role": "user",
                    "content": self._transform_user_content(msg.content),
                })

        anthropic_request = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "stream": stream,
        }

        if system_message:
            anthropic_request["system"] = system_message

        # Add optional parameters
        if request.temperature is not None:
            anthropic_request["temperature"] = request.temperature
        if request.top_p is not None:
            anthropic_request["top_p"] = request.top_p
        if request.stop_sequences:
            anthropic_request["stop_sequences"] = request.stop_sequences

        # Add tools
        if request.tools:
            anthropic_request["tools"] = [
                self._transform_tool(tool) for tool in request.tools
            ]

            # Add tool_choice
            if request.tool_choice:
                anthropic_request["tool_choice"] = self._transform_tool_choice(
                    request.tool_choice
                )

        return anthropic_request

    def _transform_user_content(self, content: list[ContentPart]) -> list[dict]:
        """Transform user content to Anthropic format."""
        result = []
        for part in content:
            if part.kind == ContentKind.TEXT:
                result.append({"type": "text", "text": part.text})
            elif part.kind == ContentKind.IMAGE:
                if part.image.url:
                    result.append({
                        "type": "image",
                        "source": {"type": "url", "url": part.image.url},
                    })
                else:
                    result.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.image.media_type or "image/png",
                            "data": part.image.data,
                        },
                    })
        return result

    def _transform_assistant_content(self, content: list[ContentPart]) -> list[dict]:
        """Transform assistant content to Anthropic format."""
        result = []
        for part in content:
            if part.kind == ContentKind.TEXT:
                result.append({"type": "text", "text": part.text})
            elif part.kind == ContentKind.TOOL_CALL:
                result.append({
                    "type": "tool_use",
                    "id": part.tool_call.id,
                    "name": part.tool_call.name,
                    "input": part.tool_call.arguments,
                })
            elif part.kind == ContentKind.THINKING:
                result.append({
                    "type": "thinking",
                    "thinking": part.thinking.text,
                    "signature": part.thinking.signature or "",
                })
        return result

    def _transform_tool_result(self, tool_result) -> dict:
        """Transform tool result to Anthropic format."""
        return {
            "type": "tool_result",
            "tool_use_id": tool_result.tool_call_id,
            "content": str(tool_result.content),
            "is_error": tool_result.is_error,
        }

    def _transform_tool(self, tool) -> dict:
        """Transform tool definition to Anthropic format."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _transform_tool_choice(self, tool_choice) -> dict | str:
        """Transform tool choice to Anthropic format."""
        if tool_choice.mode == "auto":
            return {"type": "auto"}
        elif tool_choice.mode == "none":
            return "any"  # Anthropic doesn't have "none", use "any" with no tools
        elif tool_choice.mode == "required":
            return {"type": "any"}
        elif tool_choice.mode == "named":
            return {"type": "tool", "name": tool_choice.tool_name}
        return "auto"

    def _transform_response(self, http_response, request: Request) -> Response:
        """Transform Anthropic response to unified format."""
        data = http_response.json()

        # Extract content
        content_parts = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content_parts.append(ContentPart.text(block["text"]))
            elif block["type"] == "tool_use":
                content_parts.append(
                    ContentPart.tool_call(
                        id=block["id"],
                        name=block["name"],
                        arguments=block["input"],
                    )
                )

        # Create message
        message = Message(
            role=Role.ASSISTANT,
            content=content_parts,
        )

        # Extract usage
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_input_tokens"),
            cache_write_tokens=usage_data.get("cache_creation_input_tokens"),
        )

        # Map finish reason
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = self._map_finish_reason(stop_reason)

        return Response(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            provider=self.name,
            message=message,
            finish_reason=FinishReason(reason=finish_reason, raw=stop_reason),
            usage=usage,
            raw=data,
        )

    def _map_finish_reason(self, anthropic_reason: str) -> str:
        """Map Anthropic finish reason to unified reason."""
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }
        return mapping.get(anthropic_reason, "other")

    async def _parse_sse_chunk(self, chunk: bytes) -> AsyncIterator[tuple[str, str]]:
        """Parse SSE chunk from Anthropic."""
        from unified_llm.utils._sse import parse_sse
        async for event_type, data in parse_sse(chunk):
            yield event_type, data

    def _transform_stream_event(self, event_type: str, data: str) -> StreamEvent | None:
        """Transform Anthropic stream event to unified format."""
        if event_type == "message_start":
            return None  # Ignore, already sent stream_start
        elif event_type == "content_block_start":
            # New content block starting
            return None
        elif event_type == "content_block_delta":
            # Incremental content
            import json
            delta = json.loads(data)
            if delta.get("type") == "text_delta":
                return StreamEvent.text_delta(delta.get("text", ""))
        elif event_type == "message_delta":
            # End of message with usage
            import json
            delta = json.loads(data)
            usage_data = delta.get("usage", {})
            stop_reason = delta.get("stop_reason", "end_turn")
            usage = Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                total_tokens=0,
            )
            finish_reason = self._map_finish_reason(stop_reason)
            return StreamEvent.finish(
                finish_reason=FinishReason(reason=finish_reason, raw=stop_reason),
                usage=usage,
            )
        return None

    def _check_http_error(self, http_response) -> None:
        """Check HTTP status and raise appropriate error.

        Args:
            http_response: The HTTP response from the API.

        Raises:
            AuthenticationError: For 401 status.
            RateLimitError: For 429 status.
            ServerError: For 5xx status.
            InvalidRequestError: For 4xx errors.
        """
        status = http_response.status_code

        if status == 401:
            raise AuthenticationError(
                "Invalid API key",
                provider=self.name,
                status_code=status,
            )
        elif status == 403:
            raise AccessDeniedError(
                "Access denied",
                provider=self.name,
                status_code=status,
            )
        elif status == 404:
            raise NotFoundError(
                "Model not found",
                provider=self.name,
                status_code=status,
            )
        elif status in (400, 422):
            raise InvalidRequestError(
                "Invalid request",
                provider=self.name,
                status_code=status,
            )
        elif status == 413:
            raise ContextLengthError(
                "Context length exceeded",
                provider=self.name,
                status_code=status,
            )
        elif status == 429:
            raise RateLimitError(
                "Rate limit exceeded",
                provider=self.name,
                status_code=status,
            )
        elif status >= 500:
            raise ServerError(
                f"Server error (status {status})",
                provider=self.name,
                status_code=status,
            )
        elif status >= 400:
            # Catch-all for other 4xx errors
            raise InvalidRequestError(
                f"Request failed with status {status}",
                provider=self.name,
                status_code=status,
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()

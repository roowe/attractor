"""OpenAI provider adapter using the Responses API or Chat Completions API."""

import os
from typing import AsyncIterator, Literal

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
from unified_llm.providers._base import ProviderAdapter
from unified_llm.utils._http import HttpClient


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI using the Responses API or Chat Completions API.

    Supports two API formats:
    - "responses": OpenAI's Responses API (/v1/responses) for full feature support
      including reasoning tokens, built-in tools, server-side state, vision support
    - "chat_completions": Traditional Chat Completions API (/v1/chat/completions)
      for compatibility with OpenAI-compatible providers like MiniMax, DeepSeek, etc.
    """

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = 120.0,
        api_format: Literal["responses", "chat_completions"] = "responses",
    ):
        """Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key.
            base_url: Optional custom base URL.
            default_headers: Optional default HTTP headers.
            timeout: Request timeout in seconds.
            api_format: API format to use - "responses" for OpenAI Responses API,
                       "chat_completions" for Chat Completions API (default: "responses").
        """
        self._api_key = api_key
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or
                         "https://api.openai.com/v1")
        self._default_headers = default_headers or {}
        self._timeout = timeout
        self._api_format = api_format
        self._client = HttpClient(timeout=timeout)

    async def complete(self, request: Request) -> Response:
        """Send request, block until complete, return full response."""
        openai_request = self._transform_request(request)
        headers = self._get_headers(request)

        # Choose endpoint based on API format
        endpoint = "/chat/completions" if self._api_format == "chat_completions" else "/responses"

        try:
            http_response = await self._client.request(
                method="POST",
                url=f"{self._base_url}{endpoint}",
                headers=headers,
                json=openai_request,
            )

            # Check HTTP status
            self._check_http_error(http_response.status_code, http_response)

            # Transform OpenAI response to unified format
            return self._transform_response(http_response, request)

        except Exception as e:
            if isinstance(e, (AuthenticationError, RateLimitError, ServerError)):
                raise
            raise ServerError(
                f"OpenAI API error: {e}",
                provider=self.name,
                status_code=500,
            ) from e

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send request, return async iterator of stream events."""
        openai_request = self._transform_request(request, stream=True)
        headers = self._get_headers(request)

        # Choose endpoint based on API format
        endpoint = "/chat/completions" if self._api_format == "chat_completions" else "/responses"

        yield StreamEvent.stream_start()

        try:
            async for chunk in self._client.stream(
                method="POST",
                url=f"{self._base_url}{endpoint}",
                headers=headers,
                json=openai_request,
            ):
                # Parse SSE chunk
                async for event_type, data in self._parse_sse_chunk(chunk):
                    unified_event = self._transform_stream_event(event_type, data)
                    if unified_event:
                        yield unified_event

        except Exception as e:
            yield StreamEvent.error(e)

    def _get_headers(self, request: Request) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._default_headers,
        }

        # Add org/project if available
        if org_id := os.environ.get("OPENAI_ORG_ID"):
            headers["OpenAI-Organization"] = org_id
        if project_id := os.environ.get("OPENAI_PROJECT_ID"):
            headers["OpenAI-Project"] = project_id

        return headers

    def _transform_request(self, request: Request, stream: bool = False) -> dict:
        """Transform unified request to OpenAI API format."""
        if self._api_format == "chat_completions":
            return self._transform_chat_completions_request(request, stream)
        else:
            return self._transform_responses_request(request, stream)

    def _transform_chat_completions_request(self, request: Request, stream: bool = False) -> dict:
        """Transform unified request to Chat Completions API format."""
        messages = []

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                messages.append({"role": "system", "content": msg.text})
            elif msg.role == Role.DEVELOPER:
                messages.append({"role": "developer", "content": msg.text})
            elif msg.role == Role.TOOL:
                # Tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content[0].tool_result.content,
                })
            elif msg.role == Role.ASSISTANT:
                # Assistant message - handle text and tool calls
                msg_dict = {"role": "assistant", "content": ""}
                content_parts = []
                tool_calls = []

                for part in msg.content:
                    if part.kind == ContentKind.TEXT:
                        content_parts.append(part.text)
                    elif part.kind == ContentKind.TOOL_CALL:
                        tool_calls.append({
                            "id": part.tool_call.id,
                            "type": "function",
                            "function": {
                                "name": part.tool_call.name,
                                "arguments": part.tool_call.arguments,
                            },
                        })

                msg_dict["content"] = "".join(content_parts) or None
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls

                messages.append(msg_dict)
            else:  # USER
                messages.append({
                    "role": "user",
                    "content": self._transform_user_content_chat(msg.content),
                })

        openai_request = {
            "model": request.model,
            "messages": messages,
        }

        # Add optional parameters
        if request.temperature is not None:
            openai_request["temperature"] = request.temperature
        if request.top_p is not None:
            openai_request["top_p"] = request.top_p
        if request.max_tokens is not None:
            openai_request["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            openai_request["stop"] = request.stop_sequences
        if stream:
            openai_request["stream"] = True

        return openai_request

    def _transform_user_content_chat(self, content: list[ContentPart]) -> str | list[dict]:
        """Transform user content to Chat Completions API format."""
        # If only text, return simple string
        if len(content) == 1 and content[0].kind == ContentKind.TEXT:
            return content[0].text

        # Otherwise, return content array for multimodal
        result = []
        for part in content:
            if part.kind == ContentKind.TEXT:
                result.append({"type": "text", "text": part.text})
            elif part.kind == ContentKind.IMAGE:
                if part.image.url:
                    result.append({
                        "type": "image_url",
                        "image_url": {"url": part.image.url},
                    })
                else:
                    # Base64 encoded
                    media_type = part.image.media_type or "image/png"
                    data_uri = f"data:{media_type};base64,{part.image.data}"
                    result.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    })
        return result

    def _transform_responses_request(self, request: Request, stream: bool = False) -> dict:
        """Transform unified request to Responses API format."""
        # Build input array
        input_items = []
        instructions = []

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                instructions.append(msg.text)
            elif msg.role == Role.DEVELOPER:
                instructions.append(msg.text)
            elif msg.role == Role.TOOL:
                # Tool result
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": msg.content[0].tool_result.content,
                })
            elif msg.role == Role.ASSISTANT:
                # Assistant message
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": self._transform_assistant_content(msg.content),
                })
            else:  # USER
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": self._transform_user_content(msg.content),
                })

        openai_request = {
            "model": request.model,
            "input": input_items,
        }

        if instructions:
            openai_request["instructions"] = "\n\n".join(instructions)

        # Add optional parameters
        if request.temperature is not None:
            openai_request["temperature"] = request.temperature
        if request.top_p is not None:
            openai_request["top_p"] = request.top_p
        if request.max_tokens is not None:
            openai_request["max_output_tokens"] = request.max_tokens
        if request.stop_sequences:
            openai_request["stop"] = request.stop_sequences

        # Add reasoning effort for o-series models
        if request.reasoning_effort:
            openai_request["reasoning"] = {"effort": request.reasoning_effort}

        return openai_request

    def _transform_user_content(self, content: list[ContentPart]) -> list[dict]:
        """Transform user content to OpenAI format."""
        result = []
        for part in content:
            if part.kind == ContentKind.TEXT:
                result.append({"type": "input_text", "text": part.text})
            elif part.kind == ContentKind.IMAGE:
                if part.image.url:
                    result.append({
                        "type": "input_image",
                        "image_url": part.image.url,
                    })
                else:
                    # Base64 encoded
                    media_type = part.image.media_type or "image/png"
                    data_uri = f"data:{media_type};base64,{part.image.data}"
                    result.append({
                        "type": "input_image",
                        "image_url": data_uri,
                    })
        return result

    def _transform_assistant_content(self, content: list[ContentPart]) -> list[dict]:
        """Transform assistant content to OpenAI format."""
        result = []
        for part in content:
            if part.kind == ContentKind.TEXT:
                result.append({"type": "output_text", "text": part.text})
            elif part.kind == ContentKind.TOOL_CALL:
                result.append({
                    "type": "function_call",
                    "id": part.tool_call.id,
                    "name": part.tool_call.name,
                    "arguments": part.tool_call.arguments,
                })
        return result

    def _transform_response(self, http_response, request: Request) -> Response:
        """Transform OpenAI response to unified format."""
        if self._api_format == "chat_completions":
            return self._transform_chat_completions_response(http_response, request)
        else:
            return self._transform_responses_response(http_response, request)

    def _transform_chat_completions_response(self, http_response, request: Request) -> Response:
        """Transform Chat Completions API response to unified format."""
        data = http_response.json()

        # Extract content from choice
        choice = data.get("choices", [{}])[0]
        message_data = choice.get("message", {})

        content_parts = []

        # Add text content
        if content := message_data.get("content"):
            content_parts.append(ContentPart.text(content))

        # Add tool calls if present
        for tool_call in message_data.get("tool_calls", []):
            function = tool_call.get("function", {})
            content_parts.append(
                ContentPart.tool_call(
                    id=tool_call.get("id", ""),
                    name=function.get("name", ""),
                    arguments=function.get("arguments", "{}"),
                )
            )

        # Create message
        message = Message(
            role=Role.ASSISTANT,
            content=content_parts,
        )

        # Extract usage
        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        # Map finish reason
        finish_reason = choice.get("finish_reason", "stop")

        return Response(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            provider=self.name,
            message=message,
            finish_reason=FinishReason(reason=finish_reason, raw=finish_reason),
            usage=usage,
            raw=data,
        )

    def _transform_responses_response(self, http_response, request: Request) -> Response:
        """Transform Responses API response to unified format."""
        data = http_response.json()

        # Extract content from output
        content_parts = []
        output = data.get("output", [])

        for item in output:
            if item.get("type") == "message":
                for content_item in item.get("content", []):
                    if content_item.get("type") == "output_text":
                        content_parts.append(ContentPart.text(content_item.get("text", "")))
                    elif content_item.get("type") == "function_call":
                        content_parts.append(
                            ContentPart.tool_call(
                                id=content_item.get("call_id", ""),
                                name=content_item.get("name", ""),
                                arguments=content_item.get("arguments", {}),
                            )
                        )

        # Create message
        message = Message(
            role=Role.ASSISTANT,
            content=content_parts,
        )

        # Extract usage
        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)

        # Extract reasoning tokens if available
        reasoning_tokens = None
        if "completion_tokens_details" in usage_data:
            reasoning_tokens = usage_data["completion_tokens_details"].get("reasoning_tokens")

        # Extract cache tokens
        cache_read_tokens = None
        if "prompt_tokens_details" in usage_data:
            cache_read_tokens = usage_data["prompt_tokens_details"].get("cached_tokens")

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
        )

        # Map finish reason
        status = data.get("status", "completed")
        finish_reason = self._map_finish_reason(status)

        return Response(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            provider=self.name,
            message=message,
            finish_reason=FinishReason(reason=finish_reason, raw=status),
            usage=usage,
            raw=data,
        )

    def _map_finish_reason(self, openai_reason: str) -> str:
        """Map OpenAI finish reason to unified reason."""
        mapping = {
            "completed": "stop",
            "incomplete": "length",
            "failed": "error",
        }
        return mapping.get(openai_reason, "other")

    def _check_http_error(self, status_code: int, response):
        """Check HTTP status and raise appropriate error."""
        if status_code == 401:
            raise AuthenticationError(
                "Invalid API key",
                provider=self.name,
                status_code=status_code,
            )
        elif status_code == 403:
            raise AccessDeniedError(
                "Access denied",
                provider=self.name,
                status_code=status_code,
            )
        elif status_code == 404:
            raise NotFoundError(
                "Model or endpoint not found",
                provider=self.name,
                status_code=status_code,
            )
        elif status_code in (400, 422):
            raise InvalidRequestError(
                "Invalid request",
                provider=self.name,
                status_code=status_code,
            )
        elif status_code == 413:
            raise ContextLengthError(
                "Context length exceeded",
                provider=self.name,
                status_code=status_code,
            )
        elif status_code == 429:
            import re
            retry_after = None
            if retry_after_match := re.search(r"retry after (\d+)", response.text().lower()):
                retry_after = float(retry_after_match.group(1))
            raise RateLimitError(
                "Rate limit exceeded",
                provider=self.name,
                status_code=status_code,
                retry_after=retry_after,
            )
        elif status_code >= 500:
            raise ServerError(
                "Server error",
                provider=self.name,
                status_code=status_code,
            )

    async def _parse_sse_chunk(self, chunk: bytes) -> AsyncIterator[tuple[str, str]]:
        """Parse SSE chunk from OpenAI."""
        from unified_llm.utils._sse import parse_sse
        async for event_type, data in parse_sse(chunk):
            yield event_type, data

    def _transform_stream_event(self, event_type: str, data: str) -> StreamEvent | None:
        """Transform OpenAI stream event to unified format."""
        if self._api_format == "chat_completions":
            return self._transform_chat_completions_stream_event(event_type, data)
        else:
            return self._transform_responses_stream_event(event_type, data)

    def _transform_chat_completions_stream_event(self, event_type: str, data: str) -> StreamEvent | None:
        """Transform Chat Completions API stream event to unified format."""
        import json

        try:
            data_dict = json.loads(data)
        except json.JSONDecodeError:
            return None

        # Chat Completions API sends data with "data:" prefix (SSE format)
        # The event_type might be "message" or similar, and data contains the actual content
        if delta := data_dict.get("choices", [{}])[0].get("delta", {}):
            if content := delta.get("content"):
                return StreamEvent.text_delta(content)

        # Check for finish reason
        if choice := data_dict.get("choices", [{}])[0]:
            if finish_reason := choice.get("finish_reason"):
                if finish_reason != "null":
                    usage_data = data_dict.get("usage", {})
                    usage = Usage(
                        input_tokens=usage_data.get("prompt_tokens", 0),
                        output_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=0,
                    )
                    return StreamEvent.finish(
                        finish_reason=FinishReason(reason=finish_reason),
                        usage=usage,
                    )

        return None

    def _transform_responses_stream_event(self, event_type: str, data: str) -> StreamEvent | None:
        """Transform Responses API stream event to unified format."""
        if event_type == "response.created":
            return None  # Ignore
        elif event_type == "response.output_text.delta":
            import json
            delta = json.loads(data)
            return StreamEvent.text_delta(delta.get("text", ""))
        elif event_type == "response.function_call_arguments.delta":
            # Function call arguments delta
            import json
            delta = json.loads(data)
            # Would need to accumulate these in a real implementation
            return None
        elif event_type == "response.completed":
            import json
            data_dict = json.loads(data)
            usage_data = data_dict.get("usage", {})
            usage = Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=0,
            )
            return StreamEvent.finish(
                finish_reason=FinishReason(reason="stop"),
                usage=usage,
            )
        return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()

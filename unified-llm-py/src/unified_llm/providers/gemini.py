"""Gemini provider adapter using the Gemini API."""

import os
import uuid
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
from unified_llm.providers._base import ProviderAdapter
from unified_llm.utils._http import HttpClient


class GeminiAdapter(ProviderAdapter):
    """Adapter for Google Gemini using the Gemini API.

    Uses Google's native Gemini API for full feature support:
    - Grounding with Google Search
    - Code execution
    - System instructions
    - Cached content
    - Vision/image support
    """

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = 120.0,
    ):
        """Initialize Gemini adapter.

        Args:
            api_key: Gemini API key.
            base_url: Optional custom base URL.
            default_headers: Optional default HTTP headers.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._base_url = (base_url or os.environ.get("GEMINI_BASE_URL") or
                         "https://generativelanguage.googleapis.com/v1beta")
        self._default_headers = default_headers or {}
        self._timeout = timeout
        self._client = HttpClient(timeout=timeout)
        # Track tool call IDs since Gemini doesn't provide them
        self._tool_call_ids: dict[str, str] = {}

    async def complete(self, request: Request) -> Response:
        """Send request, block until complete, return full response."""
        gemini_request = self._transform_request(request)
        url = self._build_url(request.model, request)
        headers = self._get_headers(request)

        try:
            http_response = await self._client.request(
                method="POST",
                url=url,
                headers=headers,
                json=gemini_request,
            )

            # Check HTTP status
            self._check_http_error(http_response.status_code, http_response)

            # Transform Gemini response to unified format
            return self._transform_response(http_response, request)

        except Exception as e:
            if isinstance(e, (AuthenticationError, RateLimitError, ServerError)):
                raise
            raise ServerError(
                f"Gemini API error: {e}",
                provider=self.name,
                status_code=500,
            ) from e

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send request, return async iterator of stream events."""
        gemini_request = self._transform_request(request)
        url = self._build_url(request.model, request, stream=True)
        headers = self._get_headers(request)

        yield StreamEvent.stream_start()

        try:
            async for chunk in self._client.stream(
                method="POST",
                url=url,
                headers=headers,
                json=gemini_request,
            ):
                # Parse SSE chunk
                async for event_type, data in self._parse_sse_chunk(chunk):
                    unified_event = self._transform_stream_event(event_type, data)
                    if unified_event:
                        yield unified_event

        except Exception as e:
            yield StreamEvent.error(e)

    def _build_url(self, model: str, request: Request, stream: bool = False) -> str:
        """Build the request URL."""
        endpoint = "streamGenerateContent" if stream else "generateContent"
        return f"{self._base_url}/models/{model}:{endpoint}?key={self._api_key}"

    def _get_headers(self, request: Request) -> dict[str, str]:
        """Build request headers."""
        return {
            "Content-Type": "application/json",
            **self._default_headers,
        }

    def _transform_request(self, request: Request) -> dict:
        """Transform unified request to Gemini format."""
        # Build contents array
        contents = []
        system_instruction = None

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                # System instruction
                system_instruction = {"parts": [{"text": msg.text}]}
            elif msg.role == Role.DEVELOPER:
                # Merge with system instruction
                if system_instruction is None:
                    system_instruction = {"parts": []}
                system_instruction["parts"].append({"text": msg.text})
            elif msg.role == Role.TOOL:
                # Tool result
                function_response = {
                    "functionResponse": {
                        "name": msg.content[0].tool_result.tool_call_id,  # Gemini uses name not ID
                        "response": {"result": str(msg.content[0].tool_result.content)},
                    }
                }
                contents.append({"role": "user", "parts": [function_response]})
            elif msg.role == Role.ASSISTANT:
                # Assistant message
                contents.append({
                    "role": "model",
                    "parts": self._transform_assistant_content(msg.content),
                })
            else:  # USER
                contents.append({
                    "role": "user",
                    "parts": self._transform_user_content(msg.content),
                })

        gemini_request = {"contents": contents}

        if system_instruction:
            gemini_request["systemInstruction"] = system_instruction

        # Add generation config
        config = {}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["topP"] = request.top_p
        if request.max_tokens is not None:
            config["maxOutputTokens"] = request.max_tokens
        if request.stop_sequences:
            config["stopSequences"] = request.stop_sequences

        if config:
            gemini_request["generationConfig"] = config

        return gemini_request

    def _transform_user_content(self, content: list[ContentPart]) -> list[dict]:
        """Transform user content to Gemini format."""
        result = []
        for part in content:
            if part.kind == ContentKind.TEXT:
                result.append({"text": part.text})
            elif part.kind == ContentKind.IMAGE:
                if part.image.url:
                    result.append({
                        "fileData": {
                            "mimeType": part.image.media_type or "image/png",
                            "fileUri": part.image.url,
                        },
                    })
                else:
                    result.append({
                        "inlineData": {
                            "mimeType": part.image.media_type or "image/png",
                            "data": part.image.data,
                        },
                    })
        return result

    def _transform_assistant_content(self, content: list[ContentPart]) -> list[dict]:
        """Transform assistant content to Gemini format."""
        result = []
        for part in content:
            if part.kind == ContentKind.TEXT:
                result.append({"text": part.text})
            elif part.kind == ContentKind.TOOL_CALL:
                # Generate synthetic ID for tracking
                synthetic_id = f"call_{uuid.uuid4().hex}"
                self._tool_call_ids[synthetic_id] = part.tool_call.name
                result.append({
                    "functionCall": {
                        "name": part.tool_call.name,
                        "args": part.tool_call.arguments,
                    }
                })
        return result

    def _transform_response(self, http_response, request: Request) -> Response:
        """Transform Gemini response to unified format."""
        data = http_response.json()

        # Extract content from candidates
        content_parts = []
        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    content_parts.append(ContentPart.text(part["text"]))
                elif "functionCall" in part:
                    # Look up or generate tool call ID
                    func_call = part["functionCall"]
                    func_name = func_call["name"]
                    # Try to find existing ID or generate new one
                    tool_call_id = None
                    for tid, name in self._tool_call_ids.items():
                        if name == func_name:
                            tool_call_id = tid
                            break
                    if tool_call_id is None:
                        tool_call_id = f"call_{uuid.uuid4().hex}"

                    content_parts.append(
                        ContentPart.tool_call(
                            id=tool_call_id,
                            name=func_name,
                            arguments=func_call.get("args", {}),
                        )
                    )

        # Create message
        message = Message(
            role=Role.ASSISTANT,
            content=content_parts,
        )

        # Extract usage
        usage_metadata = data.get("usageMetadata", {})
        usage = Usage(
            input_tokens=usage_metadata.get("promptTokenCount", 0),
            output_tokens=usage_metadata.get("candidatesTokenCount", 0),
            total_tokens=usage_metadata.get("totalTokenCount", 0),
            reasoning_tokens=usage_metadata.get("thoughtsTokenCount"),
            cache_read_tokens=usage_metadata.get("cachedContentTokenCount"),
        )

        # Map finish reason
        finish_reason = "stop"
        if candidates:
            finish_status = candidates[0].get("finishReason", "STOP")
            finish_reason = self._map_finish_reason(finish_status)

        return Response(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            provider=self.name,
            message=message,
            finish_reason=FinishReason(reason=finish_reason, raw=finish_status),
            usage=usage,
            raw=data,
        )

    def _map_finish_reason(self, gemini_reason: str) -> str:
        """Map Gemini finish reason to unified reason."""
        mapping = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "OTHER": "other",
        }
        return mapping.get(gemini_reason, "other")

    def _check_http_error(self, status_code: int, response):
        """Check HTTP status and raise appropriate error."""
        # Gemini may use gRPC-style status codes
        error_data = response.json() if response.content else {}
        error_status = error_data.get("error", {}).get("status", "")

        if status_code == 401 or error_status == "UNAUTHENTICATED":
            raise AuthenticationError(
                "Invalid API key",
                provider=self.name,
                status_code=status_code,
            )
        elif status_code == 403 or error_status == "PERMISSION_DENIED":
            raise AccessDeniedError(
                "Access denied",
                provider=self.name,
                status_code=status_code,
            )
        elif status_code == 404 or error_status == "NOT_FOUND":
            raise NotFoundError(
                "Model not found",
                provider=self.name,
                status_code=status_code,
            )
        elif status_code in (400, 422) or error_status == "INVALID_ARGUMENT":
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
        elif status_code == 429 or error_status == "RESOURCE_EXHAUSTED":
            raise RateLimitError(
                "Rate limit exceeded",
                provider=self.name,
                status_code=status_code,
            )
        elif status_code >= 500 or error_status in ("INTERNAL", "UNAVAILABLE"):
            raise ServerError(
                "Server error",
                provider=self.name,
                status_code=status_code,
            )

    async def _parse_sse_chunk(self, chunk: bytes) -> AsyncIterator[tuple[str, str]]:
        """Parse SSE chunk from Gemini."""
        from unified_llm.utils._sse import parse_sse
        async for event_type, data in parse_sse(chunk):
            yield event_type, data

    def _transform_stream_event(self, event_type: str, data: str) -> StreamEvent | None:
        """Transform Gemini stream event to unified format."""
        if not data or data == "[DONE]":
            return None

        import json
        try:
            data_dict = json.loads(data)
        except json.JSONDecodeError:
            return None

        candidates = data_dict.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    return StreamEvent.text_delta(part["text"])

        # Check if this is the final chunk
        usage_metadata = data_dict.get("usageMetadata")
        if usage_metadata:
            usage = Usage(
                input_tokens=usage_metadata.get("promptTokenCount", 0),
                output_tokens=usage_metadata.get("candidatesTokenCount", 0),
                total_tokens=0,
            )
            finish_reason = "stop"
            if candidates and "finishReason" in candidates[0]:
                finish_reason = self._map_finish_reason(candidates[0]["finishReason"])
            return StreamEvent.finish(
                finish_reason=FinishReason(reason=finish_reason),
                usage=usage,
            )

        return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()

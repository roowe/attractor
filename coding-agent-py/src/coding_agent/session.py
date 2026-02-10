# coding-agent-py/src/coding_agent/session.py
import asyncio
import uuid
from typing import Optional, List, Deque, AsyncIterator, Any
from collections import deque

from coding_agent.models.config import SessionConfig
from coding_agent.models.turn import (
    SessionState,
    Turn,
    UserTurn,
    AssistantTurn,
    ToolResultsTurn,
    SteeringTurn,
)
from coding_agent.models.event import SessionEvent, EventKind
from coding_agent.models.tool import ToolCall, ToolResult
from coding_agent.models.usage import Usage
from coding_agent.providers.profile import ProviderProfile
from coding_agent.exec.environment import ExecutionEnvironment
from coding_agent.utils.truncation import truncate_tool_output
from coding_agent.utils.loop_detection import detect_loop


# Import unified LLM client types (from unified-llm package)
# These will need to be adapted based on actual location
try:
    from unified_llm import Client
except ImportError:
    # For development without unified-llm installed
    Client = None


class Session:
    """A coding agent session that orchestrates the agent loop."""

    def __init__(
        self,
        provider_profile: ProviderProfile,
        execution_env: ExecutionEnvironment,
        config: Optional[SessionConfig] = None,
        llm_client: Optional[Client] = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.provider_profile = provider_profile
        self.execution_env = execution_env
        self.config = config or SessionConfig()
        self._llm_client = llm_client

        self.history: List[Turn] = []
        self.state: SessionState = SessionState.IDLE
        self.steering_queue: Deque[str] = deque()
        self.followup_queue: Deque[str] = deque()
        self.abort_signaled = False
        self.subagents: dict = {}

        # Event emitter
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue()

    def steer(self, message: str) -> None:
        """Queue a steering message to inject after the current tool round."""
        self.steering_queue.append(message)

    def follow_up(self, message: str) -> None:
        """Queue a follow-up message to process after the current input completes."""
        self.followup_queue.append(message)

    async def close(self) -> None:
        """Close the session and clean up resources."""
        self.state = SessionState.CLOSED
        await self.execution_env.cleanup()
        self._emit(EventKind.SESSION_END, state="closed")

    async def events(self) -> AsyncIterator[SessionEvent]:
        """Get an async iterator for session events."""
        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                if self.state == SessionState.CLOSED:
                    break
                continue

    def _emit(self, kind: EventKind, **data) -> None:
        """Emit an event."""
        event = SessionEvent(
            kind=kind,
            session_id=self.id,
            data=data,
        )
        self._event_queue.put_nowait(event)

    async def submit(self, user_input: str) -> AsyncIterator[SessionEvent]:
        """Submit user input and yield events as processing occurs."""
        self.state = SessionState.PROCESSING

        # Add user turn to history
        user_turn = UserTurn(content=user_input)
        self.history.append(user_turn)
        self._emit(EventKind.USER_INPUT, content=user_input)
        yield self._event_queue.get_nowait()

        # Drain steering queue before first LLM call
        await self._drain_steering()

        round_count = 0

        while True:
            # Check limits
            if round_count >= self.config.max_tool_rounds_per_input:
                self._emit(EventKind.TURN_LIMIT, round=round_count)
                yield self._event_queue.get_nowait()
                break

            if self.config.max_turns > 0 and self._count_turns() >= self.config.max_turns:
                self._emit(EventKind.TURN_LIMIT, total_turns=self._count_turns())
                yield self._event_queue.get_nowait()
                break

            if self.abort_signaled:
                break

            # Build LLM request
            system_prompt = self.provider_profile.build_system_prompt(
                self.execution_env,
                {}  # TODO: Implement project docs discovery
            )

            messages = self._convert_history_to_messages()
            tool_defs = self.provider_profile.tools()

            # Create request (using unified-llm types)
            try:
                response = await self._call_llm(system_prompt, messages, tool_defs)
            except Exception as e:
                self._emit(EventKind.ERROR, message=str(e), error_type=type(e).__name__)
                yield self._event_queue.get_nowait()
                self.state = SessionState.CLOSED
                await self.close()
                return

            # Record assistant turn
            assistant_turn = AssistantTurn(
                content=response.get("text", ""),
                tool_calls=response.get("tool_calls", []),
                reasoning=response.get("reasoning"),
                usage=response.get("usage", Usage()),
                response_id=response.get("id"),
            )
            self.history.append(assistant_turn)
            self._emit(EventKind.ASSISTANT_TEXT_END, text=assistant_turn.content)
            yield self._event_queue.get_nowait()

            # If no tool calls, natural completion
            if not assistant_turn.tool_calls:
                break

            # Execute tool calls
            round_count += 1
            results = await self._execute_tool_calls(assistant_turn.tool_calls)
            self.history.append(ToolResultsTurn(results=results))

            # Drain steering queue
            await self._drain_steering()

            # Loop detection
            if self.config.enable_loop_detection:
                if detect_loop(self.history, self.config.loop_detection_window):
                    warning = (
                        f"Loop detected: The last {self.config.loop_detection_window} "
                        "tool calls follow a repeating pattern. Please try a different approach."
                    )
                    self.history.append(SteeringTurn(content=warning))
                    self._emit(EventKind.LOOP_DETECTION, message=warning)
                    yield self._event_queue.get_nowait()

        # Process follow-up queue
        if self.followup_queue:
            next_input = self.followup_queue.popleft()
            self.state = SessionState.IDLE
            async for event in self.submit(next_input):
                yield event
            return

        self.state = SessionState.IDLE
        self._emit(EventKind.SESSION_END, state="idle")
        yield self._event_queue.get_nowait()

        # Yield any remaining queued events
        while not self._event_queue.empty():
            yield self._event_queue.get_nowait()

    async def _drain_steering(self) -> None:
        """Drain the steering queue and add messages to history."""
        while self.steering_queue:
            msg = self.steering_queue.popleft()
            self.history.append(SteeringTurn(content=msg))
            self._emit(EventKind.STEERING_INJECTED, content=msg)

    async def _call_llm(self, system_prompt: str, messages: list, tools: list) -> dict:
        """Call the LLM using the unified client."""
        # This is a mock implementation for testing
        # Real implementation will use unified-llm Client

        # For now, return a simple completion
        # In production, this would be:
        # request = Request(...)
        # response = await self._llm_client.complete(request)
        # return self._convert_response_to_dict(response)

        return {
            "text": "Mock response",
            "tool_calls": [],
            "usage": Usage(),
        }

    def _convert_history_to_messages(self) -> list:
        """Convert turn history to LLM messages."""
        messages = []

        for turn in self.history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AssistantTurn):
                messages.append({"role": "assistant", "content": turn.content})
            elif isinstance(turn, ToolResultsTurn):
                # Tool results become a user message
                for result in turn.results:
                    messages.append({
                        "role": "user",
                        "content": f"Tool {result.tool_call_id} returned: {result.content}"
                    })
            elif isinstance(turn, SteeringTurn):
                messages.append({"role": "user", "content": turn.content})

        return messages

    async def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute a list of tool calls."""
        results = []

        if self.provider_profile.supports_parallel_tool_calls and len(tool_calls) > 1:
            # Parallel execution
            tasks = [self._execute_single_tool(tc) for tc in tool_calls]
            results = await asyncio.gather(*tasks)
        else:
            # Sequential execution
            for tc in tool_calls:
                result = await self._execute_single_tool(tc)
                results.append(result)

        return results

    async def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        self._emit(EventKind.TOOL_CALL_START, tool_name=tool_call.name, call_id=tool_call.id)

        # Look up tool in registry
        registered = self.provider_profile.tool_registry.get(tool_call.name)

        if registered is None:
            error_msg = f"Unknown tool: {tool_call.name}"
            self._emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg)
            return ToolResult(tool_call_id=tool_call.id, content=error_msg, is_error=True)

        # Execute tool
        try:
            raw_output = await registered.executor(tool_call.arguments, self.execution_env)

            # Truncate output before sending to LLM
            truncated_output = truncate_tool_output(raw_output, tool_call.name, self.config)

            # Emit full output
            self._emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, output=raw_output)

            return ToolResult(
                tool_call_id=tool_call.id,
                content=truncated_output,
                is_error=False,
            )

        except Exception as e:
            error_msg = f"Tool error ({tool_call.name}): {str(e)}"
            self._emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg)
            return ToolResult(tool_call_id=tool_call.id, content=error_msg, is_error=True)

    def _count_turns(self) -> int:
        """Count the total number of turns in history."""
        return len(self.history)

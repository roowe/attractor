# coding-agent-py/src/coding_agent/utils/loop_detection.py
from typing import List, Tuple

from coding_agent.models.turn import AssistantTurn, Turn


def extract_tool_call_signatures(history: List[Turn], last: int = 10) -> List[Tuple[str, int]]:
    """Extract tool call signatures from recent history.

    Returns a list of (tool_name, arguments_hash) tuples.
    """
    signatures = []

    # Get the last N turns that contain tool calls
    assistant_turns = [t for t in history if isinstance(t, AssistantTurn)]

    for turn in assistant_turns[-last:]:
        for call in turn.tool_calls:
            # Create a hash of the arguments for comparison
            args_hash = hash(frozenset(call.arguments.items()))
            signatures.append((call.name, args_hash))

    return signatures


def detect_loop(history: List[Turn], window_size: int = 10) -> bool:
    """Detect if recent tool calls follow a repeating pattern.

    Checks for patterns of length 1, 2, or 3 that repeat within the window.
    """
    recent_calls = extract_tool_call_signatures(history, last=window_size)

    if len(recent_calls) < window_size:
        return False

    # Check for repeating patterns of length 1, 2, or 3
    for pattern_len in [1, 2, 3]:
        if window_size % pattern_len != 0:
            continue

        pattern = recent_calls[:pattern_len]
        all_match = True

        for i in range(pattern_len, window_size, pattern_len):
            if recent_calls[i:i + pattern_len] != pattern:
                all_match = False
                break

        if all_match:
            return True

    return False

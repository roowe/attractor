# coding-agent-py/tests/utils/test_loop_detection.py
import pytest
from coding_agent.utils.loop_detection import detect_loop, extract_tool_call_signatures
from coding_agent.models.turn import AssistantTurn, ToolResultsTurn
from coding_agent.models.tool import ToolCall


def test_no_loop_short_history():
    turns = [
        AssistantTurn(content="Hello", tool_calls=[
            ToolCall(id="1", name="read_file", arguments={"path": "a.txt"})
        ]),
        ToolResultsTurn(results=[]),
    ]

    assert detect_loop(turns, window_size=10) is False


def test_single_call_loop():
    # Create 10 identical tool calls
    turns = []
    for i in range(10):
        turns.extend([
            AssistantTurn(
                content=f"Attempt {i}",
                tool_calls=[
                    ToolCall(id=f"{i}", name="read_file", arguments={"path": "a.txt"})
                ]
            ),
            ToolResultsTurn(results=[]),
        ])

    assert detect_loop(turns, window_size=10) is True


def test_two_call_pattern_loop():
    # Create alternating calls: read_file -> grep -> read_file -> grep ...
    turns = []
    for i in range(10):
        if i % 2 == 0:
            turns.extend([
                AssistantTurn(
                    content=f"Read {i}",
                    tool_calls=[
                        ToolCall(id=f"{i}", name="read_file", arguments={"path": "a.txt"})
                    ]
                ),
                ToolResultsTurn(results=[]),
            ])
        else:
            turns.extend([
                AssistantTurn(
                    content=f"Grep {i}",
                    tool_calls=[
                        ToolCall(id=f"{i}", name="grep", arguments={"pattern": "test"})
                    ]
                ),
                ToolResultsTurn(results=[]),
            ])

    assert detect_loop(turns, window_size=10) is True


def test_no_pattern_random_calls():
    # Create different calls each time
    turns = []
    for i in range(10):
        turns.extend([
            AssistantTurn(
                content=f"Attempt {i}",
                tool_calls=[
                    ToolCall(id=f"{i}", name="read_file", arguments={"path": f"file{i}.txt"})
                ]
            ),
            ToolResultsTurn(results=[]),
        ])

    assert detect_loop(turns, window_size=10) is False


def test_extract_tool_call_signatures():
    turns = [
        AssistantTurn(
            content="First",
            tool_calls=[
                ToolCall(id="1", name="read_file", arguments={"path": "a.txt"}),
                ToolCall(id="2", name="grep", arguments={"pattern": "test"}),
            ]
        ),
        ToolResultsTurn(results=[]),
        AssistantTurn(
            content="Second",
            tool_calls=[
                ToolCall(id="3", name="read_file", arguments={"path": "b.txt"}),
            ]
        ),
    ]

    sigs = extract_tool_call_signatures(turns, last=10)

    # Should extract all 3 calls
    assert len(sigs) == 3
    assert sigs[0][0] == "read_file"

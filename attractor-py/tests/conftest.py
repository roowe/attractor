# tests/conftest.py
import pytest


@pytest.fixture
def sample_dot():
    return """
digraph Simple {
    graph [goal="Run tests and report"]
    rankdir=LR

    start [shape=Mdiamond, label="Start"]
    exit  [shape=Msquare, label="Exit"]

    run_tests [label="Run Tests", prompt="Run the test suite"]
    report    [label="Report", prompt="Summarize results"]

    start -> run_tests -> report -> exit
}
"""

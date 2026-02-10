# tests/models/test_checkpoint.py
from datetime import datetime

from attractor.models.checkpoint import Checkpoint


def test_checkpoint_creation():
    checkpoint = Checkpoint(
        timestamp=datetime.now(),
        current_node="test_node",
        completed_nodes=["start"],
        node_retries={},
        context_values={},
        logs=[],
    )
    assert checkpoint.current_node == "test_node"
    assert "start" in checkpoint.completed_nodes


def test_checkpoint_save_load(tmp_path):
    checkpoint = Checkpoint(
        timestamp=datetime.now(),
        current_node="test_node",
        completed_nodes=["start"],
        node_retries={},
        context_values={"key": "value"},
        logs=["log entry"],
    )
    filepath = tmp_path / "checkpoint.json"
    checkpoint.save(str(filepath))

    loaded = Checkpoint.load(str(filepath))
    assert loaded.current_node == "test_node"
    assert loaded.context_values["key"] == "value"

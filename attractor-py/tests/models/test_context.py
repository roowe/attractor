# tests/models/test_context.py
from attractor.models.context import Context, Outcome, StageStatus


def test_context_set_get():
    context = Context()
    context.set("key1", "value1")
    assert context.get("key1") == "value1"


def test_context_get_default():
    context = Context()
    assert context.get("missing", "default") == "default"


def test_context_snapshot():
    context = Context()
    context.set("key1", "value1")
    snapshot = context.snapshot()
    assert snapshot["key1"] == "value1"


def test_outcome_creation():
    outcome = Outcome(status=StageStatus.SUCCESS)
    assert outcome.status == StageStatus.SUCCESS


def test_outcome_with_context_updates():
    outcome = Outcome(status=StageStatus.SUCCESS, context_updates={"key": "value"})
    assert outcome.context_updates == {"key": "value"}

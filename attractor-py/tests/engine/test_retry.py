# tests/engine/test_retry.py
from attractor.engine.retry import RetryPolicy, execute_with_retry
from attractor.models import Context, Graph, Node, Outcome, StageStatus


class MockHandler:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or [Outcome(status=StageStatus.SUCCESS)]
        self.call_count = 0

    def execute(self, node, context, graph, logs_root):
        outcome = self.outcomes[min(self.call_count, len(self.outcomes) - 1)]
        self.call_count += 1
        return outcome


def test_retry_policy_no_retry():
    policy = RetryPolicy(max_attempts=1)
    handler = MockHandler()
    node = Node(id="test", max_retries=0)
    context = Context()
    graph = Graph(id="test", nodes={}, edges=[])

    outcome = execute_with_retry(handler, node, context, graph, "/tmp", policy)

    assert outcome.status == StageStatus.SUCCESS
    assert handler.call_count == 1


def test_retry_with_success_on_retry():
    policy = RetryPolicy(max_attempts=3)
    handler = MockHandler(
        outcomes=[Outcome(status=StageStatus.RETRY), Outcome(status=StageStatus.SUCCESS)]
    )
    node = Node(id="test", max_retries=2)
    context = Context()
    graph = Graph(id="test", nodes={}, edges=[])

    outcome = execute_with_retry(handler, node, context, graph, "/tmp", policy)

    assert outcome.status == StageStatus.SUCCESS
    assert handler.call_count == 2


def test_retry_exhausted():
    policy = RetryPolicy(max_attempts=3)
    handler = MockHandler(
        outcomes=[
            Outcome(status=StageStatus.RETRY),
            Outcome(status=StageStatus.RETRY),
            Outcome(status=StageStatus.RETRY),
        ]
    )
    node = Node(id="test", max_retries=2, allow_partial=False)
    context = Context()
    graph = Graph(id="test", nodes={}, edges=[])

    outcome = execute_with_retry(handler, node, context, graph, "/tmp", policy)

    assert outcome.status == StageStatus.FAIL
    assert handler.call_count == 3

# tests/handlers/test_human.py
from attractor.handlers.human import Answer, Interviewer, Question, WaitForHumanHandler
from attractor.models import Context, Edge, Graph, Node, StageStatus


class MockInterviewer(Interviewer):
    def __init__(self, answer: str):
        self.answer = answer

    def ask(self, question: Question) -> Answer:
        return Answer(value=self.answer)


def test_wait_human_handler():
    interviewer = MockInterviewer("Y")
    handler = WaitForHumanHandler(interviewer=interviewer)

    node = Node(id="gate", shape="hexagon", label="Continue?")
    context = Context()
    graph = Graph(
        id="test",
        nodes={"gate": node, "yes": Node(id="yes", label="Yes"), "no": Node(id="no", label="No")},
        edges=[
            Edge(from_node="gate", to_node="yes", label="[Y] Yes"),
            Edge(from_node="gate", to_node="no", label="[N] No"),
        ],
    )

    outcome = handler.execute(node, context, graph, "/tmp/logs")

    assert outcome.status == StageStatus.SUCCESS
    assert "yes" in outcome.suggested_next_ids

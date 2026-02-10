# tests/integration/test_human_interaction.py
from attractor import ExecutorConfig, PipelineExecutor, StageStatus, parse_dot
from attractor.handlers.human import Answer, Question


class MockInterviewer:
    def __init__(self):
        self.responses = ["A"]  # 自动选择 A

    def ask(self, question: Question) -> Answer:
        from attractor.handlers.human import Answer

        return Answer(value=self.responses[0])


def test_human_gate_integration(tmp_path):
    """测试人机协作门控集成"""
    dot_source = """
    digraph review_pipeline {
        rankdir=LR

        start [shape=Mdiamond]
        exit [shape=Msquare]

        review_gate [shape=hexagon]

        ship_it
        fixes

        start -> review_gate
        review_gate -> ship_it
        review_gate -> fixes
        ship_it -> exit
        fixes -> review_gate
    }
    """

    graph = parse_dot(dot_source)

    config = ExecutorConfig(logs_root=str(tmp_path), interviewer=MockInterviewer())
    executor = PipelineExecutor(config)
    result = executor.run(graph)

    assert result.status == StageStatus.SUCCESS
    assert "review_gate" in result.completed_nodes
    assert "ship_it" in result.completed_nodes

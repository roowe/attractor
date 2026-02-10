# examples/simple_pipeline.py
"""简单流水线示例"""
from attractor import parse_dot, PipelineExecutor, ExecutorConfig


class MockLLM:
    def run(self, node, prompt, context):
        return f"Mock LLM response for: {prompt}"


def main():
    dot_source = '''
    digraph Hello {
        graph [goal="Say hello to the world"]

        start [shape=Mdiamond]
        hello [shape=box, prompt="Generate a hello world message"]
        exit [shape=Msquare]

        start -> hello -> exit
    }
    '''

    graph = parse_dot(dot_source)
    config = ExecutorConfig(
        logs_root="./logs",
        llm_backend=MockLLM()
    )

    executor = PipelineExecutor(config)
    result = executor.run(graph)

    print(f"Pipeline status: {result.status}")
    print(f"Completed nodes: {result.completed_nodes}")


if __name__ == "__main__":
    main()

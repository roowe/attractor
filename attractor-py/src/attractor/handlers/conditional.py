# src/attractor/handlers/conditional.py
from ..models import Context, Graph, Node, Outcome, StageStatus
from .interface import Handler


class ConditionalHandler(Handler):
    """条件路由点处理器 - 无操作，路由由引擎处理"""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS, notes=f"Conditional node evaluated: {node.id}")

# src/attractor/handlers/basic.py
from ..models import Context, Graph, Node, Outcome, StageStatus
from .interface import Handler


class StartHandler(Handler):
    """起始节点处理器 - 无操作"""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)


class ExitHandler(Handler):
    """退出节点处理器 - 无操作（目标门检查由引擎处理）"""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)

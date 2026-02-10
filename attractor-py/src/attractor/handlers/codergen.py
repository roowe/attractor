# src/attractor/handlers/codergen.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from ..models import Context, Graph, Node, Outcome, StageStatus
from .interface import Handler


class CodergenBackend(ABC):
    """Codergen 处理器后端接口"""

    @abstractmethod
    def run(self, node: Node, prompt: str, context: Context) -> Union[str, Outcome]:
        """运行 LLM 调用"""
        pass


class CodergenHandler(Handler):
    """LLM 任务处理器"""

    def __init__(self, backend: CodergenBackend | None = None) -> None:
        self._backend = backend

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        # 1. 构建提示
        prompt = node.prompt or node.label
        prompt = self._expand_variables(prompt, graph, context)

        # 2. 创建阶段目录
        stage_dir = Path(logs_root) / node.id
        stage_dir.mkdir(parents=True, exist_ok=True)

        # 3. 写入提示
        (stage_dir / "prompt.md").write_text(prompt)

        # 4. 调用后端
        if self._backend:
            result = self._backend.run(node, prompt, context)
            if isinstance(result, Outcome):
                self._write_status(stage_dir, result)
                return result
            response_text = str(result)
        else:
            response_text = f"[Simulated] Response for stage: {node.id}"

        # 5. 写入响应
        (stage_dir / "response.md").write_text(response_text)

        # 6. 创建结果
        outcome = Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Stage completed: {node.id}",
            context_updates={"last_stage": node.id, "last_response": response_text[:200]},
        )
        self._write_status(stage_dir, outcome)
        return outcome

    def _expand_variables(self, text: str, graph: Graph, context: Context) -> str:
        """展开模板变量"""
        # 目前只支持 $goal
        return text.replace("$goal", graph.goal)

    def _write_status(self, stage_dir: Path, outcome: Outcome) -> None:
        """写入状态文件"""
        import json

        status_file = stage_dir / "status.json"
        status_file.write_text(
            json.dumps(
                {
                    "outcome": outcome.status.value,
                    "preferred_next_label": outcome.preferred_label,
                    "suggested_next_ids": outcome.suggested_next_ids,
                    "context_updates": outcome.context_updates,
                    "notes": outcome.notes,
                    "failure_reason": outcome.failure_reason,
                },
                indent=2,
            )
        )

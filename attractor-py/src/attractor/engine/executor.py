# src/attractor/engine/executor.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..handlers import (
    CodergenHandler,
    ConditionalHandler,
    ExitHandler,
    HandlerRegistry,
    StartHandler,
)
from ..handlers.codergen import CodergenBackend
from ..handlers.human import Interviewer
from ..models import Checkpoint, Context, Graph, Node, Outcome, StageStatus
from .edge_selection import select_edge
from .retry import RetryPolicy, execute_with_retry


@dataclass
class ExecutorConfig:
    """执行引擎配置"""

    logs_root: str
    llm_backend: CodergenBackend = None
    interviewer: Interviewer = None
    checkpoint_interval: int = 1


@dataclass
class ExecutionResult:
    """执行结果"""

    status: StageStatus
    completed_nodes: List[str]
    node_outcomes: Dict[str, Outcome]
    final_context: Dict[str, Any]


class PipelineExecutor:
    """流水线执行引擎"""

    def __init__(self, config: ExecutorConfig) -> None:
        self._config = config
        self._handler_registry = self._create_handler_registry()
        self._retry_counters: Dict[str, int] = {}

    def run(self, graph: Graph) -> ExecutionResult:
        """运行流水线"""
        # 初始化
        context = Context()
        self._mirror_graph_attributes(graph, context)

        completed_nodes: List[str] = []
        node_outcomes: Dict[str, Outcome] = {}

        # 查找起始节点
        current_node = self._find_start_node(graph)

        # 创建日志目录
        logs_root = Path(self._config.logs_root)
        logs_root.mkdir(parents=True, exist_ok=True)

        # 主执行循环
        while True:
            node = graph.nodes[current_node]

            # 检查终端节点
            if self._is_terminal(node):
                # 检查目标门
                if self._check_goal_gates(graph, node_outcomes):
                    break
                # 目标门未满足，尝试重试目标
                retry_target = self._get_retry_target(graph, node_outcomes)
                if retry_target:
                    current_node = retry_target
                    continue
                else:
                    return ExecutionResult(
                        status=StageStatus.FAIL,
                        completed_nodes=completed_nodes,
                        node_outcomes=node_outcomes,
                        final_context=context.snapshot(),
                    )

            # 执行节点
            outcome = self._execute_node(node, context, graph, str(logs_root))

            # 记录完成
            completed_nodes.append(node.id)
            node_outcomes[node.id] = outcome

            # 应用上下文更新
            context.apply_updates(outcome.context_updates)
            context.set("outcome", outcome.status.value)
            if outcome.preferred_label:
                context.set("preferred_label", outcome.preferred_label)

            # 保存检查点
            checkpoint = Checkpoint(
                timestamp=None,  # 简化
                current_node=current_node,
                completed_nodes=completed_nodes,
                node_retries=dict(self._retry_counters),
                context_values=context.snapshot(),
                logs=[],
            )
            # checkpoint.save(logs_root / "checkpoint.json")

            # 选择下一条边
            next_edge = select_edge(node, outcome, context, graph)
            if next_edge is None:
                if outcome.status == StageStatus.FAIL:
                    return ExecutionResult(
                        status=StageStatus.FAIL,
                        completed_nodes=completed_nodes,
                        node_outcomes=node_outcomes,
                        final_context=context.snapshot(),
                    )
                break

            # 前进到下一个节点
            current_node = next_edge.to_node

        return ExecutionResult(
            status=StageStatus.SUCCESS,
            completed_nodes=completed_nodes,
            node_outcomes=node_outcomes,
            final_context=context.snapshot(),
        )

    def _create_handler_registry(self) -> HandlerRegistry:
        """创建处理器注册表"""
        from ..handlers.parallel import ParallelHandler, FanInHandler

        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        registry.register("exit", ExitHandler())
        registry.register("codergen", CodergenHandler(self._config.llm_backend))
        registry.register("conditional", ConditionalHandler())
        registry.register("parallel", ParallelHandler(self._config.llm_backend))
        registry.register("parallel.fan_in", FanInHandler(self._config.llm_backend))
        registry.default_handler = CodergenHandler(self._config.llm_backend)
        return registry

    def _mirror_graph_attributes(self, graph: Graph, context: Context) -> None:
        """将图属性镜像到上下文"""
        context.set("graph.goal", graph.goal)
        context.set("graph.label", graph.label)

    def _find_start_node(self, graph: Graph) -> str:
        """查找起始节点"""
        for node_id, node in graph.nodes.items():
            if node.shape == "Mdiamond" or node_id.lower() in ("start", "start"):
                return node_id
        raise ValueError("No start node found")

    def _is_terminal(self, node: Node) -> bool:
        """检查是否是终端节点"""
        return node.shape == "Msquare" or node.id.lower() in ("exit", "end")

    def _execute_node(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        """执行节点处理器"""
        handler = self._handler_registry.resolve(node)

        # 构建重试策略
        max_retries = node.max_retries if node.max_retries > 0 else graph.default_max_retry
        policy = RetryPolicy(max_attempts=max_retries + 1)

        return execute_with_retry(handler, node, context, graph, logs_root, policy)

    def _check_goal_gates(self, graph: Graph, node_outcomes: Dict[str, Outcome]) -> bool:
        """检查所有目标门是否满足"""
        for node_id, outcome in node_outcomes.items():
            node = graph.nodes[node_id]
            if node.goal_gate:
                if outcome.status not in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
                    return False
        return True

    def _get_retry_target(self, graph: Graph, node_outcomes: Dict[str, Outcome]) -> Optional[str]:
        """获取失败目标门的重试目标"""
        for node_id, outcome in node_outcomes.items():
            node = graph.nodes[node_id]
            if node.goal_gate and outcome.status not in (
                StageStatus.SUCCESS,
                StageStatus.PARTIAL_SUCCESS,
            ):
                if node.retry_target and node.retry_target in graph.nodes:
                    return node.retry_target
                if node.fallback_retry_target and node.fallback_retry_target in graph.nodes:
                    return node.fallback_retry_target
        if graph.retry_target and graph.retry_target in graph.nodes:
            return graph.retry_target
        if graph.fallback_retry_target and graph.fallback_retry_target in graph.nodes:
            return graph.fallback_retry_target
        return None

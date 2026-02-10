# src/attractor/validator/rules.py
from abc import ABC, abstractmethod
from typing import List

from ..models import Graph
from .models import Diagnostic, Severity


class LintRule(ABC):
    """验证规则接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """规则名称"""
        pass

    @abstractmethod
    def apply(self, graph: Graph) -> List[Diagnostic]:
        """应用规则到图"""
        pass


class StartNodeRule(LintRule):
    """检查图是否有且仅有一个起始节点"""

    @property
    def name(self) -> str:
        return "start_node"

    def apply(self, graph: Graph) -> List[Diagnostic]:
        start_nodes = [
            n
            for n in graph.nodes.values()
            if n.shape == "Mdiamond" or n.id.lower() in ("start", "start")
        ]

        if len(start_nodes) == 0:
            return [
                Diagnostic(
                    rule=self.name,
                    severity=Severity.ERROR,
                    message="Graph must have exactly one start node (shape=Mdiamond or id=start)",
                    fix="Add a node with shape=Mdiamond or id='start'",
                )
            ]
        elif len(start_nodes) > 1:
            return [
                Diagnostic(
                    rule=self.name,
                    severity=Severity.ERROR,
                    message=f"Graph has {len(start_nodes)} start nodes, must have exactly one",
                    node_id=", ".join(n.id for n in start_nodes),
                )
            ]
        return []


class TerminalNodeRule(LintRule):
    """检查图是否有至少一个终端节点"""

    @property
    def name(self) -> str:
        return "terminal_node"

    def apply(self, graph: Graph) -> List[Diagnostic]:
        terminal_nodes = [
            n
            for n in graph.nodes.values()
            if n.shape == "Msquare" or n.id.lower() in ("exit", "end")
        ]

        if len(terminal_nodes) == 0:
            return [
                Diagnostic(
                    rule=self.name,
                    severity=Severity.ERROR,
                    message="Graph must have at least one terminal node (shape=Msquare or id=exit)",
                    fix="Add a node with shape=Msquare or id='exit'",
                )
            ]
        return []


class EdgeTargetExistsRule(LintRule):
    """检查所有边的目标节点是否存在"""

    @property
    def name(self) -> str:
        return "edge_target_exists"

    def apply(self, graph: Graph) -> List[Diagnostic]:
        diagnostics = []
        for edge in graph.edges:
            if edge.to_node not in graph.nodes:
                diagnostics.append(
                    Diagnostic(
                        rule=self.name,
                        severity=Severity.ERROR,
                        message=f"Edge target '{edge.to_node}' does not exist",
                        edge=(edge.from_node, edge.to_node),
                        fix=f"Create node '{edge.to_node}' or fix edge target",
                    )
                )
        return diagnostics

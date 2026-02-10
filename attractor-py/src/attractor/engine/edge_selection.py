# src/attractor/engine/edge_selection.py
import re
from typing import Optional

from ..models import Context, Edge, Graph, Node, Outcome


def select_edge(node: Node, outcome: Outcome, context: Context, graph: Graph) -> Optional[Edge]:
    """选择下一条边"""
    # 获取出边
    edges = [e for e in graph.edges if e.from_node == node.id]
    if not edges:
        return None

    # 步骤 1：条件匹配
    from ..conditions import evaluate_condition

    condition_matched = []
    for edge in edges:
        if edge.condition:
            if evaluate_condition(edge.condition, outcome, context):
                condition_matched.append(edge)
    if condition_matched:
        return _best_by_weight_then_lexical(condition_matched)

    # 步骤 2：首选标签
    if outcome.preferred_label:
        normalized_label = _normalize_label(outcome.preferred_label)
        for edge in edges:
            if _normalize_label(edge.label) == normalized_label:
                return edge

    # 步骤 3：建议的下一个 ID
    if outcome.suggested_next_ids:
        for suggested_id in outcome.suggested_next_ids:
            for edge in edges:
                if edge.to_node == suggested_id:
                    return edge

    # 步骤 4 & 5：权重 + 词法决胜（仅无条件边）
    unconditional = [e for e in edges if not e.condition]
    if unconditional:
        return _best_by_weight_then_lexical(unconditional)

    # 回退：所有边
    return _best_by_weight_then_lexical(edges)


def _best_by_weight_then_lexical(edges: list[Edge]) -> Edge:
    """按权重然后词法顺序排序边"""
    return sorted(edges, key=lambda e: (-e.weight, e.to_node))[0]


def _normalize_label(label: str) -> str:
    """规范化边标签以进行匹配"""
    if not label:
        return ""

    # 小写
    label = label.lower().strip()

    # 移除加速器前缀模式
    patterns = [
        r"^\[([a-z0-9])\]\s*",  # [k] Label
        r"^([a-z0-9])\)\s*",  # k) Label
        r"^([a-z0-9])\s*-\s*",  # k - Label
    ]
    for pattern in patterns:
        label = re.sub(pattern, "", label)

    return label.strip()

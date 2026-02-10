# src/attractor/conditions.py
from typing import Optional

from .models import Context, Outcome


def evaluate_condition(
    condition: str, outcome: Optional[Outcome], context: Optional[Context]
) -> bool:
    """评估条件表达式"""
    if not condition:
        return True

    # 分割 AND 子句
    clauses = [c.strip() for c in condition.split("&&")]

    for clause in clauses:
        if not clause:
            continue
        if not evaluate_clause(clause, outcome, context):
            return False

    return True


def evaluate_clause(clause: str, outcome: Optional[Outcome], context: Optional[Context]) -> bool:
    """评估单个子句"""
    clause = clause.strip()

    # 检查 != 运算符
    if "!=" in clause:
        parts = clause.split("!=", 1)
        key = parts[0].strip()
        value = parts[1].strip()
        return _resolve_key(key, outcome, context) != value

    # 检查 = 运算符
    if "=" in clause:
        parts = clause.split("=", 1)
        key = parts[0].strip()
        value = parts[1].strip()
        return _resolve_key(key, outcome, context) == value

    # 裸键：检查是否为真
    return bool(_resolve_key(clause, outcome, context))


def _resolve_key(key: str, outcome: Optional[Outcome], context: Optional[Context]) -> str:
    """解析键为字符串值"""
    # outcome 特殊键
    if key == "outcome":
        return outcome.status.value if outcome else ""
    if key == "preferred_label":
        return outcome.preferred_label if outcome else ""

    # context.* 键
    if key.startswith("context."):
        path = key[8:]  # 移除 "context."
        if context:
            value = context.get(path)
            if value is not None:
                return str(value)
        return ""

    # 尝试直接上下文查找
    if context:
        value = context.get(key)
        if value is not None:
            return str(value)

    return ""

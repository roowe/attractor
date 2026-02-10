# src/attractor/validator/validator.py
from typing import List, Optional

from ..models import Graph
from .models import Diagnostic, Severity
from .rules import EdgeTargetExistsRule, LintRule, StartNodeRule, TerminalNodeRule


class ValidationError(Exception):
    """验证错误异常"""

    def __init__(self, diagnostics: List[Diagnostic]):
        self.diagnostics = diagnostics
        messages = [f"[{d.severity.value.upper()}] {d.message}" for d in diagnostics]
        super().__init__("\n".join(messages))


# 内置规则
BUILT_IN_RULES: List[LintRule] = [
    StartNodeRule(),
    TerminalNodeRule(),
    EdgeTargetExistsRule(),
]


def validate(graph: Graph, extra_rules: Optional[List[LintRule]] = None) -> List[Diagnostic]:
    """验证图并返回诊断列表"""
    rules = list(BUILT_IN_RULES)
    if extra_rules:
        rules.extend(extra_rules)

    diagnostics: List[Diagnostic] = []
    for rule in rules:
        diagnostics.extend(rule.apply(graph))

    return diagnostics


def validate_or_raise(
    graph: Graph, extra_rules: Optional[List[LintRule]] = None
) -> List[Diagnostic]:
    """验证图，如有错误则抛出异常"""
    diagnostics = validate(graph, extra_rules)
    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    if errors:
        raise ValidationError(errors)
    return diagnostics

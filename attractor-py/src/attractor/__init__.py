# src/attractor/__init__.py
__version__ = "0.1.0"

# 核心模型
# 条件表达式
from .conditions import evaluate_clause, evaluate_condition

# 执行引擎
from .engine import ExecutionResult, ExecutorConfig, PipelineExecutor

# 处理器
from .handlers import CodergenBackend, Handler, HandlerRegistry, Interviewer
from .models import Checkpoint, Context, Edge, Graph, Node, Outcome, StageStatus

# 解析器
from .parser import ParseError, parse_dot

# 验证器
from .validator import Diagnostic, Severity, ValidationError, validate, validate_or_raise

__all__ = [
    # 版本
    "__version__",
    # 核心模型
    "Graph",
    "Node",
    "Edge",
    "Context",
    "Outcome",
    "StageStatus",
    "Checkpoint",
    # 解析器
    "parse_dot",
    "ParseError",
    # 验证器
    "validate",
    "validate_or_raise",
    "ValidationError",
    "Diagnostic",
    "Severity",
    # 处理器
    "Handler",
    "HandlerRegistry",
    "CodergenBackend",
    "Interviewer",
    # 执行引擎
    "PipelineExecutor",
    "ExecutorConfig",
    "ExecutionResult",
    # 条件表达式
    "evaluate_condition",
    "evaluate_clause",
]

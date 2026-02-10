# src/attractor/validator/models.py
from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel


class Severity(str, Enum):
    """诊断严重性级别"""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Diagnostic(BaseModel):
    """验证诊断结果"""

    rule: str
    severity: Severity
    message: str
    node_id: Optional[str] = None
    edge: Optional[Tuple[str, str]] = None
    fix: Optional[str] = None

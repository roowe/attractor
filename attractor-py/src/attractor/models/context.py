# src/attractor/models/context.py
from enum import Enum
from threading import RLock
from typing import Any, Dict, List

from pydantic import BaseModel


class StageStatus(str, Enum):
    """阶段执行状态"""

    SUCCESS = "success"
    FAIL = "fail"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    SKIPPED = "skipped"


class Outcome(BaseModel):
    """节点处理器执行结果"""

    status: StageStatus
    preferred_label: str = ""
    suggested_next_ids: List[str] = []
    context_updates: Dict[str, Any] = {}
    notes: str = ""
    failure_reason: str = ""


class Context:
    """流水线运行的键值上下文"""

    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}
        self._lock = RLock()
        self._logs: List[str] = []

    def set(self, key: str, value: Any) -> None:
        """设置上下文值"""
        with self._lock:
            self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取上下文值"""
        with self._lock:
            return self._values.get(key, default)

    def get_string(self, key: str, default: str = "") -> str:
        """获取字符串值"""
        value = self.get(key)
        if value is None:
            return default
        return str(value)

    def append_log(self, entry: str) -> None:
        """添加日志条目"""
        with self._lock:
            self._logs.append(entry)

    def snapshot(self) -> Dict[str, Any]:
        """创建上下文快照"""
        with self._lock:
            return dict(self._values)

    def clone(self) -> "Context":
        """克隆上下文（用于并行分支）"""
        with self._lock:
            new_context = Context()
            new_context._values = dict(self._values)
            new_context._logs = list(self._logs)
            return new_context

    def apply_updates(self, updates: Dict[str, Any]) -> None:
        """应用上下文更新"""
        with self._lock:
            for key, value in updates.items():
                self._values[key] = value

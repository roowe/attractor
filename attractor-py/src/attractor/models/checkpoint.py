# src/attractor/models/checkpoint.py
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Checkpoint(BaseModel):
    """执行状态的可序列化快照"""

    timestamp: Optional[datetime] = None
    current_node: str
    completed_nodes: List[str]
    node_retries: Dict[str, int]
    context_values: Dict[str, Any]
    logs: List[str]

    def save(self, path: str) -> None:
        """序列化为 JSON 并写入文件系统"""
        data = {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "current_node": self.current_node,
            "completed_nodes": self.completed_nodes,
            "node_retries": self.node_retries,
            "context_values": self.context_values,
            "logs": self.logs,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        """从 JSON 文件反序列化"""
        data = json.loads(Path(path).read_text())
        timestamp = datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        return cls(
            timestamp=timestamp,
            current_node=data["current_node"],
            completed_nodes=data["completed_nodes"],
            node_retries=data["node_retries"],
            context_values=data["context_values"],
            logs=data["logs"],
        )

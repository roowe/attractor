# src/attractor/models/graph.py
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Node(BaseModel):
    """图中的节点"""

    id: str
    label: str = ""
    shape: str = "box"
    type: Optional[str] = None
    prompt: str = ""
    max_retries: int = 0
    goal_gate: bool = False
    retry_target: str = ""
    fallback_retry_target: str = ""
    fidelity: Optional[str] = None
    thread_id: Optional[str] = None
    class_attr: str = Field(default="", alias="class")
    timeout: Optional[str] = None
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None
    reasoning_effort: str = "high"
    auto_status: bool = False
    allow_partial: bool = False
    attrs: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class Edge(BaseModel):
    """图中的边"""

    from_node: str
    to_node: str
    label: str = ""
    condition: str = ""
    weight: int = 0
    fidelity: Optional[str] = None
    thread_id: Optional[str] = None
    loop_restart: bool = False


class Graph(BaseModel):
    """完整的流水线图"""

    id: str = ""
    goal: str = ""
    label: str = ""
    model_stylesheet: str = ""
    default_max_retry: int = 50
    retry_target: str = ""
    fallback_retry_target: str = ""
    default_fidelity: str = ""
    nodes: Dict[str, Node]
    edges: List[Edge]
    attrs: Dict[str, Any] = Field(default_factory=dict)

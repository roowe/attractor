# src/attractor/handlers/interface.py
from abc import ABC, abstractmethod
from typing import Dict, Optional

from ..models import Context, Graph, Node, Outcome

# 形状到处理器类型映射
SHAPE_TO_HANDLER_TYPE = {
    "Mdiamond": "start",
    "Msquare": "exit",
    "box": "codergen",
    "hexagon": "wait.human",
    "diamond": "conditional",
    "component": "parallel",
    "tripleoctagon": "parallel.fan_in",
    "parallelogram": "tool",
    "house": "stack.manager_loop",
}


class Handler(ABC):
    """节点处理器接口"""

    @abstractmethod
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        """执行节点处理器"""
        pass


class HandlerRegistry:
    """处理器注册表"""

    def __init__(self) -> None:
        self._handlers: Dict[str, Handler] = {}
        self._default_handler: Optional[Handler] = None

    def register(self, type_string: str, handler: Handler) -> None:
        """注册处理器"""
        self._handlers[type_string] = handler

    def resolve(self, node: Node) -> Handler:
        """解析节点的处理器"""
        # 1. 显式 type 属性
        if node.type and node.type in self._handlers:
            return self._handlers[node.type]

        # 2. 基于形状的解析
        handler_type = SHAPE_TO_HANDLER_TYPE.get(node.shape)
        if handler_type and handler_type in self._handlers:
            return self._handlers[handler_type]

        # 3. 默认
        if self._default_handler:
            return self._default_handler

        raise ValueError(
            f"No handler found for node {node.id} (type={node.type}, shape={node.shape})"
        )

    @property
    def default_handler(self) -> Optional[Handler]:
        return self._default_handler

    @default_handler.setter
    def default_handler(self, handler: Handler) -> None:
        self._default_handler = handler

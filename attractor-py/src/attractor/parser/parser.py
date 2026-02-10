# src/attractor/parser/parser.py
from typing import Any, Dict, List

from ..models import Edge, Graph, Node
from .lexer import Token, TokenType, tokenize


class ParseError(Exception):
    """DOT 解析错误"""

    pass


def parse_dot(source: str) -> Graph:
    """解析 DOT 源代码并返回图模型"""
    tokens = tokenize(source)
    parser = _Parser(tokens)
    return parser.parse()


class _Parser:
    """DOT 语法分析器"""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> Graph:
        """解析图"""
        # 期望 digraph 关键字
        self._expect(TokenType.DIGRAPH)

        # 图 ID
        graph_id = self._expect(TokenType.IDENTIFIER).value

        # 左大括号
        self._expect(TokenType.LBRACE)

        # 解析语句
        nodes: Dict[str, dict] = {}
        edges: List[dict] = []
        graph_attrs: Dict[str, Any] = {}

        while self._peek().type != TokenType.RBRACE:
            stmt = self._parse_statement()
            if stmt["type"] == "graph_attr":
                graph_attrs.update(stmt["attrs"])
            elif stmt["type"] == "node":
                nodes[stmt["id"]] = stmt["attrs"]
            elif stmt["type"] == "edge":
                edges.extend(stmt["edges"])

        # 右大括号
        self._expect(TokenType.RBRACE)

        # 构建模型
        return self._build_graph(graph_id, graph_attrs, nodes, edges)

    def _parse_statement(self) -> dict:
        """解析单个语句"""
        tok = self._peek()

        if tok.type == TokenType.GRAPH:
            self._advance()
            attrs = self._parse_attr_block()
            return {"type": "graph_attr", "attrs": attrs}

        elif tok.type == TokenType.NODE:
            self._advance()
            # 默认节点属性（暂存）
            self._parse_attr_block()
            return {"type": "node_defaults"}

        elif tok.type == TokenType.EDGE:
            self._advance()
            # 默认边属性（暂存）
            self._parse_attr_block()
            return {"type": "edge_defaults"}

        elif tok.type == TokenType.SUBGRAPH:
            return self._parse_subgraph()

        else:
            # 节点或边语句
            return self._parse_node_or_edge_stmt()

    def _parse_node_or_edge_stmt(self) -> dict:
        """解析节点或边语句"""
        start_id = self._expect(TokenType.IDENTIFIER).value

        # 检查是否是边
        if self._peek().type == TokenType.EDGE_OP:
            return self._parse_edge_stmt(start_id)
        else:
            # 节点语句
            attrs = self._parse_attr_block()
            return {"type": "node", "id": start_id, "attrs": attrs}

    def _parse_edge_stmt(self, from_id: str) -> dict:
        """解析边语句（支持链式 A -> B -> C）"""
        edges = []
        current_from = from_id

        while self._peek().type == TokenType.EDGE_OP:
            self._advance()  # ->
            to_id = self._expect(TokenType.IDENTIFIER).value
            edge_attrs = self._parse_attr_block()
            edges.append({"from_node": current_from, "to_node": to_id, "attrs": edge_attrs})
            current_from = to_id

        return {"type": "edge", "edges": edges}

    def _parse_subgraph(self) -> dict:
        """解析子图（简化版本：展平内容）"""
        self._advance()  # subgraph
        if self._peek().type == TokenType.IDENTIFIER:
            self._advance()  # 子图 ID
        self._expect(TokenType.LBRACE)

        # 展平子图内容
        while self._peek().type != TokenType.RBRACE:
            stmt = self._parse_statement()
            if stmt["type"] == "node":
                return stmt  # 返回第一个节点

        self._expect(TokenType.RBRACE)
        return {"type": "subgraph"}

    def _parse_attr_block(self) -> dict:
        """解析属性块 [key=value, key2=value2]"""
        attrs = {}
        if self._peek().type != TokenType.LBRACKET:
            return attrs

        self._advance()  # [
        while self._peek().type != TokenType.RBRACKET:
            key = self._expect(TokenType.IDENTIFIER).value
            self._expect(TokenType.EQUAL)
            value = self._parse_value()
            attrs[key] = value

            if self._peek().type == TokenType.COMMA:
                self._advance()

        self._advance()  # ]
        return attrs

    def _parse_value(self) -> Any:
        """解析属性值"""
        tok = self._advance()
        if tok.type == TokenType.STRING:
            return tok.value
        elif tok.type == TokenType.INTEGER:
            return int(tok.value)
        elif tok.type == TokenType.FLOAT:
            return float(tok.value)
        elif tok.type == TokenType.BOOLEAN:
            return tok.value == "true"
        elif tok.type == TokenType.DURATION:
            return tok.value
        else:  # IDENTIFIER
            return tok.value

    def _expect(self, token_type: TokenType) -> Token:
        """消费并返回期望类型的令牌"""
        if self._peek().type == token_type:
            return self._advance()
        raise ParseError(f"Expected {token_type}, got {self._peek().type}")

    def _peek(self) -> Token:
        """查看当前令牌"""
        if self.pos >= len(self.tokens):
            return Token(TokenType.RBRACE, "", -1)  # EOF 伪令牌
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        """消费并返回当前令牌"""
        tok = self._peek()
        self.pos += 1
        return tok

    def _build_graph(self, graph_id: str, graph_attrs: dict, nodes: dict, edges: list) -> Graph:
        """从解析数据构建图模型"""
        # 转换节点
        model_nodes = {}
        for node_id, attrs in nodes.items():
            # 从 shape 推断类型
            shape = attrs.get("shape", "box")
            node_type = attrs.get("type", None)
            model_nodes[node_id] = Node(
                id=node_id,
                label=attrs.get("label", node_id),
                shape=shape,
                type=node_type,
                prompt=attrs.get("prompt", ""),
                max_retries=attrs.get("max_retries", 0),
                goal_gate=attrs.get("goal_gate", False),
                retry_target=attrs.get("retry_target", ""),
                fallback_retry_target=attrs.get("fallback_retry_target", ""),
                fidelity=attrs.get("fidelity"),
                thread_id=attrs.get("thread_id"),
                class_attr=attrs.get("class", ""),
                timeout=attrs.get("timeout"),
                llm_model=attrs.get("llm_model"),
                llm_provider=attrs.get("llm_provider"),
                reasoning_effort=attrs.get("reasoning_effort", "high"),
                auto_status=attrs.get("auto_status", False),
                allow_partial=attrs.get("allow_partial", False),
                attrs=attrs,
            )

        # 转换边
        model_edges = []
        for edge_data in edges:
            model_edges.append(
                Edge(
                    from_node=edge_data["from_node"],
                    to_node=edge_data["to_node"],
                    label=edge_data["attrs"].get("label", ""),
                    condition=edge_data["attrs"].get("condition", ""),
                    weight=edge_data["attrs"].get("weight", 0),
                    fidelity=edge_data["attrs"].get("fidelity"),
                    thread_id=edge_data["attrs"].get("thread_id"),
                    loop_restart=edge_data["attrs"].get("loop_restart", False),
                )
            )

        return Graph(
            id=graph_id,
            goal=graph_attrs.get("goal", ""),
            label=graph_attrs.get("label", ""),
            model_stylesheet=graph_attrs.get("model_stylesheet", ""),
            default_max_retry=graph_attrs.get("default_max_retry", 50),
            retry_target=graph_attrs.get("retry_target", ""),
            fallback_retry_target=graph_attrs.get("fallback_retry_target", ""),
            default_fidelity=graph_attrs.get("default_fidelity", ""),
            nodes=model_nodes,
            edges=model_edges,
            attrs=graph_attrs,
        )

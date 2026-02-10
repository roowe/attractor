# src/attractor/parser/lexer.py
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class TokenType(Enum):
    """令牌类型"""

    DIGRAPH = auto()
    SUBGRAPH = auto()
    GRAPH = auto()
    NODE = auto()
    EDGE = auto()
    IDENTIFIER = auto()
    EDGE_OP = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    EQUAL = auto()
    COMMA = auto()
    SEMICOLON = auto()
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    DURATION = auto()


@dataclass
class Token:
    """词法令牌"""

    type: TokenType
    value: str
    line: int = 1
    column: int = 0


def tokenize(source: str) -> List[Token]:
    """将 DOT 源代码转换为令牌列表"""
    # 移除注释
    source = _strip_comments(source)

    tokens = []
    line = 1
    i = 0

    while i < len(source):
        # 跳过空白
        while i < len(source) and source[i].isspace():
            if source[i] == "\n":
                line += 1
            i += 1
        if i >= len(source):
            break

        # 识别关键字
        matched = False
        for keyword, token_type in [
            ("digraph", TokenType.DIGRAPH),
            ("subgraph", TokenType.SUBGRAPH),
            ("graph", TokenType.GRAPH),
            ("node", TokenType.NODE),
            ("edge", TokenType.EDGE),
            ("true", TokenType.BOOLEAN),
            ("false", TokenType.BOOLEAN),
        ]:
            if _match_keyword(source, i, keyword):
                tokens.append(Token(token_type, keyword, line))
                i += len(keyword)
                matched = True
                break

        if not matched:
            # 识别字符串
            if source[i] == '"':
                s, end = _read_string(source, i)
                tokens.append(Token(TokenType.STRING, s, line))
                i = end
            # 识别边操作符
            elif source[i : i + 2] == "->":
                tokens.append(Token(TokenType.EDGE_OP, "->", line))
                i += 2
            # 识别单字符令牌
            elif source[i] == "{":
                tokens.append(Token(TokenType.LBRACE, "{", line))
                i += 1
            elif source[i] == "}":
                tokens.append(Token(TokenType.RBRACE, "}", line))
                i += 1
            elif source[i] == "[":
                tokens.append(Token(TokenType.LBRACKET, "[", line))
                i += 1
            elif source[i] == "]":
                tokens.append(Token(TokenType.RBRACKET, "]", line))
                i += 1
            elif source[i] == "=":
                tokens.append(Token(TokenType.EQUAL, "=", line))
                i += 1
            elif source[i] == ",":
                tokens.append(Token(TokenType.COMMA, ",", line))
                i += 1
            elif source[i] == ";":
                tokens.append(Token(TokenType.SEMICOLON, ";", line))
                i += 1
            # 识别数字
            elif source[i].isdigit() or source[i] == "-":
                num, end = _read_number(source, i)
                if "." in num:
                    tokens.append(Token(TokenType.FLOAT, num, line))
                else:
                    tokens.append(Token(TokenType.INTEGER, num, line))
                i = end
            # 识别标识符
            else:
                ident, end = _read_identifier(source, i)
                # 检查是否是持续时间
                if _is_duration(source, end):
                    duration_end = _read_duration(source, i)
                    tokens.append(Token(TokenType.DURATION, source[i:duration_end], line))
                    i = duration_end
                else:
                    tokens.append(Token(TokenType.IDENTIFIER, ident, line))
                    i = end

    return tokens


def _strip_comments(source: str) -> str:
    """移除 // 和 /* */ 注释"""
    # 移除单行注释
    result = re.sub(r"//.*", "", source)
    # 移除多行注释
    result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)
    return result


def _match_keyword(source: str, pos: int, keyword: str) -> bool:
    """检查指定位置是否匹配关键字"""
    if not source.startswith(keyword, pos):
        return False
    # 检查后面是否是标识符结束符
    if pos + len(keyword) < len(source):
        next_char = source[pos + len(keyword)]
        if next_char.isalnum() or next_char == "_":
            return False
    return True


def _read_string(source: str, start: int) -> tuple[str, int]:
    """读取字符串字面量"""
    i = start + 1  # 跳过开始引号
    result = []
    while i < len(source):
        if source[i] == "\\" and i + 1 < len(source):
            # 处理转义序列
            esc = source[i + 1]
            if esc == "n":
                result.append("\n")
            elif esc == "t":
                result.append("\t")
            elif esc == '"' or esc == "\\":
                result.append(esc)
            i += 2
        elif source[i] == '"':
            return "".join(result), i + 1
        else:
            result.append(source[i])
            i += 1
    raise ValueError("Unterminated string")


def _read_number(source: str, start: int) -> tuple[str, int]:
    """读取数字（整数或浮点数）"""
    i = start
    if source[i] == "-":
        i += 1
    while i < len(source) and source[i].isdigit():
        i += 1
    if i < len(source) and source[i] == ".":
        i += 1
        while i < len(source) and source[i].isdigit():
            i += 1
    return source[start:i], i


def _read_identifier(source: str, start: int) -> tuple[str, int]:
    """读取标识符"""
    i = start
    while i < len(source) and (source[i].isalnum() or source[i] == "_"):
        i += 1
    return source[start:i], i


def _is_duration(source: str, pos: int) -> bool:
    """检查位置是否在持续时间单位处"""
    for unit in ["ms", "s", "m", "h", "d"]:
        if pos + len(unit) <= len(source) and source[pos : pos + len(unit)] == unit:
            return True
    return False


def _read_duration(source: str, start: int) -> int:
    """读取完整持续时间"""
    i = start
    while i < len(source) and (source[i].isalnum() or source[i] == "_"):
        i += 1
    return i

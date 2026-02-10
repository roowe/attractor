# tests/parser/test_lexer.py
from attractor.parser.lexer import TokenType, tokenize


def test_tokenize_keywords():
    tokens = tokenize("digraph Test { node [shape=box] }")
    assert tokens[0].type == TokenType.DIGRAPH
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == "Test"


def test_tokenize_attributes():
    tokens = tokenize('[label="Hello", shape=box]')
    assert any(t.type == TokenType.STRING and t.value == "Hello" for t in tokens)


def test_tokenize_edge():
    tokens = tokenize("A -> B")
    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[1].type == TokenType.EDGE_OP

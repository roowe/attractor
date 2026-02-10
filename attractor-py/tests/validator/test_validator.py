# tests/validator/test_validator.py
import pytest

from attractor.parser import parse_dot
from attractor.validator.validator import ValidationError, validate, validate_or_raise


def test_validate_valid_graph():
    source = """
    digraph Test {
        start [shape=Mdiamond]
        end [shape=Msquare]
        start -> end
    }
    """
    graph = parse_dot(source)
    diagnostics = validate(graph)
    # 应该没有错误
    errors = [d for d in diagnostics if d.severity.value == "error"]
    assert len(errors) == 0


def test_validate_invalid_graph():
    source = """
    digraph Test {
        start [shape=box]
        start -> end
    }
    """
    graph = parse_dot(source)
    diagnostics = validate(graph)
    # 应该有错误（缺少起始和终端节点）
    errors = [d for d in diagnostics if d.severity.value == "error"]
    assert len(errors) > 0


def test_validate_or_raise_passes():
    source = """
    digraph Test {
        start [shape=Mdiamond]
        end [shape=Msquare]
        start -> end
    }
    """
    graph = parse_dot(source)
    # 不应该抛出异常
    diagnostics = validate_or_raise(graph)
    assert True


def test_validate_or_raise_fails():
    source = """
    digraph Test {
        lone [shape=box]
    }
    """
    graph = parse_dot(source)
    with pytest.raises(ValidationError) as exc_info:
        validate_or_raise(graph)
    assert "start" in str(exc_info.value).lower()

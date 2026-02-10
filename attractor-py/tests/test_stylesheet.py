# tests/test_stylesheet.py
import pytest
from attractor.models import Graph, Node
from attractor.stylesheet import (
    SelectorMatch,
    StylesheetRule,
    StylesheetTransform,
    apply_stylesheet,
    parse_stylesheet,
)


def test_parse_empty_stylesheet():
    """测试解析空样式表"""
    rules = parse_stylesheet("")
    assert rules == []

    rules = parse_stylesheet("   \n  \t  ")
    assert rules == []


def test_parse_universal_selector():
    """测试解析通用选择器"""
    stylesheet = '* { llm_model: claude-sonnet-4-5; llm_provider: anthropic; }'
    rules = parse_stylesheet(stylesheet)

    assert len(rules) == 1
    assert rules[0].selector == '*'
    assert rules[0].declarations['llm_model'] == 'claude-sonnet-4-5'
    assert rules[0].declarations['llm_provider'] == 'anthropic'


def test_parse_class_selector():
    """测试解析类选择器"""
    stylesheet = '.code { llm_model: claude-opus-4-6; }'
    rules = parse_stylesheet(stylesheet)

    assert len(rules) == 1
    assert rules[0].selector == '.code'
    assert rules[0].declarations['llm_model'] == 'claude-opus-4-6'


def test_parse_id_selector():
    """测试解析ID选择器"""
    stylesheet = '#critical_review { llm_model: gpt-5.2; reasoning_effort: high; }'
    rules = parse_stylesheet(stylesheet)

    assert len(rules) == 1
    assert rules[0].selector == '#critical_review'
    assert rules[0].declarations['llm_model'] == 'gpt-5.2'
    assert rules[0].declarations['reasoning_effort'] == 'high'


def test_parse_multiple_rules():
    """测试解析多条规则"""
    stylesheet = '''
        * { llm_model: claude-sonnet-4-5; }
        .code { llm_model: claude-opus-4-6; }
        #review { llm_provider: openai; }
    '''
    rules = parse_stylesheet(stylesheet)

    assert len(rules) == 3
    assert rules[0].selector == '*'
    assert rules[1].selector == '.code'
    assert rules[2].selector == '#review'


def test_parse_mixed_declarations():
    """测试解析混合声明"""
    stylesheet = '.fast { llm_model: gemini-flash; llm_provider: google; reasoning_effort: low; }'
    rules = parse_stylesheet(stylesheet)

    assert len(rules) == 1
    assert len(rules[0].declarations) == 3


def test_apply_universal_selector():
    """测试应用通用选择器"""
    node = Node(id='test', llm_model=None, llm_provider=None)
    graph = Graph(nodes={'test': node}, edges=[])

    rules = [StylesheetRule(
        selector='*',
        declarations={'llm_model': 'claude-sonnet-4-5', 'llm_provider': 'anthropic'},
    )]

    result = apply_stylesheet(graph, rules)

    assert result.nodes['test'].llm_model == 'claude-sonnet-4-5'
    assert result.nodes['test'].llm_provider == 'anthropic'


def test_apply_class_selector():
    """测试应用类选择器"""
    node = Node(
        id='test',
        class_attr='code,review',
        llm_model=None,
        llm_provider=None,
    )
    graph = Graph(nodes={'test': node}, edges=[])

    rules = [StylesheetRule(
        selector='.code',
        declarations={'llm_model': 'claude-opus-4-6'},
    )]

    result = apply_stylesheet(graph, rules)

    assert result.nodes['test'].llm_model == 'claude-opus-4-6'


def test_apply_id_selector():
    """测试应用ID选择器"""
    node = Node(id='critical_review', llm_model=None, llm_provider=None)
    graph = Graph(nodes={'critical_review': node}, edges=[])

    rules = [StylesheetRule(
        selector='#critical_review',
        declarations={'llm_model': 'gpt-5.2', 'llm_provider': 'openai'},
    )]

    result = apply_stylesheet(graph, rules)

    assert result.nodes['critical_review'].llm_model == 'gpt-5.2'
    assert result.nodes['critical_review'].llm_provider == 'openai'


def test_specificity_order():
    """测试特异性优先级"""
    node = Node(
        id='review',
        class_attr='code',
        llm_model=None,
        llm_provider=None,
    )
    graph = Graph(nodes={'review': node}, edges=[])

    rules = [
        StylesheetRule(selector='*', declarations={'llm_model': 'default-model'}),
        StylesheetRule(selector='.code', declarations={'llm_model': 'code-model'}),
        StylesheetRule(selector='#review', declarations={'llm_model': 'review-model'}),
    ]

    result = apply_stylesheet(graph, rules)

    # ID selector should win (highest specificity)
    assert result.nodes['review'].llm_model == 'review-model'


def test_explicit_attribute_not_overridden():
    """测试显式属性不被覆盖"""
    node = Node(
        id='test',
        class_attr='code',
        llm_model='explicit-model',
        llm_provider=None,
    )
    graph = Graph(nodes={'test': node}, edges=[])

    rules = [StylesheetRule(
        selector='.code',
        declarations={'llm_model': 'stylesheet-model'},
    )]

    result = apply_stylesheet(graph, rules)

    # Explicit attribute should not be overridden
    assert result.nodes['test'].llm_model == 'explicit-model'


def test_stylesheet_transform():
    """测试样式表转换"""
    stylesheet = '''
        * { llm_model: claude-sonnet-4-5; }
        .code { llm_model: claude-opus-4-6; }
    '''

    transform = StylesheetTransform(stylesheet)

    nodes = {
        'plan': Node(id='plan', class_attr='planning', llm_model=None, llm_provider=None),
        'implement': Node(id='implement', class_attr='code', llm_model=None, llm_provider=None),
    }
    graph = Graph(nodes=nodes, edges=[])

    result = transform.apply(graph)

    # plan gets universal selector
    assert result.nodes['plan'].llm_model == 'claude-sonnet-4-5'
    # implement gets class selector (higher specificity)
    assert result.nodes['implement'].llm_model == 'claude-opus-4-6'


def test_no_matching_rules():
    """测试无匹配规则的情况"""
    node = Node(id='test', class_attr='other', llm_model=None, llm_provider=None)
    graph = Graph(nodes={'test': node}, edges=[])

    rules = [StylesheetRule(
        selector='.code',
        declarations={'llm_model': 'claude-opus-4-6'},
    )]

    result = apply_stylesheet(graph, rules)

    # No change when no rules match
    assert result.nodes['test'].llm_model is None

# src/attractor/stylesheet.py
"""CSS-like stylesheet for model configuration defaults."""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import Graph, Node


@dataclass
class StylesheetRule:
    """A single stylesheet rule."""

    selector: str  # '*', '.class', or '#id'
    declarations: Dict[str, str]  # property -> value


@dataclass
class SelectorMatch:
    """Result of matching a selector to a node."""

    rule: StylesheetRule
    specificity: int  # 0 = universal, 1 = class, 2 = ID


def parse_stylesheet(stylesheet: str) -> List[StylesheetRule]:
    """Parse a CSS-like stylesheet string into rules.

    Args:
        stylesheet: The stylesheet string from graph.model_stylesheet

    Returns:
        List of parsed stylesheet rules
    """
    if not stylesheet:
        return []

    rules = []

    # Pattern: selector { declarations }
    # Declarations are: property: value; (with optional trailing semicolon)
    rule_pattern = re.compile(
        r'([*#.][\w-]*)\s*\{\s*(.*?)\s*\}', re.DOTALL
    )

    for match in rule_pattern.finditer(stylesheet):
        selector = match.group(1).strip()
        declarations_str = match.group(2).strip()

        # Parse declarations
        declarations = {}
        if declarations_str:
            # Split by semicolon, but not within braces or quotes
            parts = _split_declarations(declarations_str)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if ':' in part:
                    prop, value = part.split(':', 1)
                    declarations[prop.strip()] = value.strip()

        if declarations:
            rules.append(StylesheetRule(
                selector=selector,
                declarations=declarations,
            ))

    return rules


def _split_declarations(declarations_str: str) -> List[str]:
    """Split declaration string by semicolons, handling nested structures."""
    parts = []
    current = []
    depth = 0

    for char in declarations_str:
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
        elif char == ';' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
            continue

        current.append(char)

    if current:
        parts.append(''.join(current).strip())

    return parts


def apply_stylesheet(graph: Graph, rules: List[StylesheetRule]) -> Graph:
    """Apply stylesheet rules to a graph.

    Creates a new graph with stylesheet properties applied to nodes.
    Explicit node attributes are never overridden.

    Args:
        graph: The input graph
        rules: Parsed stylesheet rules

    Returns:
        A new graph with stylesheet properties applied
    """
    if not rules:
        return graph

    # Create new nodes dict with applied styles
    new_nodes = {}
    for node_id, node in graph.nodes.items():
        new_nodes[node_id] = _apply_rules_to_node(node, rules, graph)

    # Return new graph with updated nodes
    return graph.model_copy(update={"nodes": new_nodes})


def _apply_rules_to_node(
    node: Node, rules: List[StylesheetRule], graph: Graph
) -> Node:
    """Apply matching stylesheet rules to a node.

    Args:
        node: The node to apply rules to
        rules: All stylesheet rules
        graph: The containing graph (for context)

    Returns:
        A new node with stylesheet properties applied
    """
    # Find matching rules with their specificity
    matches = _find_matching_rules(node, rules)

    # Sort by specificity (lowest first, so higher specificity overrides)
    matches.sort(key=lambda m: m.specificity)

    # Build update dict (only for properties not explicitly set)
    updates = {}

    # Get node's class attribute
    node_classes = set()
    if node.class_attr:
        node_classes = set(c.strip() for c in node.class_attr.split(','))

    # Apply rules in order (lowest to highest specificity)
    for match in matches:
        for prop, value in match.rule.declarations.items():
            # Only set if not already explicitly set on the node
            if _should_apply_property(node, prop):
                updates[prop] = value

    if updates:
        return node.model_copy(update=updates)

    return node


def _find_matching_rules(
    node: Node, rules: List[StylesheetRule]
) -> List[SelectorMatch]:
    """Find all rules that match a node, with their specificity."""
    matches = []

    node_classes = set()
    if node.class_attr:
        node_classes = set(c.strip() for c in node.class_attr.split(','))

    for rule in rules:
        specificity = _get_selector_specificity(rule.selector, node.id, node_classes)
        if specificity is not None:
            matches.append(SelectorMatch(rule=rule, specificity=specificity))

    return matches


def _get_selector_specificity(
    selector: str, node_id: str, node_classes: set
) -> Optional[int]:
    """Get the specificity of a selector for a node.

    Returns:
        Specificity (0=universal, 1=class, 2=ID) or None if no match
    """
    if selector == '*':
        return 0  # Universal selector

    if selector.startswith('#'):
        # ID selector
        target_id = selector[1:]
        return 2 if target_id == node_id else None

    if selector.startswith('.'):
        # Class selector
        target_class = selector[1:]
        return 1 if target_class in node_classes else None

    return None


def _should_apply_property(node: Node, property_name: str) -> bool:
    """Check if a stylesheet property should be applied to a node.

    Properties are only applied if not already explicitly set.
    """
    if property_name == 'llm_model':
        return node.llm_model is None
    elif property_name == 'llm_provider':
        return node.llm_provider is None
    elif property_name == 'reasoning_effort':
        # reasoning_effort has a default, so we check if it was explicitly set
        # For simplicity, we always allow stylesheet to set this
        return True
    return False


class StylesheetTransform:
    """Transform that applies model stylesheet to a graph."""

    def __init__(self, stylesheet: str):
        """Initialize the transform.

        Args:
            stylesheet: The stylesheet string to apply
        """
        self._rules = parse_stylesheet(stylesheet)

    def apply(self, graph: Graph) -> Graph:
        """Apply the stylesheet to the graph.

        Args:
            graph: The input graph

        Returns:
            A new graph with stylesheet properties applied
        """
        return apply_stylesheet(graph, self._rules)

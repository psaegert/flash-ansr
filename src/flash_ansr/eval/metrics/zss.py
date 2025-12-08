"""ZSS-based tree similarity metrics used during evaluation."""

from __future__ import annotations

from typing import Sequence

import zss
from zss import Node


def build_tree(prefix_expression: Sequence[str], operators: dict[str, int]) -> Node:
    """Convert a prefix expression into a ``zss.Node`` tree."""
    stack: list[Node] = []

    for token in reversed(prefix_expression):
        node = Node(token, [stack.pop() for _ in range(operators.get(token, 0))])
        stack.append(node)

    if not stack:
        raise ValueError("prefix_expression must contain at least one token")

    return stack[0]


def zss_tree_edit_distance(
    expression1: Sequence[str],
    expression2: Sequence[str],
    operators: dict[str, int],
) -> float:
    """Compute the Zhang-Shasha tree edit distance between two prefix expressions."""
    if len(expression1) == 0:
        expression1 = ["<EMPTY>"]
    if len(expression2) == 0:
        expression2 = ["<EMPTY>"]

    tree1 = build_tree(expression1, operators)
    tree2 = build_tree(expression2, operators)

    return zss.simple_distance(tree1, tree2)


__all__ = ["build_tree", "zss_tree_edit_distance"]

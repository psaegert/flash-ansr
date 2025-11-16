import zss
from zss import Node


def build_tree(prefix_expression: list[str], operators: dict[str, int]) -> Node:
    '''
    Convert prefix notation expression to a tree using zss.Node.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression.
    operators : dict[str, int]
        The number of operands for each operator.

    Returns
    -------
    zss.Node
        The root node of the tree.
    '''
    stack: list[str] = []

    for token in reversed(prefix_expression):
        node = Node(token, [stack.pop() for _ in range(operators.get(token, 0))])
        stack.append(node)

    return stack[0]


def zss_tree_edit_distance(expression1: list[str], expression2: list[str], operators: dict[str, int]) -> float:
    '''
    Compute the tree edit distance between two prefix expressions.

    Parameters
    ----------
    expression1 : list[str]
        The first prefix expression.
    expression2 : list[str]
        The second prefix expression.
    operators : dict[str, int]
        The number of operands for each operator.

    Returns
    -------
    float
        The tree edit distance.
    '''
    tree1 = build_tree(expression1, operators)
    tree2 = build_tree(expression2, operators)

    return zss.simple_distance(tree1, tree2)

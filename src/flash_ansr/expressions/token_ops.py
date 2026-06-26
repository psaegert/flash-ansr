"""Token-level helpers for manipulating prefix expressions."""
import re
from typing import Any

import numpy as np


def substitute_constants(
    prefix_expression: list[str],
    values: list | np.ndarray,
    constants: list[str] | None = None,
    inplace: bool = False,
) -> list[str]:
    """Fill ``<constant>`` placeholders (and known constant names) with ``values``."""
    modified_prefix_expression = prefix_expression if inplace else prefix_expression.copy()

    constant_index = 0
    constants = [] if constants is None else list(constants)

    for i, token in enumerate(prefix_expression):
        if token == "<constant>" or re.match(r"C_\d+", token) or token in constants:
            modified_prefix_expression[i] = str(values[constant_index])
            constant_index += 1

    return modified_prefix_expression


def apply_variable_mapping(prefix_expression: list[str], variable_mapping: dict[str, str]) -> list[str]:
    """Return a new prefix expression with variables remapped via ``variable_mapping``."""
    return [variable_mapping.get(token, token) for token in prefix_expression]


def identify_constants(
    prefix_expression: list[str],
    constants: list[str] | None = None,
    inplace: bool = False,
    convert_numbers_to_constant: bool = True,
) -> tuple[list[str], list[str]]:
    """Rename ``<constant>`` tokens (optionally numeric literals) to ``C_i`` symbols."""
    modified_prefix_expression = prefix_expression if inplace else prefix_expression.copy()

    constant_index = 0
    constants = [] if constants is None else list(constants)

    for i, token in enumerate(prefix_expression):
        matches_constant = token == "<constant>" or (
            convert_numbers_to_constant and (re.match(r"C_\d+", token) or token.isnumeric())
        )
        if matches_constant:
            if len(constants) > constant_index:
                modified_prefix_expression[i] = constants[constant_index]
            else:
                modified_prefix_expression[i] = f"C_{constant_index}"
                constants.append(f"C_{constant_index}")
            constant_index += 1

    return modified_prefix_expression, constants


def flatten_nested_list(nested_list: list[Any] | Any, reverse: bool = False) -> list[str]:
    """Flatten a nested structure of lists into a flat list of tokens."""
    flat_list: list[str] = []
    stack: list[Any] = [nested_list]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            flat_list.append(current)
    if reverse:
        flat_list.reverse()
    return flat_list

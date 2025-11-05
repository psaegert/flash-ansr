"""Token-level helpers for manipulating prefix expressions."""
import re
from copy import deepcopy
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


def numbers_to_num(prefix_expression: list[str], inplace: bool = False) -> list[str]:
    """Replace numeric literals in ``prefix_expression`` with ``'<constant>'``."""
    modified_prefix_expression = prefix_expression if inplace else prefix_expression.copy()

    for i, token in enumerate(prefix_expression):
        try:
            float(token)
        except ValueError:
            modified_prefix_expression[i] = token
        else:
            modified_prefix_expression[i] = "<constant>"

    return modified_prefix_expression


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


def remap_expression(
    source_expression: list[str],
    dummy_variables: list[str],
    variable_mapping: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Rename dummy variables consistently within ``source_expression``."""
    source_expression = deepcopy(source_expression)
    if variable_mapping is None:
        variable_mapping = {}
        for token in source_expression:
            if token in dummy_variables and token not in variable_mapping:
                variable_mapping[token] = f"_{len(variable_mapping)}"

    for i, token in enumerate(source_expression):
        if token in dummy_variables:
            source_expression[i] = variable_mapping[token]

    return source_expression, variable_mapping


def deduplicate_rules(
    rules_list: list[tuple[tuple[str, ...], tuple[str, ...]]],
    dummy_variables: list[str],
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Collapse equivalent rewrite rules, keeping the shortest replacement."""
    deduplicated_rules: dict[tuple[str, ...], tuple[str, ...]] = {}
    for rule in rules_list:
        remapped_source, variable_mapping = remap_expression(list(rule[0]), dummy_variables=dummy_variables)
        remapped_target, _ = remap_expression(list(rule[1]), dummy_variables, variable_mapping)

        remapped_source_key = tuple(remapped_source)
        remapped_target_value = tuple(remapped_target)

        existing_replacement = deduplicated_rules.get(remapped_source_key)
        if existing_replacement is None or len(remapped_target_value) < len(existing_replacement):
            deduplicated_rules[remapped_source_key] = remapped_target_value

    return list(deduplicated_rules.items())

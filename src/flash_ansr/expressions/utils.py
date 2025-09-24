import time
import re
from types import CodeType
from typing import Any, Callable
from functools import partial
import math
from copy import deepcopy

import numpy as np


def codify(code_string: str, variables: list[str] | None = None) -> CodeType:
    '''
    Compile a string into a code object.

    Parameters
    ----------
    code_string : str
        The string to compile.
    variables : list[str] | None
        The variables to use in the code.

    Returns
    -------
    CodeType
        The compiled code object.
    '''
    if variables is None:
        variables = []
    func_string = f'lambda {", ".join(variables)}: {code_string}'
    filename = f'<lambdifygenerated-{time.time_ns()}'
    return compile(func_string, filename, 'eval')


def get_used_modules(infix_expression: str) -> list[str]:
    '''
    Get the python modules used in an infix expression.

    Parameters
    ----------
    infix_expression : str
        The infix expression to parse.

    Returns
    -------
    list[str]
        The python modules used in the expression.
    '''
    # Match the expression against `module.submodule. ... .function(`
    pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\(')

    # Find all matches in the whole expression
    matches = pattern.findall(infix_expression)

    # Return the unique matches
    modules_set = set(m.split('.')[0] for m in matches)

    modules_set.update(['numpy'])

    return list(modules_set)


def substitude_constants(prefix_expression: list[str], values: list | np.ndarray, constants: list[str] | None = None, inplace: bool = False) -> list[str]:
    '''
    Substitute the numeric placeholders or constants in a prefix expression with the given values.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to substitute the values in.
    values : list | np.ndarray
        The values to substitute in the expression, in order.
    constants : list[str] | None
        The constants to substitute in the expression.
    inplace : bool
        Whether to modify the expression in place.

    Returns
    -------
    list[str]
        The prefix expression with the values substituted.
    '''
    if inplace:
        modified_prefix_expression = prefix_expression
    else:
        modified_prefix_expression = prefix_expression.copy()

    constant_index = 0
    if constants is None:
        constants = []
    else:
        constants = list(constants)

    for i, token in enumerate(prefix_expression):
        if token == "<constant>" or re.match(r"C_\d+", token) or token in constants:
            modified_prefix_expression[i] = str(values[constant_index])
            constant_index += 1

    return modified_prefix_expression


def apply_variable_mapping(prefix_expression: list[str], variable_mapping: dict[str, str]) -> list[str]:
    '''
    Apply a variable mapping to a prefix expression.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to apply the mapping to.
    variable_mapping : dict[str, str]
        The variable mapping to apply.

    Returns
    -------
    list[str]
        The prefix expression with the variable mapping applied.
    '''
    return list(map(lambda token: variable_mapping.get(token, token), prefix_expression))


def numbers_to_num(prefix_expression: list[str], inplace: bool = False) -> list[str]:
    '''
    Replace all numbers in a prefix expression with the string '<constant>'.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to replace the numbers in.
    inplace : bool
        Whether to modify the expression in place.

    Returns
    -------
    list[str]
        The prefix expression with the numbers replaced.
    '''
    if inplace:
        modified_prefix_expression = prefix_expression
    else:
        modified_prefix_expression = prefix_expression.copy()

    for i, token in enumerate(prefix_expression):
        try:
            float(token)
            modified_prefix_expression[i] = '<constant>'
        except ValueError:
            modified_prefix_expression[i] = token

    return modified_prefix_expression


def identify_constants(prefix_expression: list[str], constants: list[str] | None = None, inplace: bool = False, convert_numbers_to_constant: bool = True) -> tuple[list[str], list[str]]:
    '''
    Replace all '<constant>' tokens in a prefix expression with constants named 'C_i'.
    This allows the expression to be compiled into a function.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to replace the '<constant>' tokens in.
    constants : list[str] | None
        The constants to use in the expression.
    inplace : bool
        Whether to modify the expression in place.

    Returns
    -------
    tuple[list[str], list[str]]
        The prefix expression with the constants replaced and the list of constants used.
    '''
    if inplace:
        modified_prefix_expression = prefix_expression
    else:
        modified_prefix_expression = prefix_expression.copy()

    constant_index = 0
    if constants is None:
        constants = []
    else:
        constants = list(constants)

    for i, token in enumerate(prefix_expression):
        if token == "<constant>" or (convert_numbers_to_constant and (re.match(r"C_\d+", token) or token.isnumeric())):
            if constants is not None and len(constants) > constant_index:
                modified_prefix_expression[i] = constants[constant_index]
            else:
                modified_prefix_expression[i] = f"C_{constant_index}"
            constants.append(f"C_{constant_index}")
            constant_index += 1

    return modified_prefix_expression, constants


def flatten_nested_list(nested_list: list) -> list[str]:
    '''
    Flatten a nested list.

    Parameters
    ----------
    nested_list : list
        The nested list to flatten.

    Returns
    -------
    list[str]
        The flattened list.
    '''
    flat_list: list[str] = []
    stack = [nested_list]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            flat_list.append(current)
    return flat_list


def generate_ubi_dist(max_n_operators: int, n_leaves: int, n_unary_operators: int, n_binary_operators: int) -> list[list[int]]:
    '''
    Precompute the number of possible trees for a given number of operators and leaves.

    Parameters
    ----------
    max_n_operators : int
        The maximum number of operators.
    n_leaves : int
        The number of leaves.
    n_unary_operators : int
        The number of unary operators.
    n_binary_operators : int
        The number of binary operators.

    Notes
    -----
    See https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/blob/main/src/nesymres/dataset/generator.py
    '''
    # enumerate possible trees
    # first generate the tranposed version of D, then transpose it
    D: list[list[int]] = []
    D.append([0] + ([n_leaves ** i for i in range(1, 2 * max_n_operators + 1)]))
    for n in range(1, 2 * max_n_operators + 1):  # number of operators
        s = [0]
        for e in range(1, 2 * max_n_operators - n + 1):  # number of empty nodes
            s.append(n_leaves * s[e - 1] + n_unary_operators * D[n - 1][e] + n_binary_operators * D[n - 1][e + 1])
        D.append(s)
    assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
    D = [[D[j][i] for j in range(len(D)) if i < len(D[j])] for i in range(max(len(x) for x in D))]
    return D


def is_prime(n: int) -> bool:
    '''
    Check if a number is prime.

    Parameters
    ----------
    n : int
        The number to check.

    Returns
    -------
    bool
        True if the number is prime, False otherwise.
    '''
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


# --- Simple, base distributions that sample data points ---

def uniform_dist(low: float, high: float, min_value: float | None = None, max_value: float | None = None, size: Any = 1) -> np.ndarray:
    """Samples from a uniform distribution, with optional clipping."""
    # Ensure low <= high for sampling
    low, high = min(low, high), max(low, high)
    samples = np.random.uniform(low, high, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def normal_dist(loc: float, scale: float, min_value: float | None = None, max_value: float | None = None, size: Any = 1) -> np.ndarray:
    """Samples from a normal distribution, with optional clipping."""
    # Ensure scale is non-negative
    scale = max(scale, 1e-9)
    samples = np.random.normal(loc, scale, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def log_uniform_dist(low: float, high: float, min_value: float | None = None, max_value: float | None = None, size: Any = 1) -> np.ndarray:
    """Samples from a log-uniform distribution, with optional clipping."""
    low, high = min(low, high), max(low, high)
    samples = np.exp(np.random.uniform(np.log(low), np.log(high), size=size))
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def log_normal_dist(mean: float, sigma: float, min_value: float | None = None, max_value: float | None = None, size: Any = 1) -> np.ndarray:
    """Samples from a log-normal distribution, with optional clipping."""
    sigma = max(sigma, 1e-9)
    samples = np.random.lognormal(mean, sigma, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def gamma_dist(shape: float, scale: float, min_value: float | None = None, max_value: float | None = None, size: Any = 1) -> np.ndarray:
    """Samples from a gamma distribution, with optional clipping."""
    samples = np.random.gamma(shape, scale, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


# Dictionary of our base callable functions
BASE_DISTRIBUTIONS: dict[str, Callable[..., np.ndarray]] = {
    'uniform': uniform_dist,
    'normal': normal_dist,
    'log_uniform': log_uniform_dist,
    'log_normal': log_normal_dist,
    'gamma': gamma_dist,
}


def sampler_dist(
    base_dist_name: str,
    param_samplers: dict[str, Callable[[], np.ndarray]],
    base_kwargs: dict[str, Any] | None = None,
    size: Any = 1
) -> np.ndarray:
    """
    Generates samples by first dynamically sampling parameters for a base distribution.
    """
    if base_dist_name not in BASE_DISTRIBUTIONS:
        raise ValueError(f"Unknown base_dist_name: {base_dist_name}")

    # Initialize kwargs for the base distribution with any fixed values
    final_kwargs = base_kwargs.copy() if base_kwargs else {}

    # Sample the dynamic parameters
    for param_name, sampler_func in param_samplers.items():
        # Each sampler returns a single value for its parameter
        final_kwargs[param_name] = sampler_func(size=1)[0]  # type: ignore

    # Get the final base distribution function and call it
    base_dist_func = BASE_DISTRIBUTIONS[base_dist_name]
    return base_dist_func(**final_kwargs, size=size)


def get_distribution(config: dict[str, Any]) -> Callable[..., np.ndarray]:
    """
    Factory to get a distribution function from a configuration dictionary.
    Handles simple, constant, and nested sampler distributions recursively.
    """
    name = config['name']
    kwargs = config.get('kwargs', {})

    if name == 'constant':
        return lambda size=1: np.full(size, kwargs['value'])

    if name in BASE_DISTRIBUTIONS:
        return partial(BASE_DISTRIBUTIONS[name], **kwargs)

    if name == 'sampler':
        # --- This is the recursive part ---
        # Resolve the sampler functions for each parameter by calling this factory again
        resolved_samplers = {
            param_name: get_distribution(sampler_config)
            for param_name, sampler_config in kwargs['param_samplers'].items()
        }

        # Prepare the arguments for the sampler_dist function
        sampler_args = {
            'base_dist_name': kwargs['base_dist_name'],
            'param_samplers': resolved_samplers,
            'base_kwargs': kwargs.get('base_kwargs', {})
        }
        return partial(sampler_dist, **sampler_args)

    raise ValueError(f"Unknown distribution name: {name}")


def safe_f(f: Callable, X: np.ndarray, constants: np.ndarray | None = None) -> np.ndarray:
    if constants is None:
        y = f(*X.T)
    else:
        y = f(*X.T, *constants)
    if not isinstance(y, np.ndarray) or y.shape[0] == 1:
        y = np.full(X.shape[0], y)
    return y


def remap_expression(source_expression: list[str], dummy_variables: list[str], variable_mapping: dict | None = None) -> tuple[list[str], dict]:
    source_expression = deepcopy(source_expression)
    if variable_mapping is None:
        variable_mapping = {}
        for i, token in enumerate(source_expression):
            if token in dummy_variables:
                if token not in variable_mapping:
                    variable_mapping[token] = f'_{len(variable_mapping)}'

    for i, token in enumerate(source_expression):
        if token in dummy_variables:
            source_expression[i] = variable_mapping[token]

    return source_expression, variable_mapping


def deduplicate_rules(rules_list: list[tuple[tuple[str, ...], tuple[str, ...]]], dummy_variables: list[str]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    deduplicated_rules: dict[tuple[str, ...], tuple[str, ...]] = {}
    for rule in rules_list:
        # Rename variables in the source expression
        remapped_source, variable_mapping = remap_expression(list(rule[0]), dummy_variables=dummy_variables)
        remapped_target, _ = remap_expression(list(rule[1]), dummy_variables, variable_mapping)

        remapped_source_key = tuple(remapped_source)
        remapped_target_value = tuple(remapped_target)

        existing_replacement = deduplicated_rules.get(remapped_source_key)
        if existing_replacement is None or len(remapped_target_value) < len(existing_replacement):
            # Found a better (shorter) target expression for the same source
            deduplicated_rules[remapped_source_key] = remapped_target_value

    return list(deduplicated_rules.items())

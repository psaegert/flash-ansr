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


def uniform_dist(low: float, high: float, min_value: int | None, max_value: int | None, size: Any = 1) -> np.ndarray:
    if min_value is None and max_value is None:
        return np.array(np.random.uniform(float(low), float(high), size=size))
    return np.clip(np.array(np.random.uniform(float(low), float(high), size=size)), min_value, max_value)


def uniform_uniform_intervals_dist(low: float, high: float, min_value: int | None, max_value: int | None, size: Any = 1) -> np.ndarray:
    lower_extreme, higher_extreme = np.sort(np.random.uniform(float(low), float(high), size=2), axis=-1)
    if min_value is None and max_value is None:
        return np.array(np.random.uniform(lower_extreme, higher_extreme, size=size))
    return np.clip(np.array(np.random.uniform(lower_extreme, higher_extreme, size=size)), min_value, max_value)


def normal_uniform_intervals_dist(loc: float, scale: float, min_value: int | None, max_value: int | None, size: Any = 1) -> np.ndarray:
    lower_extreme, higher_extreme = np.sort(np.random.normal(float(loc), float(scale), size=2), axis=-1)
    if min_value is None and max_value is None:
        return np.array(np.random.uniform(lower_extreme, higher_extreme, size=size))
    return np.clip(np.array(np.random.uniform(lower_extreme, higher_extreme, size=size)), min_value, max_value)


def normal_normal_intervals_dist(loc_mean: float, scale_mean: float, alpha_std: float, beta_std: float, min_value: int | None, max_value: int | None, size: Any = 1) -> np.ndarray:
    loc = np.random.normal(float(loc_mean), float(scale_mean), size=1)
    scale = np.random.gamma(float(alpha_std), float(beta_std), size=1)
    if min_value is None and max_value is None:
        return np.array(np.random.normal(loc, scale, size=size))
    return np.clip(np.array(np.random.normal(loc, scale, size=size)), min_value, max_value)


def log_uniform_dist(low: float, high: float, min_value: int | None, max_value: int | None, size: Any = 1) -> np.ndarray:
    if min_value is None and max_value is None:
        return np.array(np.exp(np.random.uniform(np.log(float(low)), np.log(float(high)), size=size)))
    return np.clip(np.exp(np.random.uniform(np.log(float(low)), np.log(float(high)), size=size)), min_value, max_value)


def normal_dist(loc: float, scale: float, min_value: int | None, max_value: int | None, size: Any = 1) -> np.ndarray:
    if min_value is None and max_value is None:
        return np.array(np.random.normal(float(loc), float(scale), size=size))
    return np.clip(np.array(np.random.normal(float(loc), float(scale), size=size)), min_value, max_value)


def gamma_dist(shape: float, scale: float, min_value: int | None, max_value: int | None, size: Any = 1) -> np.ndarray:
    if min_value is None and max_value is None:
        return np.array(np.random.gamma(float(shape), float(scale), size=size))
    return np.clip(np.random.gamma(float(shape), float(scale), size=size), min_value, max_value)


def get_distribution(distribution: str | Callable[..., np.ndarray], distribution_kwargs: dict[str, Any]) -> Callable[..., np.ndarray]:
    '''
    Get the distribution function from a string or function.

    Parameters
    ----------
    distribution : str or Callable[..., np.ndarray]
        The distribution to use.
    distribution_kwargs : dict[str, Any]
        The keyword arguments to pass to the distribution function.

    Returns
    -------
    Callable[..., np.ndarray]
        The distribution function.
    '''
    if distribution == 'uniform':
        return partial(uniform_dist, low=distribution_kwargs['low'], high=distribution_kwargs['high'], min_value=distribution_kwargs.get('min_value'), max_value=distribution_kwargs.get('max_value'))
    if distribution == 'uniform_uniform_intervals':
        return partial(uniform_uniform_intervals_dist, low=distribution_kwargs['low'], high=distribution_kwargs['high'], min_value=distribution_kwargs.get('min_value'), max_value=distribution_kwargs.get('max_value'))
    if distribution == 'normal_uniform_intervals':
        return partial(normal_uniform_intervals_dist, loc=distribution_kwargs['loc'], scale=distribution_kwargs['scale'], min_value=distribution_kwargs.get('min_value'), max_value=distribution_kwargs.get('max_value'))
    if distribution == 'normal_normal_intervals':
        return partial(normal_normal_intervals_dist, loc_mean=distribution_kwargs['loc_mean'], scale_mean=distribution_kwargs['scale_mean'], alpha_std=distribution_kwargs['alpha_std'], beta_std=distribution_kwargs['beta_std'], min_value=distribution_kwargs.get('min_value'), max_value=distribution_kwargs.get('max_value'))
    if distribution == 'log_uniform':
        return partial(log_uniform_dist, low=distribution_kwargs['low'], high=distribution_kwargs['high'], min_value=distribution_kwargs.get('min_value'), max_value=distribution_kwargs.get('max_value'))
    if distribution == 'normal':
        return partial(normal_dist, loc=distribution_kwargs['loc'], scale=distribution_kwargs['scale'], min_value=distribution_kwargs.get('min_value'), max_value=distribution_kwargs.get('max_value'))
    if distribution == 'gamma':
        return partial(gamma_dist, shape=distribution_kwargs['shape'], scale=distribution_kwargs['scale'], min_value=distribution_kwargs.get('min_value'), max_value=distribution_kwargs.get('max_value'))
    if callable(distribution):
        return partial(distribution, **distribution_kwargs)

    raise ValueError(f'Distribution must be a function (int -> float) or one of ["uniform", "log_uniform", "normal"], got {distribution}')


def get_multi_distribution(distributions: list[tuple[float, str | Callable[..., np.ndarray], dict[str, Any]]]) -> Callable[..., np.ndarray]:
    '''
    Get a mixture distribution from a list of distributions and their weights.

    Parameters
    ----------
    distributions : list[tuple[str | Callable[..., np.ndarray], dict[str, Any], float]]
        The distributions to use and their weights.

    Returns
    -------
    Callable[..., np.ndarray]
        The mixture distribution function.
    '''
    dists = [get_distribution(dist, dist_kwargs) for weight, dist, dist_kwargs in distributions]
    weights = np.array([weight for weight, dist, dist_kwargs in distributions], dtype=np.float64)
    weights = weights / weights.sum()

    def mixture_distribution(size: Any = 1) -> np.ndarray:
        distribution_choice = np.random.choice(len(dists), size=1, p=weights)
        return dists[distribution_choice[0]](size=size)

    return mixture_distribution


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

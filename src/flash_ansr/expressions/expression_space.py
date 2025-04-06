import re
import importlib
import fractions
from typing import Any, Callable, Literal, Generator
from copy import deepcopy
from types import CodeType, FunctionType
from math import prod
import signal
import os
import itertools
from collections import defaultdict
import warnings
import time

import numpy as np
import json
from sympy import simplify, parse_expr
from scipy.optimize import curve_fit, OptimizeWarning
from tqdm import tqdm

from flash_ansr.models.transformer_utils import Tokenizer
from flash_ansr.utils import load_config, substitute_root_path
from flash_ansr.expressions.utils import get_used_modules, numbers_to_num, flatten_nested_list, is_prime, num_to_constants, codify, safe_f, deduplicate_rules
from flash_ansr.expressions.simplify import _simplify_flash


def timeout_handler(signum: Any, frame: Any) -> None:
    raise TimeoutError("Timed out")


signal.signal(signal.SIGALRM, timeout_handler)


class ExpressionSpace:
    """
    Management and manipulation of expressions / equations with properties and methods for parsing, encoding, decoding, and transforming equations

    Parameters
    ----------
    operators : dict[str, dict[str, Any]]
        A dictionary of operators with their properties
    variables : int
        The number of variables
    """
    def __init__(self, operators: dict[str, dict[str, Any]], variables: int, simplification: Literal['flash', 'auto_flash', 'sympy'] = 'flash', simplification_kwargs: dict[str, Any] | None = None, special_tokens: list[str] | None = None) -> None:
        self.simplification = simplification
        self.simplification_kwargs = simplification_kwargs or {}

        self.special_constants = {"pi": np.pi}

        self.operator_tokens = list(operators.keys())

        self.operator_aliases = {alias: operator for operator, properties in operators.items() for alias in properties['alias']}

        self.operator_inverses = {k: v["inverse"] for k, v in operators.items() if v.get("inverse") is not None}
        self.inverse_base = {
            '*': ['inv', '/', '<1>'],
            '+': ['neg', '-', '<0>'],
        }
        self.inverse_unary = {v[0]: [k, v[1], v[2]] for k, v in self.inverse_base.items()}
        self.inverse_binary = {v[1]: [k, v[0], v[2]] for k, v in self.inverse_base.items()}

        self.unary_mult_div_operators = {k: v["inverse"] for k, v in operators.items() if k.startswith('mult') or k.startswith('div')}

        self.commutative_operators = [k for k, v in operators.items() if v.get("commutative", False)]

        self.symmetric_operators = [k for k, v in operators.items() if v.get("symmetry", 0) == 1]
        self.antisymmetric_operators = [k for k, v in operators.items() if v.get("symmetry", 0) == -1]

        self.positive_operators = [k for k, v in operators.items() if v.get("positive", False)]

        self.monotonous_increasing_operators = [k for k, v in operators.items() if v.get("monotonicity", 0) == 1]
        self.monotonous_decreasing_operators = [k for k, v in operators.items() if v.get("monotonicity", 0) == -1]

        self.operator_realizations = {k: v["realization"] for k, v in operators.items()}
        self.realization_to_operator = {v: k for k, v in self.operator_realizations.items()}

        self.operator_precedence_compat = {k: v.get("precedence", i) for i, (k, v) in enumerate(operators.items())}
        self.operator_precedence_compat['**'] = 3  # FIXME: Don't hardcode this
        self.operator_precedence_compat['sqrt'] = 3  # FIXME: Don't hardcode this

        self.operator_weights = {k: v.get("weight", 1.0) for k, v in operators.items()}
        total_weight = sum(self.operator_weights.values())
        if total_weight > 0:
            self.operator_weights = {k: v / total_weight for k, v in self.operator_weights.items()}

        self.operator_arity = {k: v["arity"] for k, v in operators.items()}
        self.operator_arity_compat = deepcopy(self.operator_arity)
        self.operator_arity_compat['**'] = 2

        self.max_power = max([int(op[3:]) for op in self.operator_tokens if re.match(r'pow\d+(?!\_)', op)] + [0])
        self.max_fractional_power = max([int(op[5:]) for op in self.operator_tokens if re.match(r'pow1_\d+', op)] + [0])

        self.n_variables = variables
        self.variables = [f'x{i + 1}' for i in range(variables)]

        self.tokenizer = Tokenizer(vocab=self.operator_tokens + self.variables, special_tokens=special_tokens)

        self.modules = get_used_modules(''.join(f"{op}(" for op in self.operator_realizations.values()))  # HACK: This can be done more elegantly for sure

        self.connection_classes = {
            'add': (set(['+', '-']), "0"),
            'mult': (set(['*', '/']), "1"),
        }

        self.operator_to_class = {
            '+': 'add',
            '-': 'add',
            '*': 'mult',
            '/': 'mult'
        }

        self.connection_classes_inverse = {
            'add': "neg",
            'mult': "inv",
        }

        self.connectable_operators = set(['+', '-', '*', '/'])

        self.import_modules()

        if simplification == 'auto_flash':
            dummy_variables = [f'x{i}' for i in range(100)]  # HACK
            if not os.path.exists(substitute_root_path(self.simplification_kwargs['rules_file'])):
                self.simplification_rules: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
            else:
                with open(substitute_root_path(self.simplification_kwargs['rules_file']), 'r') as f:
                    self.simplification_rules = deduplicate_rules(json.load(f), dummy_variables=dummy_variables)

            self.simplification_rules_trees: dict[tuple, list[tuple[list, list]]] = self.rules_trees_from_rules_list(self.simplification_rules, dummy_variables=dummy_variables)  # HACK

    def import_modules(self) -> None:  # TODO. Still necessary?
        for module in self.modules:
            if module not in globals():
                globals()[module] = importlib.import_module(module)

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "ExpressionSpace":
        '''
        Load an ExpressionSpace from a configuration file or dictionary.

        Parameters
        ----------
        config : dict[str, Any] | str
            The configuration file or dictionary.

        Returns
        -------
        ExpressionSpace
            The ExpressionSpace object.
        '''
        config_ = load_config(config, resolve_paths=True)

        if "expressions" in config_.keys():
            config_ = config_["expressions"]

        return cls(operators=config_["operators"], variables=config_["variables"], simplification=config_.get("simplification", 'flash'), simplification_kwargs=config_.get("simplification_kwargs"), special_tokens=config_.get("special_tokens", None))

    def is_valid(self, prefix_expression: list[str], verbose: bool = False) -> bool:
        '''
        Check if a prefix expression is valid.

        Parameters
        ----------
        prefix_expression : list[str]
            The prefix expression.
        verbose : bool, optional
            Whether to print error messages, by default False.

        Returns
        -------
        bool
            Whether the expression is valid.
        '''
        stack: list[str] = []

        if len(prefix_expression) > 1 and prefix_expression[0] in self.variables:
            if verbose:
                print(f'Invalid expression {prefix_expression}: Variable must be leaf node')
            return False

        for token in reversed(prefix_expression):
            if token not in self.tokenizer.vocab and not token == '<num>':
                try:
                    float(token)
                except ValueError:
                    if verbose:
                        print(f'Invalid token {token} in expression {prefix_expression}')
                    return False

            if token in self.operator_arity:
                if len(stack) < self.operator_arity[token]:
                    if verbose:
                        print(f'Not enough operands for operator {token} in expression {prefix_expression}')
                    return False

                # Consume the operands based on the arity of the operator
                for _ in range(self.operator_arity[token]):
                    stack.pop()

            # Add the token to the stack
            stack.append(token)

        if len(stack) != 1:
            if verbose:
                print(f'Stack is not empty after parsing the expression {prefix_expression}')
            return False

        return True

    def _deparenthesize(self, term: str) -> str:
        '''
        Removes outer parentheses from a term.

        Parameters
        ----------
        term : str
            The term.

        Returns
        -------
        str
            The term without parentheses.
        '''
        if term.startswith('(') and term.endswith(')'):
            return term[1:-1]
        return term

    def prefix_to_infix(self, tokens: list[str], power: Literal['func', '**'] = 'func', realization: bool = False) -> str:
        '''
        Convert a list of tokens in prefix notation to infix notation

        Parameters
        ----------
        tokens : list[str]
            List of tokens in prefix notation
        power : Literal['func', '**'], optional
            Whether to use the 'func' or '**' notation for power operators, by default 'func'
        realization : bool, optional
            Whether to use the realization (python code) of the operators, by default False

        Returns
        -------
        str
            The infix notation of the expression
        '''
        stack: list[str] = []

        for token in reversed(tokens):
            operator = self.realization_to_operator.get(token, token)
            operator_realization = self.operator_realizations.get(operator, operator)
            if operator in self.operator_tokens or operator in self.operator_aliases or operator in self.operator_arity_compat:
                write_operator = operator_realization if realization else operator
                write_operands = [stack.pop() for _ in range(self.operator_arity_compat[operator])]

                # If the operator is a power operator, format it as
                # "pow(operand1, operand2)" if power is 'func'
                # "operand1**operand2" if power is '**'
                # This regex must not match pow1_2 or pow1_3
                if re.match(r'pow\d+(?!\_)', operator) and power == '**':
                    exponent = int(operator[3:])
                    stack.append(f'(({write_operands[0]})**{exponent})')

                # If the operator is a fractional power operator such as pow1_2, format it as
                # "pow(operand1, 0.5)" if power is 'func'
                # "operand1**0.5" if power is '**'
                elif re.match(r'pow1_\d+', operator) and power == '**':
                    exponent = int(operator[5:])
                    stack.append(f'(({write_operands[0]})**(1/{exponent}))')

                # If the operator is a function from a module, format it as
                # "module.function(operand1, operand2, ...)"
                elif '.' in operator_realization or self.operator_arity_compat[operator] > 2:
                    # No need for parentheses here
                    stack.append(f'{write_operator}({", ".join([self._deparenthesize(operand) for operand in write_operands])})')

                # ** stays **
                elif self.operator_aliases.get(operator, operator) == '**':
                    stack.append(f'({write_operands[0]} {write_operator} {write_operands[1]})')

                # If the operator is a binary operator, format it as
                # "(operand1 operator operand2)"
                elif self.operator_arity_compat[operator] == 2:
                    stack.append(f'({write_operands[0]} {write_operator} {write_operands[1]})')

                elif operator == 'neg':
                    stack.append(f'-{write_operands[0]}')

                elif operator == 'inv':
                    stack.append(f'(1/{write_operands[0]})')

                else:
                    stack.append(f'{write_operator}({", ".join([self._deparenthesize(operand) for operand in write_operands])})')

            else:
                stack.append(token)

        infix_expression = stack.pop()

        return self._deparenthesize(infix_expression)  # FIXME: Sometimes result in "1 + x) / (2 * x" instead of "(1 + x) / (2 * x)"

    def infix_to_prefix(self, infix_expression: str) -> list[str]:
        '''
        Convert an infix expression to a prefix expression

        Parameters
        ----------
        infix_expression : str
            The infix expression

        Returns
        -------
        list[str]
            The prefix expression
        '''
        # Regex to tokenize expression properly (handles floating-point numbers)
        token_pattern = re.compile(r'\d+\.\d+|\d+|[A-Za-z_][\w.]*|\*\*|[-+*/()]')

        # Tokenize the infix expression
        tokens = token_pattern.findall(infix_expression.replace(' ', ''))

        stack: list[str] = []
        prefix_expr: list[str] = []

        # Reverse the tokens for right-to-left parsing
        tokens = tokens[::-1]

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Handle numbers (integers or floats)
            if re.match(r'\d+\.\d+|\d+', token):  # Match positive or negative floats and integers
                prefix_expr.append(token)
            elif re.match(r'[A-Za-z_][\w.]*', token):  # Match functions and variables
                prefix_expr.append(token)
            elif token == ')':
                stack.append(token)
            elif token == '(':
                while stack and stack[-1] != ')':
                    prefix_expr.append(stack.pop())
                if stack and stack[-1] == ')':
                    stack.pop()  # Pop the ')'
            else:
                # Handle binary and unary operators
                if token == '-' and (i == len(tokens) - 1 or tokens[i + 1] == '(' or (tokens[i + 1]) in self.operator_precedence_compat):
                    # Handle unary negation (not part of a number)
                    token = 'neg'

                if stack and stack[-1] != ')' and token != ')':
                    while stack and self.operator_precedence_compat.get(stack[-1], 0) >= self.operator_precedence_compat.get(token, 0):
                        prefix_expr.append(stack.pop())
                    stack.append(token)
                else:

                    if (token == 'neg' and not stack) or (stack and stack[-1] != ')'):
                        stack.insert(-1, token)
                    else:
                        stack.append(token)

            i += 1

        while stack:
            prefix_expr.append(stack.pop())

        return prefix_expr[::-1]

    def factorize_to_at_most(self, p: int, max_factor: int, max_iter: int = 1000) -> list[int]:
        '''
        Factorize an integer into factors at most max_factor

        Parameters
        ----------
        p : int
            The integer to factorize
        max_factor : int
            The maximum factor
        max_iter : int, optional
            The maximum number of iterations, by default 1000

        Returns
        -------
        list[int]
            The factors of the integer
        '''
        if is_prime(p):
            return [p]
        p_factors = []
        i = 0
        while p > 1:
            for j in range(max_factor, 0, -1):
                if j == 1:
                    p_factors.append(p)
                    p = 1
                    break
                if p % j == 0:
                    p_factors.append(j)
                    p //= j
                    break
            i += 1
            if i > max_iter:
                raise ValueError(f'Factorization of {p} into at most {max_factor} factors failed after {max_iter} iterations')

        return p_factors

    def convert_expression(self, prefix_expr: list[str]) -> list[str]:
        '''
        Convert an expression to a supported form

        Parameters
        ----------
        prefix_expr : list[str]
            The prefix expression

        Returns
        -------
        list[str]
            The converted expression
        '''
        stack: list = []
        i = len(prefix_expr) - 1

        while i >= 0:
            token = prefix_expr[i]

            if token in self.operator_arity_compat or token in self.operator_aliases or re.match(r'pow\d+(?!\_)', token) or re.match(r'pow1_\d+', token):
                operator = self.operator_aliases.get(token, token)
                arity = self.operator_arity_compat[operator]

                if operator == 'neg':
                    # If the operand of neg is a number, combine them
                    if isinstance(stack[-1][0], str):
                        if re.match(r'\d+\.\d+|\d+', stack[-1][0]):
                            stack[-1][0] = f'-{stack[-1][0]}'
                        elif re.match(r'-\d+\.\d+|-\d+', stack[-1][0]):
                            stack[-1][0] = stack[-1][0][1:]
                        else:
                            # General case: assemble operator and its operands
                            operands = [stack.pop() for _ in range(arity)]
                            stack.append([operator, operands])
                    else:
                        # General case: assemble operator and its operands+
                        operands = [stack.pop() for _ in range(arity)]
                        stack.append([operator, operands])

                elif operator == '**':
                    # Check for floating-point exponent
                    base = stack.pop()
                    exponent = stack.pop()

                    if len(exponent) == 1:
                        if re.match(r'-?\d+$', exponent[0]):  # Integer exponent
                            exponent_value: int | float = int(exponent[0])
                            pow_operator = f'pow{abs(exponent_value)}'
                            if exponent_value < 0:
                                stack.append(['inv', [[pow_operator, [base]]]])
                            else:
                                stack.append([pow_operator, [base]])
                        elif re.match(r'-?\d*\.\d+$', exponent[0]):  # Floating-point exponent
                            exponent_value = float(exponent[0])

                            # Try to convert the exponent into a fraction
                            abs_exponent_fraction = fractions.Fraction(abs(float(exponent[0]))).limit_denominator()
                            if abs_exponent_fraction.numerator <= 5 and abs_exponent_fraction.denominator <= 5:
                                # Format the fraction as a combination of power operators, i.e. "x**(2/3)" -> "pow1_3(pow2(x))"
                                new_expression = [base]
                                if abs_exponent_fraction.numerator != 1:
                                    new_expression = [f'pow{abs_exponent_fraction.numerator}', new_expression]
                                if abs_exponent_fraction.denominator != 1:
                                    new_expression = [f'pow1_{abs_exponent_fraction.denominator}', new_expression]
                                if exponent_value < 0:
                                    new_expression = ['inv', new_expression]
                                stack.append(new_expression)
                            else:
                                # Replace '** base exponent' with 'exp(log(base) * exponent)'
                                stack.append(['exp', [['*', [['log', [base]], exponent]]]])
                        else:
                            # Replace '** base exponent' with 'exp(log(base) * exponent)'
                            stack.append(['exp', [['*', [['log', [base]], exponent]]]])
                    elif len(exponent) == 2 and exponent[0][0] == '/' and \
                            isinstance(exponent[1][0][0], str) and re.match(r'-?\d+$', exponent[1][0][0]) and \
                            isinstance(exponent[1][1][0], str) and re.match(r'-?\d+$', exponent[1][1][0]):
                        exponent_value = int(exponent[1][0][0]) / int(exponent[1][1][0])
                        abs_exponent_fraction = fractions.Fraction(abs(exponent_value)).limit_denominator()
                        if abs_exponent_fraction.numerator <= 5 and abs_exponent_fraction.denominator <= 5:
                            # Format the fraction as a combination of power operators, i.e. "x**(2/3)" -> "pow1_3(pow2(x))"
                            new_expression = [base]
                            if abs_exponent_fraction.numerator != 1:
                                new_expression = [f'pow{abs_exponent_fraction.numerator}', new_expression]
                            if abs_exponent_fraction.denominator != 1:
                                new_expression = [f'pow1_{abs_exponent_fraction.denominator}', new_expression]
                            if exponent_value < 0:
                                new_expression = ['inv', new_expression]
                            stack.append(new_expression)
                        else:
                            stack.append(['exp', [['*', [['log', [base]], exponent]]]])
                    else:
                        # Replace '** base exponent' with 'exp(log(base) * exponent)'
                        stack.append(['exp', [['*', [['log', [base]], exponent]]]])

                else:
                    # General case: assemble operator and its operands
                    operands = [stack.pop() for _ in range(arity)]
                    stack.append([operator, operands])
            else:
                # Non-operator token (operand)
                stack.append([token])

            i -= 1

        need_to_convert_powers_expression = flatten_nested_list(stack)[::-1]

        stack = []
        i = len(need_to_convert_powers_expression) - 1

        while i >= 0:
            token = need_to_convert_powers_expression[i]

            if token in self.operator_arity_compat or token in self.operator_aliases or re.match(r'pow\d+(?!\_)', token) or re.match(r'pow1_\d+', token):
                operator = self.operator_aliases.get(token, token)
                arity = self.operator_arity_compat.get(operator, 1)
                operands = list(reversed(stack[-arity:]))

                if operator.startswith('pow'):
                    # Identify chains of pow<i> xor pow1_<i> operators
                    # Mixed chains are ignored
                    operator_chain = [operator]
                    current_operand = operands[0]

                    operator_bases = ['pow1_', 'pow']
                    operator_patterns = [r'pow1_\d+', r'pow\d+']
                    operator_patterns_grouped = [r'pow1_(\d+)', r'pow(\d+)']
                    max_powers = [self.max_fractional_power, self.max_power]
                    for base, pattern, pattern_grouped, p in zip(operator_bases, operator_patterns, operator_patterns_grouped, max_powers):
                        if re.match(pattern, operator):
                            operator_base = base
                            operator_pattern = pattern
                            operator_pattern_grouped = pattern_grouped
                            max_power = p
                            break

                    while len(current_operand) == 2 and re.match(operator_pattern, current_operand[0]):
                        operator_chain.append(current_operand[0])
                        current_operand = current_operand[1]

                    if len(operator_chain) > 0:
                        p = prod(int(re.match(operator_pattern_grouped, op).group(1)) for op in operator_chain)  # type: ignore

                        # Factorize p into at most self.max_power or self.max_fractional_power
                        p_factors = self.factorize_to_at_most(p, max_power)

                        # Construct the new operators
                        new_operators = []
                        for p in p_factors:
                            new_operators.append(f'{operator_base}{p}')

                        if len(new_operators) == 0:
                            new_chain = current_operand
                        else:
                            new_chain = [new_operators[-1], [current_operand]]
                            for op in new_operators[-2::-1]:
                                new_chain = [op, [new_chain]]

                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(new_chain)
                        i -= 1
                        continue

                _ = [stack.pop() for _ in range(arity)]
                stack.append([operator, operands])
                i -= 1
                continue

            else:
                stack.append([token])
                i -= 1

        return flatten_nested_list(stack)[::-1]

    # PARSING
    def parse_expression(
            self,
            infix_expression: str,
            substitute_special_constants: bool = True,
            convert_expression: bool = True,
            convert_variable_names: bool = True,
            mask_numbers: bool = False,
            too_many_variables: Literal['ignore', 'raise'] = 'ignore') -> list[str]:
        '''
        Parse an infix expression into a prefix expression

        Parameters
        ----------
        infix_expression : str
            The infix expression
        substitute_special_constants : bool, optional
            Whether to substitute special constants, by default True
        convert_expression : bool, optional
            Whether to convert the expression, by default True
        convert_variable_names : bool, optional
            Whether to convert variable names, by default True
        mask_numbers : bool, optional
            Whether to mask numbers, by default False
        too_many_variables : Literal['ignore', 'raise'], optional
            Whether to ignore or raise an error if there are too many variables, by default 'ignore'

        Returns
        -------
        list[str]
            The prefix expression
        '''

        parsed_expression = self.infix_to_prefix(infix_expression)

        if substitute_special_constants:
            parsed_expression = self.numerify_special_constants(parsed_expression, inplace=True)
        if convert_variable_names:
            parsed_expression = self.convert_variable_names(parsed_expression, too_many_variables=too_many_variables)
        if convert_expression:
            parsed_expression = self.convert_expression(parsed_expression)
        if mask_numbers:
            parsed_expression = numbers_to_num(parsed_expression, inplace=True)

        return self.remove_pow1(parsed_expression)  # HACK: Find a better place to put this

    def remove_pow1(self, prefix_expression: list[str]) -> list[str]:
        filtered_expression = []
        for token in prefix_expression:
            if token == 'pow1':
                continue

            if token == 'pow_1':
                filtered_expression.append('inv')
                continue

            filtered_expression.append(token)

        return filtered_expression

    def extract_expression_from_beam(self, beam: list[int] | list[str]) -> tuple[list[int] | list[str], list[int] | list[str], list[int] | list[str]]:
        '''
        Extract the expression from a beam. The expression starts with the <bos> token and ends with the <eos> token.

        Parameters
        ----------
        beam : list[int] | list[str]
            The beam to extract the expression from.

        Returns
        -------
        list[int] | list[str]
            The expression
        list[int] | list[str]
            The prefix of the expression
        list[int] | list[str]
            The suffix of the expression
        '''
        if not isinstance(beam, list):
            beam = list(beam)

        if isinstance(beam[0], (int, np.int64, np.int32)):
            bos_position = beam.index(self.tokenizer["<bos>"]) if self.tokenizer["<bos>"] in beam else 0
            eos_position = beam.index(self.tokenizer["<eos>"]) if self.tokenizer["<eos>"] in beam else len(beam)
        elif isinstance(beam[0], str):
            bos_position = beam.index('<bos>') if '<bos>' in beam else 0  # type: ignore
            eos_position = beam.index('<eos>') if '<eos>' in beam else len(beam)  # type: ignore
        else:
            raise ValueError("The beam must be a list of integers or strings")

        return beam[bos_position + 1:eos_position], beam[:bos_position + 1], beam[eos_position:]

    # Compatibility
    def convert_variable_names(self, prefix_expr: list[str], too_many_variables: Literal['ignore', 'raise'] = 'ignore') -> list[str]:
        '''
        Convert variable names to a supported form

        Parameters
        ----------
        prefix_expr : list[str]
            The prefix expression
        too_many_variables : Literal['ignore', 'raise'], optional
            Whether to ignore or raise an error if there are too many variables, by default 'ignore'

        Returns
        -------
        list[str]
            The converted expression
        '''
        converted_prefix_expr: list = []
        variable_translation_dict: dict[str, str] = {}

        for token in prefix_expr:
            # If the token is numeric, an operator, or an already existing variable, push it onto the stack
            if token in self.operator_arity_compat or token in self.operator_aliases or token == '<num>' or token in self.variables or re.match(r'-?\d+\.\d+|-?\d+', token) or re.match(r'pow\d+(?!\_)', token) or re.match(r'pow1_\d+', token):
                operator = self.operator_aliases.get(token, token)
                converted_prefix_expr.append(operator)
            else:
                if token not in variable_translation_dict:
                    if len(variable_translation_dict) >= len(self.variables):
                        if too_many_variables == 'raise':
                            raise ValueError(f'Too many variables in expression: {prefix_expr}')

                        if too_many_variables == 'ignore':
                            converted_prefix_expr.append(token)
                            continue

                    variable_translation_dict[token] = self.variables[len(variable_translation_dict)]
                converted_prefix_expr.append(variable_translation_dict[token])

        return converted_prefix_expr

    def numerify_special_constants(self, prefix_expression: list[str], inplace: bool = False) -> list[str]:
        '''
        Replace special constants with their numerical values

        Parameters
        ----------
        prefix_expression : list[str]
            The prefix expression
        inplace : bool, optional
            Whether to modify the expression in place, by default False

        Returns
        -------
        list[str]
            The expression with special constants replaced by their numerical values
        '''
        if inplace:
            modified_prefix_expression = prefix_expression
        else:
            modified_prefix_expression = prefix_expression.copy()

        for i, token in enumerate(prefix_expression):
            if token in self.special_constants:
                modified_prefix_expression[i] = str(self.special_constants[token])

        return modified_prefix_expression

    def remove_num(self, expression: list[str], verbose: bool = False, debug: bool = False) -> list[str]:
        stack: list = []
        i = len(expression) - 1

        if debug:
            print(f'Input expression: {expression}')

        while i >= 0:
            token = expression[i]

            if debug:
                print(f'Stack: {stack}')
                print(f'Processing token {token}')

            if token in self.operator_arity_compat or token in self.operator_aliases:
                operator = self.operator_aliases.get(token, token)
                arity = self.operator_arity_compat[operator]
                operands = list(reversed(stack[-arity:]))

                if any(operand[0] == '<num>' for operand in operands):
                    if verbose:
                        print('Removing constant')

                    non_num_operands = [operand for operand in operands if operand[0] != '<num>']

                    if len(non_num_operands) == 0:
                        new_term = '<num>'
                    elif len(non_num_operands) == 1:
                        new_term = non_num_operands[0]
                    else:
                        raise NotImplementedError('Removing a constant from n-operand operator is not implemented')

                    _ = [stack.pop() for _ in range(arity)]
                    stack.append([new_term])
                    i -= 1
                    continue

                _ = [stack.pop() for _ in range(arity)]
                stack.append([operator, operands])

            else:
                stack.append([token])

            i -= 1

        return flatten_nested_list(stack)[::-1]

    # SIMPLIFICATION (Sympy)
    def simplify_sympy(self, prefix_expression: list[str], ratio: float | None = None, timeout: int = 1) -> list[str]:
        prefix_expression, constants = num_to_constants(list(prefix_expression), inplace=True)

        infix_expression = self.prefix_to_infix(prefix_expression, power='**')

        for c in constants:
            infix_expression = infix_expression.replace(c, str(np.random.uniform(-10, 10)))

        sympy_expression = parse_expr(infix_expression)

        signal.alarm(timeout)
        try:
            simplified_expression = str(simplify(sympy_expression, ratio=ratio) if ratio is not None else simplify(sympy_expression))
        except (TimeoutError, OverflowError):
            return prefix_expression

        translations = {
            'Abs': 'abs',
        }

        for translate_from, translate_to in translations.items():
            simplified_expression = simplified_expression.replace(translate_from, translate_to)

        parsed_expression = self.parse_expression(simplified_expression)

        return numbers_to_num(parsed_expression, inplace=True)

    def simplify(self, prefix_expression: list[str], *args: Any, **kwargs: Any) -> list[str]:
        match self.simplification:
            case 'flash':
                return self.simplify_flash(prefix_expression, *args, **kwargs)
            case 'sympy':
                return self.simplify_sympy(prefix_expression, *args, **kwargs)
            case 'auto_flash':
                return self.simplify_auto_flash(prefix_expression, *args, **kwargs)  # type: ignore
            case _:
                raise ValueError(f'Invalid simplification method: {self.simplification}')

    # AUTO SIMPLIFICATION
    def prefix_to_tree(self, expression: list[str]) -> list:
        def build_tree(index: int) -> tuple[list | None, int]:
            if index >= len(expression):
                return None, index

            token = expression[index]

            # If token is not an operator or is an operator with arity 0
            if isinstance(token, dict) or token not in self.operator_arity or self.operator_arity[token] == 0:
                return [token], index + 1

            # If token is an operator
            operands = []
            current_index = index + 1

            # Process operands based on the operator's arity
            for _ in range(self.operator_arity[token]):
                if current_index >= len(expression):
                    break

                subtree, current_index = build_tree(current_index)
                if subtree:
                    operands.append(subtree)

            return [token, operands], current_index

        result, _ = build_tree(0)

        if result is None:
            raise ValueError(f'Failed to build tree from expression {expression}')

        return result

    def rules_trees_from_rules_list(self, rules_list: list[tuple[tuple[str, ...], tuple[str, ...]]], dummy_variables: list[str]) -> dict[tuple, list[tuple[list, list]]]:
        # Deduplicate the rules (remove patterns that are represented by other patterns, e.g. with other variables)
        deduplicated_rules = deduplicate_rules(rules_list, dummy_variables)

        # Group the rules by arity
        deduplicated_rules_of_operator: defaultdict[str, list] = defaultdict(list)
        for rule in deduplicated_rules:
            deduplicated_rules_of_operator[rule[0][0]].append(rule)
        deduplicated_rules_of_operator = dict(deduplicated_rules_of_operator)  # type: ignore

        # Sort the rules by length of the left-hand side to make matching more efficient
        for operator, deduplicated_rules_of_operator_list in deduplicated_rules_of_operator.items():
            deduplicated_rules_of_operator[operator] = sorted(deduplicated_rules_of_operator_list, key=lambda x: len(x[0]))

        # Construct the trees for pattern matching
        rules_trees = {operator: [
            (
                self.prefix_to_tree(list(rule[0])),
                self.prefix_to_tree(list(rule[1]))
            )
            for rule in deduplicated_rules_of_operator_a] for operator, deduplicated_rules_of_operator_a in deduplicated_rules_of_operator.items()}

        rules_trees_organized: defaultdict[tuple, list] = defaultdict(list)
        for operator, rules in rules_trees.items():
            for (pattern, replacement) in rules:
                operands_heads: list[str] = [operand[0] for operand in pattern[1]]
                if any(head.startswith('_') for head in operands_heads):
                    rules_trees_organized[(operator,)].append((pattern, replacement))
                else:
                    # More specific structure possible
                    rules_key = (operator, *operands_heads)
                    rules_trees_organized[rules_key].append((pattern, replacement))

        return rules_trees_organized

    def match_pattern(self, tree: list, pattern: list, mapping: dict[str, Any] | None = None) -> tuple[bool, dict[str, Any]]:
        if mapping is None:
            mapping = {}

        pattern_length = len(pattern)

        # The leaf node is a variable but the pattern is not
        if len(tree) == 1 and isinstance(tree[0], str) and pattern_length != 1:
            return False, mapping

        # Elementary pattern
        pattern_key = pattern[0]
        if pattern_length == 1 and isinstance(pattern_key, str):
            # Check if the pattern is a placeholder to be filled with the tree
            if pattern_key.startswith('_'):
                # Try to match the tree with the placeholder pattern
                existing_value = mapping.get(pattern_key)
                if existing_value is None:
                    # Placeholder is not yet filled, can be filled with the tree
                    mapping[pattern_key] = tree
                    return True, mapping
                else:
                    # Placeholder is occupied by another tree, the tree does not match the pattern
                    return (existing_value == tree), mapping
            # The literal pattern must match the tree
            return (tree == pattern), mapping

        # The pattern is tree-structured
        tree_operator, tree_operands = tree
        pattern_operator, pattern_operands = pattern

        # If the operators do not match, the tree does not match the pattern
        if tree_operator != pattern_operator:
            return False, mapping

        # Try to recursively match the operands
        for tree_operand, pattern_operand in zip(tree_operands, pattern_operands):
            # If the pattern operand is a leaf node
            if isinstance(pattern_operand, str):
                # Check if the pattern operand is a placeholder to be filled with the tree operand
                existing_value = mapping.get(pattern_operand)
                if existing_value is None:
                    # Placeholder is not yet filled, can be filled with the tree operand
                    mapping[pattern_operand] = tree_operand
                    return True, mapping
                elif existing_value != tree_operand:
                    # Placeholder is occupied by another tree, the tree does not match the pattern
                    return False, mapping
            else:
                # Recursively match the tree operand with the pattern operand
                does_match, mapping = self.match_pattern(tree_operand, pattern_operand, mapping)

                # If the tree operand does not match the pattern operand, the tree does not match the pattern
                if not does_match:
                    return False, mapping

        # The tree matches the pattern
        return True, mapping

    def apply_mapping(self, tree: list, mapping: dict[str, Any]) -> list:
        # If the tree is a leaf node, replace the placeholder with the actual subtree defined in the mapping
        if len(tree) == 1 and isinstance(tree[0], str):
            if tree[0].startswith('_'):
                return mapping[tree[0]]  # TODO: I put a bracket here. Find out why this is necessary
            return tree

        operator, operands = tree
        return [operator, [self.apply_mapping(operand, mapping) for operand in operands]]

    def _apply_auto_flash_simplifcation_rules(self, expression: list[str] | tuple[str, ...], rules_trees: dict[tuple, list[tuple[list[str], list[str]]]]) -> list[str]:
        if all(t == '<num>' or t in self.operator_arity for t in expression):
            return ['<num>']

        stack: list = []
        i = len(expression) - 1

        # Traverse the expression from right to left
        while i >= 0:
            token = expression[i]

            # Remember if a rule was applied in this iteration
            applied_rule = False

            # If the token is an operator, check for rules that can be applied
            if token in self.operator_arity_compat or token in self.operator_aliases:
                operator = self.operator_aliases.get(token, token)
                arity = self.operator_arity_compat[operator]
                operands = list(reversed(stack[-arity:]))
                operands_heads = [operand[0] for operand in operands]
                rules_key = (operator, *operands_heads)

                if all(operand[0] == '<num>' for operand in operands):
                    # All operands are constants
                    _ = [stack.pop() for _ in range(arity)]
                    stack.append(['<num>'])
                    i -= 1
                    continue

                # TODO: Optimize by hashing operands. e.g. rules_trees[(operator, operand1_type, operand2_type, ...)]

                subtree = [operator, operands]
                # Check if a pattern matches the current subtree
                for rule in rules_trees.get(rules_key, rules_trees.get((operator,), [])):
                    does_match, mapping = self.match_pattern(subtree, rule[0], mapping=None)
                    if does_match:
                        # print(f'Applying rule {rule}')
                        # Replace the placeholders (keys of the mapping) with the actual subtrees (values of the mapping) in the entire subtree at any depth
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(self.apply_mapping(deepcopy(rule[1]), mapping))
                        i -= 1
                        applied_rule = True
                        break

                if not applied_rule:
                    _ = [stack.pop() for _ in range(arity)]
                    stack.append([operator, operands])
                    i -= 1
                    continue

            if not applied_rule:
                stack.append([token])
                i -= 1

        # Unroll the tree into a flat expression in the correct order
        return flatten_nested_list(stack)[::-1]

    def find_connected_num_paths(self, expression: list[str] | tuple[str, ...]) -> tuple[list, list]:
        stack: list = []
        stack_annotations: list = []

        i = len(expression) - 1

        # Traverse the expression from right to left
        while i >= 0:
            token = expression[i]

            if token in self.connectable_operators:
                operator = token
                arity = 2
                operands = list(reversed(stack[-arity:]))
                operands_annotations_sets = list(reversed(stack_annotations[-arity:]))

                if all(operand[0] == '<num>' for operand in operands):
                    # All operands are constants. Simplify to a single constant
                    _ = [stack.pop() for _ in range(arity)]
                    _ = [stack_annotations.pop() for _ in range(arity)]
                    stack.append(['<num>'])
                    stack_annotations.append([set('<num>')])  # A node can have multiple annotations
                    i -= 1
                    continue

                operator_annotation_set = set()
                num_in_operands = '<num>' in [operand[0] for operand in operands]
                connection_class = self.operator_to_class[operator]
                if num_in_operands:
                    if any(any(operand_an.startswith(connection_class) for operand_an in operand_ans[0]) for operand_ans in operands_annotations_sets):
                        # Both operands are connected by a path of + or -
                        operator_annotation_set.add(f'{connection_class}_connected')
                    else:
                        operator_annotation_set.add(connection_class)
                elif all(any(operand_an.startswith(connection_class) for operand_an in operand_ans[0]) for operand_ans in operands_annotations_sets):
                    operator_annotation_set.add(f'{connection_class}_connected')

                _ = [stack.pop() for _ in range(arity)]
                _ = [stack_annotations.pop() for _ in range(arity)]
                stack.append([operator, operands])
                stack_annotations.append([operator_annotation_set, operands_annotations_sets])
                i -= 1
                continue

            if token == '<num>':
                # If the token is a number, push it onto the stack
                stack.append([token])
                stack_annotations.append([set(['<num>'])])
                i -= 1
                continue

            stack.append([token])
            stack_annotations.append([set()])
            i -= 1

        return stack, stack_annotations

    def cancel_nums(self, expression_tree: list, expression_annotations_tree: list) -> list[str]:
        stack = expression_tree
        stack_annotations = expression_annotations_tree

        expression: list[str] = []

        while len(stack) > 0:
            # Traverse in dfs
            subtree = stack.pop()
            subtree_annotation = stack_annotations.pop()

            # Leaf node
            if len(subtree) == 1:
                operand = subtree[0]
                operand_annotation = subtree_annotation[0]

                if operand == '<num>':
                    pruned = False
                    for connection_class in self.connection_classes:
                        if f'{connection_class}_prune' in operand_annotation:
                            expression.append(self.connection_classes[connection_class][1])  # Neural element
                            pruned = True
                            break
                    if pruned:
                        continue

                expression.append(operand)
                continue

            # Non-leaf node
            operator, operands = subtree
            operator_annotation_set, operands_annotations_sets = subtree_annotation

            for connection_class in self.connection_classes:
                if f'{connection_class}_prune' in operator_annotation_set:
                    # Promote children to '_prune'
                    for operand_an in operands_annotations_sets:
                        operand_an[0].add(f'{connection_class}_prune')
                    continue

                if f'{connection_class}_connected' in operator_annotation_set:
                    # Promote children to 'connected'
                    for operand_an in operands_annotations_sets:
                        if connection_class in operand_an[0] or '<num>' in operand_an[0]:
                            operand_an[0].add(f'{connection_class}_connected')

                    # If both children are connected, promote the left child to '_prune'
                    if all(any(operand_an.startswith(f'{connection_class}_connected') for operand_an in operand_ans[0]) for operand_ans in operands_annotations_sets):
                        # operands_annotations_sets[0][0].remove(f'{connection_class}_connected')  # Not neecessary because prune is checked first
                        operands_annotations_sets[0][0].add(f'{connection_class}_prune')
                    continue

            # Add the operator to the expression
            expression.append(operator)

            # Add the children to the stack
            for operand, operand_an in zip(reversed(operands), reversed(operands_annotations_sets)):
                stack.append(operand)
                stack_annotations.append(operand_an)

        return expression

    def collect_multiplicities(self, expression: list[str] | tuple[str, ...]) -> tuple[list, list, list]:
        stack: list = []
        stack_annotations: list = []
        stack_labels: list = []

        i = len(expression) - 1

        # Traverse the expression from right to left
        while i >= 0:
            token = expression[i]

            if token in self.connectable_operators:
                operator = token
                arity = 2
                operands = list(reversed(stack[-arity:]))
                operands_annotations_dicts = list(reversed(stack_annotations[-arity:]))
                operands_labels = list(reversed(stack_labels[-arity:]))

                operator_annotation_dict: dict[str, dict[tuple[str, ...], list[int]]] = {cc: {} for cc in self.connection_classes}

                cc = self.operator_to_class[operator]

                # Carry over annotations from operand nodes
                for operand_annotations_dict in operands_annotations_dicts:
                    for subtree_hash in operand_annotations_dict[0][cc]:
                        if subtree_hash not in operator_annotation_dict[cc]:
                            operator_annotation_dict[cc][subtree_hash] = [0, 0]

                        for p in range(2):
                            operator_annotation_dict[cc][subtree_hash][p] += operand_annotations_dict[0][cc][subtree_hash][p]

                        if operator in {'-', '/'}:
                            operator_annotation_dict[cc][subtree_hash][0], operator_annotation_dict[cc][subtree_hash][1] = operator_annotation_dict[cc][subtree_hash][1], operator_annotation_dict[cc][subtree_hash][0]

                # Add subtree hashes for both operand subtrees
                operand_tuple_0 = tuple(flatten_nested_list(operands[0])[::-1])
                operand_tuple_1 = tuple(flatten_nested_list(operands[1])[::-1])

                if operand_tuple_0 not in operator_annotation_dict[cc]:
                    operator_annotation_dict[cc][operand_tuple_0] = [1, 0]
                else:
                    operator_annotation_dict[cc][operand_tuple_0][0] += 1

                index = int(operator in {'+', '*'})
                if operand_tuple_1 not in operator_annotation_dict[cc]:
                    operator_annotation_dict[cc][operand_tuple_1] = [index, index - 1]
                else:
                    operator_annotation_dict[cc][operand_tuple_1][index - 1] += 1

                # Label each subtree with its own hash to know which to prune later
                _ = [stack.pop() for _ in range(arity)]
                _ = [stack_annotations.pop() for _ in range(arity)]
                _ = [stack_labels.pop() for _ in range(arity)]
                stack.append([operator, operands])
                stack_annotations.append([operator_annotation_dict, operands_annotations_dicts])
                new_label = tuple(flatten_nested_list([operator, operands])[::-1])
                stack_labels.append([new_label, operands_labels])
                i -= 1
                continue

            stack.append([token])
            stack_annotations.append([{cc: {tuple([token]): [0, 0]} for cc in self.connection_classes}])
            stack_labels.append([tuple([token])])
            i -= 1

        return stack, stack_annotations, stack_labels

    def cancel_terms(self, expression_tree: list, expression_annotations_tree: list, stack_labels: list) -> list[str]:
        stack = expression_tree
        stack_annotations = expression_annotations_tree
        stack_parity = [{cc: 1 for cc in self.connection_classes} for _ in range(len(stack_labels))]

        expression: list[str] = []

        argmax_candidate = None
        max_subtree_length = 0
        n_replaced = 0

        while len(stack) > 0:
            subtree = stack.pop()
            subtree_annotation = stack_annotations.pop()
            subtree_labels = stack_labels.pop()
            subtree_parities = stack_parity.pop()

            if argmax_candidate is not None:
                argmax_class, argmax_subtree, argmax_multiplicity_sum = argmax_candidate

                if argmax_subtree == subtree_labels[0]:
                    current_parity = subtree_parities[argmax_class]
                    inverse_operator = self.connection_classes_inverse[argmax_class]

                    if current_parity * argmax_multiplicity_sum < 0:
                        inverse_operator_prefix = (inverse_operator,)
                        double_inverse_operator_prefix = ()
                    else:
                        inverse_operator_prefix = ()
                        double_inverse_operator_prefix = (inverse_operator,)

                    neutral_element = self.connection_classes[argmax_class][1]
                    if argmax_multiplicity_sum == 0:
                        # Term is cancelled entirely. Replace all occurences with the neutral element
                        first_replacement = (neutral_element,)
                        other_replacements = neutral_element

                    if argmax_multiplicity_sum == 1:
                        # Term occurs once. Replace every occurence after the first one with the neutral element
                        first_replacement = inverse_operator_prefix + argmax_subtree
                        other_replacements = (neutral_element,)

                    if argmax_multiplicity_sum == -1:
                        # Term occurs once but inverted. Replace the first occurence with the inverse of the term. Replace every occurence after the first one with the neutral element
                        first_replacement = double_inverse_operator_prefix + argmax_subtree
                        other_replacements = (neutral_element,)

                    if argmax_multiplicity_sum > 1:
                        # Term occurs multiple times. Replace the first occurence with a multiplication or power of the term. Replace every occurence after the first one with the neutral element
                        if argmax_class == 'mult':
                            if argmax_multiplicity_sum > 5 and is_prime(argmax_multiplicity_sum):
                                powers = self.factorize_to_at_most(argmax_multiplicity_sum - 1, self.max_power)
                                first_replacement = inverse_operator_prefix + ('*',) + tuple(f'pow_{p}' for p in powers) + argmax_subtree + argmax_subtree
                            else:
                                powers = self.factorize_to_at_most(argmax_multiplicity_sum, self.max_power)
                                first_replacement = inverse_operator_prefix + tuple(f'pow_{p}' for p in powers) + argmax_subtree
                        else:
                            first_replacement = ('*', '<num>',) + argmax_subtree

                        other_replacements = (neutral_element,)

                    if argmax_multiplicity_sum < -1:
                        # Term occurs multiple times. Replace the first occurence with a multiplication or power of the term. Replace every occurence after the first one with the neutral element
                        if argmax_class == 'mult':
                            if argmax_multiplicity_sum > 5 and is_prime(argmax_multiplicity_sum):
                                powers = self.factorize_to_at_most(argmax_multiplicity_sum - 1, self.max_power)
                                first_replacement = double_inverse_operator_prefix + ('*',) + tuple(f'pow_{p}' for p in powers) + argmax_subtree + argmax_subtree
                            else:
                                powers = self.factorize_to_at_most(argmax_multiplicity_sum, self.max_power)
                                first_replacement = double_inverse_operator_prefix + tuple(f'pow_{p}' for p in powers) + argmax_subtree
                        else:
                            first_replacement = ('*', '<num>',) + argmax_subtree

                        other_replacements = (neutral_element,)

                    if n_replaced == 0:
                        expression.extend(first_replacement)
                    else:
                        expression.extend(other_replacements)
                    n_replaced += 1
                    continue

            # Leaf node
            if len(subtree) == 1:
                expression.append(subtree[0])
                continue

            # Non-leaf node
            operator, operands = subtree
            _, operands_annotations_sets = subtree_annotation
            _, operands_labels = subtree_labels
            operator_parity = subtree_parities  # No operand parity information yet

            if len(operands) == 2:
                propagated_operand_parities: list[dict[str, int]] = [{}, {}]
                for cc, (operator_set, _) in self.connection_classes.items():
                    if operator in operator_set:
                        propagated_operand_parities[0][cc] = operator_parity[cc]
                        propagated_operand_parities[1][cc] = operator_parity[cc] * (-1 if operator in {'-', '/'} else 1)
                    else:
                        propagated_operand_parities[0][cc] = operator_parity[cc]
                        propagated_operand_parities[1][cc] = operator_parity[cc]

                # If no cancellation candidate has been identified yet, try to find one in the current subtree
                if argmax_candidate is None:
                    for cc in self.connection_classes:
                        for subtree_hash, multiplicity in subtree_annotation[0][cc].items():
                            if len(subtree_hash) > max_subtree_length and sum(abs(m) for m in multiplicity) > 1 and not all(term == '<num>' or term in self.operator_arity for term in subtree_hash):
                                argmax_candidate = (cc, subtree_hash, multiplicity[0] - multiplicity[1])

            # Add the operator to the expression
            expression.append(operator)

            # Add the children to the stack
            for operand, operand_an, operand_label, propagated_operand_parity in zip(reversed(operands), reversed(operands_annotations_sets), reversed(operands_labels), reversed(propagated_operand_parities)):
                stack.append(operand)
                stack_annotations.append(operand_an)
                stack_labels.append(operand_label)
                stack_parity.append(propagated_operand_parity)

        return expression

    def sort_operands(self, expression: list[str] | tuple[str, ...]) -> list[str]:
        stack: list = []
        i = len(expression) - 1

        while i >= 0:
            token = expression[i]

            if token in self.operator_arity_compat or token in self.operator_aliases:
                operator = self.operator_aliases.get(token, token)
                arity = self.operator_arity_compat[operator]
                operands = list(reversed(stack[-arity:]))

                if operator in self.commutative_operators:
                    # Check for the pattern [*, *, A, B, C] -> [*, A, *, B, C] or [+, +, A, B, C] -> [+, A, +, B, C]
                    if len(operands[0]) == 2 and operator == operands[0][0]:
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append([operator, [operands[0][1][0], [operator, [operands[0][1][1], operands[1]]]]])
                        i -= 1
                        continue

                    subtree = [operator, operands]

                    # Traverse through the tree in breadth-first order
                    queue = [subtree]
                    commutative_paths: list[tuple] = [tuple()]
                    commutative_positions = []
                    while queue:
                        node = queue.pop(0)
                        current_path = commutative_paths.pop(0)
                        for child_index, child in enumerate(node[1]):  # I conclude that using `i` as a variable name here is not very clever
                            if len(child) > 1:
                                if child[0] == node[0]:
                                    # Continue: Same commutative perator
                                    queue.append(child)
                                    commutative_paths.append(current_path + (child_index,))
                                else:
                                    # Stop: Different operator
                                    commutative_positions.append(current_path + (child_index,))
                            else:
                                # Stop: Leaf
                                commutative_positions.append(current_path + (child_index,))

                    # Sort the positions
                    sorted_indices = sorted(range(len(commutative_positions)), key=lambda x: commutative_positions[x])

                    commutative_paths = [commutative_positions[i] for i in sorted_indices]
                    commutative_positions = [commutative_positions[i] for i in sorted_indices]

                    operands_to_sort = []
                    for position in commutative_positions:
                        node = subtree
                        for position_index in position:
                            node = node[1][position_index]
                        operands_to_sort.append(node)

                    sorted_operands = sorted(operands_to_sort, key=self.operand_key)

                    # Replace the operands in the tree
                    new_subtree: list = deepcopy(subtree)

                    for position, operand in zip(commutative_positions, sorted_operands):
                        node = new_subtree
                        for position_index in position:
                            node = node[1][position_index]
                        node[:] = operand

                    operands = new_subtree[1]

                    _ = [stack.pop() for _ in range(arity)]
                    stack.append([operator, operands])
                    i -= 1
                    continue

                _ = [stack.pop() for _ in range(arity)]
                stack.append([operator, operands])

            else:
                stack.append([token])

            i -= 1

        return flatten_nested_list(stack)[::-1]

    def simplify_auto_flash(self, expression: list[str] | tuple[str, ...], max_iter: int = 5, mask_elementary_literals: bool = True, inplace: bool = False) -> list[str] | tuple[str, ...]:
        if isinstance(expression, tuple):
            was_tuple = True
            expression = list(expression)
            new_expression = expression.copy()
        else:
            was_tuple = False
            new_expression = expression

        for _ in range(max_iter):
            # Sort operands
            new_expression = self.sort_operands(new_expression)

            # Cancel constants
            expression_tree, annotated_expression_tree = self.find_connected_num_paths(new_expression)
            new_expression = self.cancel_nums(expression_tree, annotated_expression_tree)
            # print(f'Cancelled constants: {new_expression}')

            # Apply simplification rules
            new_expression = self._apply_auto_flash_simplifcation_rules(new_expression, self.simplification_rules_trees)
            # print(f'Applied rules 1: {new_expression}')

            # Cancel any terms
            expression_tree, annotated_expression_tree, stack_labels = self.collect_multiplicities(new_expression)
            new_expression = self.cancel_terms(expression_tree, annotated_expression_tree, stack_labels)
            # print(f'Cancelled terms: {new_expression}')

            # Apply simplification rules
            new_expression = self._apply_auto_flash_simplifcation_rules(new_expression, self.simplification_rules_trees)
            # print(f'Applied rules 2: {new_expression}')

            if new_expression == expression:
                break
            expression = new_expression

        if mask_elementary_literals:
            new_expression = self.mask_elementary_literals(new_expression, inplace=inplace)

        if was_tuple:
            return tuple(new_expression)
        return new_expression

    def construct_expressions(self, hashes_of_size: dict[int, set[tuple[str, ...]]], non_leaf_nodes: dict[str, int], must_have_sizes: list | set | None = None) -> Generator[tuple[str, ...], None, None]:
        hashes_of_size_with_lists = {k: list(v) for k, v in hashes_of_size.items()}

        filter_sizes = must_have_sizes is not None and not len(must_have_sizes) == 0
        if must_have_sizes is not None and filter_sizes:
            must_have_sizes_set = set(must_have_sizes)

        # Append existing trees to every operator
        for new_root_operator, arity in non_leaf_nodes.items():
            # Start with the smallest arity-tuples of trees
            for child_lengths in sorted(itertools.product(list(hashes_of_size_with_lists.keys()), repeat=arity), key=lambda x: sum(x)):
                # Check all possible combinations of child trees
                if filter_sizes and not any(length in must_have_sizes_set for length in child_lengths):
                    continue
                for child_combination in itertools.product(*[hashes_of_size_with_lists[child_length] for child_length in child_lengths]):
                    yield (new_root_operator,) + tuple(itertools.chain.from_iterable(child_combination))

    def exist_constants_that_fit(self, expression: list[str] | tuple[str, ...], variables: list[str], X: np.ndarray, y_target: np.ndarray) -> bool:
        if isinstance(expression, tuple):
            expression = list(expression)

        executable_prefix_expression = self.operators_to_realizations(expression)
        prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression, convert_numbers_to_constant=False)
        code_string = self.prefix_to_infix(prefix_expression_with_constants, realization=True)
        code = codify(code_string, variables + constants)
        f = self.code_to_lambda(code)

        def pred_function(X: np.ndarray, *constants: np.ndarray | None) -> float | np.ndarray:
            if len(constants) == 0:
                y = safe_f(f, X)
            y = safe_f(f, X, constants)

            # If the numbers are complex, return nan
            if np.iscomplexobj(y):
                return np.full(X.shape[0], np.nan)

            return y

        p0 = np.random.normal(loc=0, scale=5, size=len(constants))

        is_valid = np.isfinite(X).all(axis=1) & np.isfinite(y_target)

        if not np.any(is_valid):
            return False

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                popt, _ = curve_fit(pred_function, X[is_valid], y_target[is_valid].flatten(), p0=p0)
        except RuntimeError:
            return False

        y = f(*X.T, *popt)
        if not isinstance(y, np.ndarray):
            y = np.full(X.shape[0], y)  # type: ignore

        return np.allclose(y_target, y, equal_nan=True)

    def find_rules(
            self,
            max_n_rules: int | None = None,
            max_pattern_length: int | None = None,
            timeout: float | None = None,
            dummy_variables: int | list[str] | None = None,
            additional_leaf_nodes: list[str] | None = None,
            max_simplify_steps: int = 1,
            X: np.ndarray | int | None = None,
            C: np.ndarray | int | None = None,
            constants_fit_retries: int = 5,
            output_file: str | None = None,
            save_every: int = 100,
            reset_rules: bool = True,
            verbose: bool = False) -> None:

        hashes_of_size: defaultdict[int, set[tuple[str, ...]]] = defaultdict(set)

        if dummy_variables is None:
            dummy_variables = [f"x{i}" for i in range(3)]  # Room for up to 3 different terms in simplification patterns
        elif isinstance(dummy_variables, int):
            dummy_variables = [f"x{i}" for i in range(dummy_variables)]

        if reset_rules:
            self.simplification_rules = []
            self.simplification_rules_trees = self.rules_trees_from_rules_list(self.simplification_rules, dummy_variables)

        if X is None:
            X = np.random.normal(loc=0, scale=5, size=(1024, len(dummy_variables)))
        elif isinstance(X, int):
            X = np.random.normal(loc=0, scale=5, size=(X, len(dummy_variables)))

        if C is None:
            C = np.random.normal(loc=0, scale=5, size=128)
        elif isinstance(C, int):
            C = np.random.normal(loc=0, scale=5, size=C)

        if additional_leaf_nodes is None:
            # TODO: Think about more special constants like pi, e, 2, 3, 1/2, etc.
            additional_leaf_nodes = ['<num>', '0', '1', '(-1)', 'float("inf")', 'float("-inf")', 'float("nan")']

        leaf_nodes = dummy_variables + additional_leaf_nodes
        non_leaf_nodes = dict(sorted(self.operator_arity.items(), key=lambda x: x[1]))

        pbar = tqdm(disable=not verbose)
        n_scanned = 0

        simplify_rules_every = 1000

        start_time = time.time()

        max_rules_string = f'/{max_n_rules:,}' if max_n_rules is not None else ''
        max_time_string = f'/{timeout/60:.1f}' if timeout is not None else ''

        # debug_file = os.path.dirname(output_file) + '/debug.txt'  # type: ignore
        # with open(debug_file, 'w') as debug_file_handle:
        #     debug_file_handle.write('')
        # debug_file_handle = open(debug_file, 'a')

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # Create all leaf nodes
                for leaf in leaf_nodes[:max_n_rules]:
                    simplified_skeleton = self.simplify_auto_flash([leaf], max_simplify_steps, mask_elementary_literals=False)

                    executable_prefix_expression = self.operators_to_realizations(simplified_skeleton)
                    prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression, convert_numbers_to_constant=False)
                    code_string = self.prefix_to_infix(prefix_expression_with_constants, realization=True)
                    code = codify(code_string, dummy_variables + constants)

                    hashes_of_size[len(simplified_skeleton)].add(tuple(simplified_skeleton))  # type: ignore

                new_sizes: set[int] = set()

                while (max_n_rules is None or len(self.simplification_rules) < max_n_rules) and (timeout is None or time.time() - start_time <= timeout):
                    simplified_hashes_of_size: defaultdict[int, set[tuple[str, ...]]] = defaultdict(set)
                    for length, hashes_list in hashes_of_size.items():
                        for h in hashes_list:
                            simplified_skeleton = self.simplify_auto_flash(h, max_simplify_steps, mask_elementary_literals=False)
                            simplified_hashes_of_size[len(simplified_skeleton)].add(tuple(simplified_skeleton))  # type: ignore
                    hashes_of_size = simplified_hashes_of_size

                    if max_pattern_length is not None and max(hashes_of_size.keys()) > max_pattern_length:
                        # If the maximum pattern length is exceeded, stop searching
                        if verbose:
                            print(f'Maximum pattern length reached: {max_pattern_length}')
                        break

                    # For logging
                    hashes_of_size_lengths = {k: len(v) for k, v in hashes_of_size.items()}

                    if len(hashes_of_size[1]) == 1:
                        # Should not happen since the leaf nodes cannot disappear
                        print(hashes_of_size[1])
                        print(hashes_of_size_lengths)
                        print(self.simplification_rules)
                        exit()

                    new_hashes_of_size: defaultdict[int, set[tuple[str, ...]]] = defaultdict(set)
                    # debug_file_handle.write(f'{"-" * 100}\n')

                    # debug_file_handle.write(f'Hashes of size: {hashes_of_size_lengths}\n')
                    # debug_file_handle.write(f'Must include sizes: {new_sizes}\n')
                    for combination in self.construct_expressions(hashes_of_size, non_leaf_nodes, must_have_sizes=new_sizes):
                        # debug_file_handle.write(f'{combination}\n')
                        if timeout is not None and time.time() - start_time > timeout:
                            if verbose:
                                print('Reached timeout')
                            break

                        if n_scanned % simplify_rules_every == 0:
                            for i, rule in enumerate(self.simplification_rules):
                                self.simplification_rules[i] = (rule[0], tuple(self.simplify_auto_flash(rule[1], max_simplify_steps, mask_elementary_literals=False)))

                            self.simplification_rules = deduplicate_rules(self.simplification_rules, dummy_variables)
                            self.simplification_rules_trees = self.rules_trees_from_rules_list(self.simplification_rules, dummy_variables)

                        # Write the rules to a file to check the progress
                        if output_file is not None and n_scanned % save_every == 0:
                            with open(output_file, 'w') as file:
                                json.dump(self.simplification_rules, file, indent=4)

                        if max_n_rules is not None and len(self.simplification_rules) >= max_n_rules:
                            print(f'Reached maximum number of rules: {len(self.simplification_rules)}')
                            break

                        simplified_skeleton = self.simplify_auto_flash(list(combination), max_simplify_steps, mask_elementary_literals=False)
                        simplified_skeleton_hash = tuple(simplified_skeleton)  # type: ignore

                        new_hashes_of_size_lengths = {k: len(v) for k, v in new_hashes_of_size.items()}
                        pbar.set_postfix_str(f"Rules: {len(self.simplification_rules):,}{max_rules_string}, Time: {(time.time() - start_time)/60:.1f}{max_time_string} min, Subtrees: {hashes_of_size_lengths} <- {new_hashes_of_size_lengths}, Current: {combination} -> {simplified_skeleton_hash}")

                        # Check if all leaf nodes are <num> (i.e. if the skeleton is purely numerical)
                        if all([t == '<num>' or t in self.operator_arity for t in simplified_skeleton]) and len(simplified_skeleton) > 1:
                            # Simplify the skeleton to a single <num>
                            new_rule_candidates: list[tuple[tuple[str, ...], tuple[str, ...]]] = [(simplified_skeleton_hash, ('<num>',))]
                        else:
                            executable_prefix_expression = self.operators_to_realizations(simplified_skeleton)
                            prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression, convert_numbers_to_constant=False)
                            code_string = self.prefix_to_infix(prefix_expression_with_constants, realization=True)
                            code = codify(code_string, dummy_variables + constants)

                            f = self.code_to_lambda(code)

                            y = safe_f(f, X, C[:len(constants)])  # type: ignore

                            all_sizes = set(hashes_of_size.keys()) | set(new_hashes_of_size.keys())
                            new_rule_candidates = []
                            for length in all_sizes:
                                for candidate_hashes_of_size in (hashes_of_size, new_hashes_of_size):
                                    if length not in candidate_hashes_of_size:
                                        continue

                                    # Ignore simplification candidates that do not shorten the expression
                                    if length >= len(simplified_skeleton_hash):
                                        continue

                                    # TODO: If we traverse the candidates from short to long, we can stop early once we found a match
                                    for candidate_hash in candidate_hashes_of_size[length]:
                                        if candidate_hash == simplified_skeleton_hash:
                                            continue
                                        executable_prefix_candidate_hash = self.operators_to_realizations(candidate_hash)
                                        prefix_candidate_hash_with_constants, constants_candidate_hash = num_to_constants(executable_prefix_candidate_hash, convert_numbers_to_constant=False)
                                        code_string_candidate_hash = self.prefix_to_infix(prefix_candidate_hash_with_constants, realization=True)
                                        code_candidate_hash = codify(code_string_candidate_hash, dummy_variables + constants_candidate_hash)

                                        f_candidate = self.code_to_lambda(code_candidate_hash)

                                        # Record the image
                                        if len(constants_candidate_hash) == 0:
                                            y_candidate = safe_f(f_candidate, X)
                                            if not isinstance(y_candidate, np.ndarray):
                                                y_candidate = np.full(X.shape[0], y_candidate)  # type: ignore

                                            if np.allclose(y, y_candidate, equal_nan=True):
                                                new_rule_candidates.append((simplified_skeleton_hash, candidate_hash))
                                        else:
                                            if any([self.exist_constants_that_fit(candidate_hash, dummy_variables, X, y) for _ in range(constants_fit_retries)]):
                                                new_rule_candidates.append((simplified_skeleton_hash, candidate_hash))

                                if len(new_rule_candidates) > 0:
                                    # Found a match that is shorter than other futher candidates could ever be
                                    break

                        # Find the shortest rule
                        if len(new_rule_candidates) > 0:
                            # If there are rules with and without <num>, prefer the ones without
                            new_rule_candidates_without_num = [c for c in new_rule_candidates if '<num>' not in c[1]]
                            if len(new_rule_candidates_without_num) > 0:
                                new_rule_candidates = new_rule_candidates_without_num
                            self.simplification_rules.append(new_rule_candidates[0])
                            new_hashes_of_size[len(simplified_skeleton_hash)].add(new_rule_candidates[0][1])
                        else:
                            new_hashes_of_size[len(simplified_skeleton_hash)].add(simplified_skeleton_hash)

                        n_scanned += 1
                        pbar.update(1)

                    new_sizes = set()

                    hashes_of_size_lengths_before = {k: len(v) for k, v in hashes_of_size.items()}
                    for new_length, new_hashes in new_hashes_of_size.items():
                        hashes_of_size[new_length].update(new_hashes)
                    hashes_of_size_lengths_after = {k: len(v) for k, v in hashes_of_size.items()}

                    for size in hashes_of_size_lengths_after.keys():
                        if size not in hashes_of_size_lengths_before or hashes_of_size_lengths_after[size] > hashes_of_size_lengths_before[size]:
                            new_sizes.add(size)

                else:
                    if verbose:
                        print('Reached maximum number of rules or timeout')

                # Simplify the rules one last time
                for i, rule in enumerate(self.simplification_rules):
                    self.simplification_rules[i] = (rule[0], tuple(self.simplify_auto_flash(rule[1], max_simplify_steps, mask_elementary_literals=False)))

                self.simplification_rules_trees = self.rules_trees_from_rules_list(self.simplification_rules, dummy_variables)

            pbar.close()

            if verbose:
                print('Finished.')

            if output_file is not None:
                if verbose:
                    print('Saving the rules...')
                with open(output_file, 'w') as file:
                    json.dump(self.simplification_rules, file, indent=4)

            # debug_file_handle.close()

        except KeyboardInterrupt:
            # debug_file_handle.close()
            pbar.close()
            if output_file is not None:
                if verbose:
                    print('Interrupted. Trying to save the rules...')
                time.sleep(1)  # Allow the user to interrupt the process
                with open(output_file, 'w') as file:
                    json.dump(self.simplification_rules, file, indent=4)
                if verbose:
                    print('Rules saved.')
            raise

    # SIMPLIFICATION
    def simplify_flash(self, prefix_expression: list[str], mask_elementary_literals: bool = True, max_iter: int = 5, inplace: bool = False, verbose: bool = False, debug: bool = False) -> list[str]:
        '''
        Simplify an expression

        Parameters
        ----------
        prefix_expression : list[str] | tuple[str]
            The prefix expression
        mask_elementary_literals : bool, optional
            Whether to mask elementary literals such as <0> and <1> with <num>, by default True
        max_iter : int, optional
            The maximum number of iterations, by default 5
        inplace : bool, optional
            Whether to modify the expression in place, by default False
        verbose : bool, optional
            Whether to print verbose output, by default False
        debug : bool, optional
            Whether to print debug output, by default False

        Returns
        -------
        list[str]
            The simplified expression
        '''
        if not isinstance(prefix_expression, list):
            prefix_expression = list(prefix_expression)

        if inplace:
            modified_prefix_expression = prefix_expression
        else:
            modified_prefix_expression = prefix_expression.copy()

        new_modified_prefix_expression = prefix_expression.copy()

        for _ in range(max_iter):
            for __ in range(max_iter):
                new_modified_prefix_expression = _simplify_flash(self, new_modified_prefix_expression, verbose=verbose, debug=debug)
                if new_modified_prefix_expression == modified_prefix_expression:
                    break
                modified_prefix_expression = new_modified_prefix_expression

            if mask_elementary_literals:
                new_modified_prefix_expression = self.mask_elementary_literals(new_modified_prefix_expression, inplace=inplace)

            if new_modified_prefix_expression == modified_prefix_expression:
                break

        return new_modified_prefix_expression

    def mask_elementary_literals(self, prefix_expression: list[str], inplace: bool = False) -> list[str]:
        '''
        Mask elementary literals such as <0> and <1> with <num>

        Parameters
        ----------
        prefix_expression : list[str]
            The prefix expression
        inplace : bool, optional
            Whether to modify the expression in place, by default False

        Returns
        -------
        list[str]
            The expression with elementary literals masked
        '''
        if inplace:
            modified_prefix_expression = prefix_expression
        else:
            modified_prefix_expression = prefix_expression.copy()

        for i, token in enumerate(prefix_expression):
            if token not in self.variables and token not in self.operator_arity_compat and token not in self.operator_aliases and token not in self.special_constants:
                modified_prefix_expression[i] = '<num>'

        return modified_prefix_expression

    def operand_key(self, operands: list) -> tuple:
        '''
        Returns a key for sorting the operands of a commutative operator.

        Parameters
        ----------
        operands : list
            The operands to sort.

        Returns
        -------
        tuple
            The key for sorting the operands.
        '''
        if len(operands) > 1 and isinstance(operands[0], str):
            # if operands[0] in self.operator_arity_compat or operands[0] in self.operator_aliases:
            # Node
            operand_keys = tuple(self.operand_key(op) for op in operands[1])
            return (2, len(flatten_nested_list(operands)), operand_keys, operands[0])

        # Leaf
        if len(operands) == 1 and isinstance(operands[0], str):
            try:
                return (1, float(operands[0]))
            except ValueError:
                return (0, operands[0])

        if isinstance(operands, str):
            return (0, operands)

        raise ValueError(f'None of the criteria matched for operands {operands}:\n1. ({len(operands) > 1}, {isinstance(operands[0], str)}, {operands[0] in self.operator_arity_compat or operands[0] in self.operator_aliases})\n2. ({len(operands) == 1}, {isinstance(operands[0], str)})\n3. ({isinstance(operands, str)})')

    # CODIFYING
    def operators_to_realizations(self, prefix_expression: list[str] | tuple[str, ...]) -> list[str] | tuple[str, ...]:
        '''
        Converts a prefix expression from operators to realizations.

        Parameters
        ----------
        prefix_expression : list[str]
            The prefix expression to convert.

        Returns
        -------
        list[str]
            The converted prefix expression.
        '''
        return [self.operator_realizations.get(token, token) for token in prefix_expression]

    def realizations_to_operators(self, prefix_expression: list[str]) -> list[str]:
        '''
        Converts a prefix expression from realizations to operators.

        Parameters
        ----------
        prefix_expression : list[str]
            The prefix expression to convert.

        Returns
        -------
        list[str]
            The converted prefix expression.
        '''
        return [self.realization_to_operator.get(token, token) for token in prefix_expression]

    @staticmethod
    def code_to_lambda(code: CodeType) -> Callable[..., float]:
        '''
        Converts a code object to a lambda function.

        Parameters
        ----------
        code : CodeType
            The code object to convert.

        Returns
        -------
        Callable[..., float]
            The lambda function.
        '''
        return FunctionType(code, globals())()

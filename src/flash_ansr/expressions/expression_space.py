import re
import importlib
import fractions
from typing import Any, Callable, Literal
from copy import deepcopy
from types import CodeType, FunctionType
from math import prod
from fractions import Fraction
import signal

import numpy as np
from sympy import simplify, parse_expr

from flash_ansr.models.transformer_utils import Tokenizer
from flash_ansr.utils import load_config
from flash_ansr.expressions.utils import get_used_modules, codify, numbers_to_num, flatten_nested_list, is_prime, num_to_constants


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
    def __init__(self, operators: dict[str, dict[str, Any]], variables: int, simplification: Literal['flash', 'sympy'] = 'flash', special_tokens: list[str] | None = None) -> None:
        self.simplification = simplification

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

        self.variables = [f'x{i + 1}' for i in range(variables)]

        self.tokenizer = Tokenizer(vocab=self.operator_tokens + self.variables, special_tokens=special_tokens)

        self.modules = get_used_modules(''.join(f"{op}(" for op in self.operator_realizations.values()))  # HACK: This can be done more elegantly for sure

        self.import_modules()

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
        config_ = load_config(config)

        if "expressions" in config_.keys():
            config_ = config_["expressions"]

        return cls(operators=config_["operators"], variables=config_["variables"], simplification=config_.get("simplification", 'flash'), special_tokens=config_.get("special_tokens", None))

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

        return self._deparenthesize(infix_expression)

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
            case _:
                raise ValueError(f'Invalid simplification method: {self.simplification}')

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
                new_modified_prefix_expression = self._simplify(new_modified_prefix_expression, verbose=verbose, debug=debug)
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
            if token in ['<1>', '<0>']:
                modified_prefix_expression[i] = '<num>'

        return modified_prefix_expression

    def _simplify(self, expression: list[str], verbose: bool = False, debug: bool = False) -> list[str]:
        '''
        Simplify an expression

        Parameters
        ----------
        expression : list[str]
            The prefix expression
        verbose : bool, optional
            Whether to print verbose output, by default False
        debug : bool, optional
            Whether to print debug output, by default False

        Returns
        -------
        list[str]
            The simplified expression
        '''
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

                if all(operand[0] == '<num>' for operand in operands):
                    if verbose:
                        print('Applying numeric subtree simplification')
                    # All operands are <num>, so we can simplify
                    _ = [stack.pop() for _ in range(arity)]
                    stack.append(['<num>'])  # Replace the operands with a single <num>
                    i -= 1
                    continue

                # if all(isinstance(stack[-j], str) and re.match(r'-?\d+\.\d+|\d+', stack[-j]) for j in range(1, arity + 1)):
                if all(len(operand) == 1 and isinstance(operand[0], str) and re.match(r'-?\d+\.\d+|-?\d+', operand[0]) for operand in operands):
                    if verbose:
                        print('Applying numeric subtree evaluation')
                    # All operands are numbers, so we can simplify by computing the result
                    # code_str = self.prefix_to_infix([operator] + [stack.pop() for _ in range(arity)], power='**', realization=True)
                    flat_prefix_expression = flatten_nested_list([operator, operands])[::-1]
                    code_str = self.prefix_to_infix(flat_prefix_expression, power='**', realization=True)
                    code = codify(code_str)
                    result = self.code_to_lambda(code)()
                    if int(result) == result:
                        result = int(result)
                    _ = [stack.pop() for _ in range(arity)]
                    stack.append([str(result)])
                    i -= 1
                    continue

                if operator == '*':
                    # Check for the pattern [*, A, 0] -> [0] and [*, 0, A] -> [0]
                    if ['<0>'] in operands:
                        if verbose:
                            print('Applying [*, A, 0] -> [0]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['<0>'])
                        # print(stack)
                        i -= 1
                        continue

                    # Check for the pattern [*, A, A] -> [pow2, A]
                    if operands[0] == operands[1]:
                        if verbose:
                            print('Applying [*, A, A] -> [pow2, A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['pow2', [operands[0]]])
                        i -= 1
                        continue

                    if len(operands[0]) == 2 and len(operands[1]) == 2:
                        # Check for the pattern [*, abs, A, abs, B] -> [abs, *, A, B]
                        if operands[0][0] == 'abs' and operands[1][0] == 'abs':
                            if verbose:
                                print(f'Applying [{operator}, abs, A, abs, B] -> [abs, {operator}, A, B]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append(['abs', [[operator, [operands[0][1][0], operands[1][1][0]]]]])
                            i -= 1
                            continue

                    # Check for the pattern [*, A, neg, A] -> [neg, pow2, A] or [*, neg, A, A] -> [neg, pow2, A]
                    if (len(operands[1]) == 2 and operands[1][0] == 'neg' and operands[0] == operands[1][1][0]) or len(operands[0]) == 2 and operands[0][0] == 'neg' and operands[1] == operands[0][1][0]:
                        if verbose:
                            print('Applying [*, A, neg, A] | [*, neg, A, A] -> [neg, pow2, A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['neg', [['pow2', [operands[0]] if len(operands[1]) == 2 else [operands[1]]]]])
                        i -= 1
                        continue

                    if len(operands[1]) == 2 and operands[1][0] == '+':
                        # Check for the pattern [*, A, +, B, B] and sort operands A and B. (The case [*, +, B, B, A] will be transformed into [*, A, +, B, B] during normal operand sorting)
                        if operands[1][1][0] == operands[1][1][1]:
                            if verbose:
                                print('Sorting A and B in expansion with commutative operators [*, A, +, B, B]', end='')
                            unique_operands_to_sort = [operands[0], operands[1][1][0]]
                            unique_sorted_operands = sorted(unique_operands_to_sort, key=self.operand_key)
                            if unique_operands_to_sort != unique_sorted_operands:
                                if verbose:
                                    print(f': {unique_operands_to_sort} -> {unique_sorted_operands}')
                                operands[0] = unique_sorted_operands[0]
                                operands[1][1][0] = unique_sorted_operands[1]
                                operands[1][1][1] = unique_sorted_operands[1]
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append([operator, [operands[0], operands[1]]])
                                i -= 1
                                continue
                            if verbose:
                                print()

                        # Check for the pattern [*, A, +, A, A] -> [+, pow2, A, pow2, A]
                        if operands[0] == operands[1][1][0] == operands[1][1][1]:
                            if verbose:
                                print('Applying [*, A, +, A, A] -> [+, pow2, A, pow2, A]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append(['+', [['pow2', [operands[0]]], ['pow2', [operands[0]]]]])
                            i -= 1
                            continue

                    if len(operands[0]) == len(operands[1]) == 2:
                        if operands[0][0] == operands[1][0] == '-':
                            A = operands[0][1][0]
                            B = operands[0][1][1]
                            C = operands[1][1][0]
                            D = operands[1][1][1]

                            # Check for the pattern [*, -, A, B, -, B, A] -> [neg, pow2, -, A, B]
                            if B == C and A == D:
                                if verbose:
                                    print('Applying [*, -, A, B, -, B, A] -> [neg, pow2, -, A, B]')
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append(['neg', [['pow2', [['-', [A, B]]]]]])
                                i -= 1
                                continue

                            # Sort the operands
                            potential_subtrees = [
                                [operator, [['-', [A, B]], ['-', [C, D]]]],
                                [operator, [['-', [B, A]], ['-', [D, C]]]],  # Swapped
                                [operator, [['-', [C, D]], ['-', [A, B]]]],  # Swapped
                                [operator, [['-', [D, C]], ['-', [B, A]]]],  # Swapped
                            ]
                            sorted_potential_subtrees = sorted(potential_subtrees, key=self.operand_key)

                            if potential_subtrees[0] != sorted_potential_subtrees[0]:
                                if verbose:
                                    print(f'Swapping positions from [{operator}, -, A, B, -, C, D]')
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append(sorted_potential_subtrees[0])
                                i -= 1
                                continue

                if operator == '/':
                    # Check for the pattern [/, A, 1] -> [A]
                    if operands[1] == ['<1>']:
                        if verbose:
                            print('Applying [/, A, 1] -> [A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(operands[0])
                        i -= 1
                        continue

                    # Check for the pattern [/, 1, A] -> [inv A]
                    if operands[0] == ['<1>']:
                        if verbose:
                            print('Applying [/, 1, A] -> [inv A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['inv', [operands[1]]])
                        i -= 1
                        continue

                    # Check for the pattern [/, abs, A, A] _> [/, A, abs, A]
                    if len(operands[0]) == 2 and operands[0][0] == 'abs' and operands[0][1][0] == operands[1]:
                        if verbose:
                            print('Applying [/, abs, A, A] -> [/, A, abs, A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['/', [operands[1], operands[0]]])
                        i -= 1
                        continue

                    if len(operands[0]) == 2 and len(operands[1]) == 2:
                        # Check for the pattern [/, +, A, A, +, B, B] -> [/, A, B]
                        if operands[0][0] == operands[1][0] == '+' and operands[0][1][0] == operands[0][1][1] and operands[1][1][0] == operands[1][1][1]:
                            if verbose:
                                print('Applying [/, +, A, A, +, B, B] -> [/, A, B]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append(['/', [operands[0][1][0], operands[1][1][0]]])
                            i -= 1
                            continue

                        # Check for the pattern [/, -, A, B, -, B, A] -> [neg, <1>]
                        if operands[0][0] == operands[1][0] == '-' and operands[0][1][1] == operands[1][1][0] and operands[0][1][0] == operands[1][1][1]:
                            if verbose:
                                print('Applying [/, -, A, B, -, B, A] -> [neg, <1>]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append(['neg', [['<1>']]])
                            i -= 1
                            continue

                    if len(operands[0]) == len(operands[1]) == 2:
                        if operands[0][0] == operands[1][0] == '-':
                            A = operands[0][1][0]
                            B = operands[0][1][1]
                            C = operands[1][1][0]
                            D = operands[1][1][1]
                            potential_subtrees = [
                                [operator, [['-', [A, B]], ['-', [C, D]]]],
                                [operator, [['-', [B, A]], ['-', [D, C]]]],  # Swapped
                            ]
                            sorted_potential_subtrees = sorted(potential_subtrees, key=self.operand_key)

                            if potential_subtrees[0] != sorted_potential_subtrees[0]:
                                if verbose:
                                    print(f'Swapping positions from [{operator}, -, A, B, -, C, D]')
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append(sorted_potential_subtrees[0])
                                i -= 1
                                continue

                if operator == '+':
                    # Check for the patterns [+, A, 0] -> [A] or [+, 0, A] -> [A]
                    if ['<0>'] in operands:
                        if verbose:
                            print('Applying [+, A, 0] -> [A]')
                        _ = [stack.pop() for _ in range(arity)]
                        # print(operands)
                        stack.append(operands[0] if operands[1] == '<0>' else operands[1])
                        # print(stack)
                        i -= 1
                        continue

                    # Check for the pattern [+, *, A, B, *, A, C] -> [*, A, +, B, C]
                    if len(operands[0]) == 2 and operands[0][0] == operands[1][0] == '*' and len(operands[0][1]) == 2 and operands[0][1][0] == operands[1][1][0]:
                        if verbose:
                            print('Applying [+, *, A, B, *, A, C] -> [*, A, +, B, C]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['*', [operands[0][1][0], ['+', [operands[0][1][1], operands[1][1][1]]]]])
                        i -= 1
                        continue

                    # Check for the pattern [+, /, A, B, /, C, B] -> [/, +, A, C, B]
                    if len(operands[0]) == 2 and operands[0][0] == operands[1][0] == '/' and len(operands[0][1]) == 2 and operands[0][1][1] == operands[1][1][1]:
                        if verbose:
                            print('Applying [+, /, A, B, /, C, B] -> [/, +, A, C, B]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['/', [['+', [operands[0][1][0], operands[1][1][0]]], operands[0][1][1]]])
                        i -= 1
                        continue

                    # Check for the pattern [+, abs, A, abs, A] -> [abs, +, A, A]
                    if len(operands[0]) == 2 and operands[0][0] == 'abs' and operands[0] == operands[1]:
                        if verbose:
                            print('Applying [+, abs, A, abs, A] -> [abs, +, A, A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['abs', [['+', [operands[0][1][0], operands[0][1][0]]]]])
                        i -= 1
                        continue

                if operator == '-':
                    # Check for the patterns [-, A, 0] -> [A]
                    if operands[1] == ['<0>']:
                        if verbose:
                            print('Applying [-, A, 0] -> [A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(operands[0])
                        i -= 1
                        continue

                    # Check for the patterns [-, 0, A] -> [neg A]
                    if operands[0] == ['<0>']:
                        if verbose:
                            print('Applying [-, 0, A] -> [neg A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['neg', [operands[1]]])
                        i -= 1
                        continue

                    # Check for the pattern [-, *, A, B, *, A, C] -> [*, A, -, B, C]
                    if len(operands[0]) == 2 and operands[0][0] == operands[1][0] == '*' and len(operands[0][1]) == 2 and operands[0][1][0] == operands[1][1][0]:
                        if verbose:
                            print('Applying [-, *, A, B, *, A, C] -> [*, A, -, B, C]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['*', [operands[0][1][0], ['-', [operands[0][1][1], operands[1][1][1]]]]])
                        i -= 1
                        continue

                    # Check for the pattern [-, /, A, B, /, C, B] -> [/, -, A, C, B]
                    if len(operands[0]) == 2 and operands[0][0] == operands[1][0] == '/' and len(operands[0][1]) == 2 and operands[0][1][1] == operands[1][1][1]:
                        if verbose:
                            print('Applying [-, /, A, B, /, C, B] -> [/, -, A, C, B]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['/', [['-', [operands[0][1][0], operands[1][1][0]]], operands[0][1][1]]])
                        i -= 1
                        continue

                if operator in ['*', '/']:
                    if (len(operands[1]) == 2 and operands[1][0] == 'neg'):
                        # Check for the pattern [*, -, A, B, neg, C] -> [*, -, B, A, C] or [/, -, A, B, neg, C] -> [/, -, B, A, C]
                        if operands[0][0] == '-':
                            A = operands[0][1][0]
                            B = operands[0][1][1]
                            C = operands[1][1][0]

                            if verbose:
                                print(f'Applying [{operator}, -, A, B, neg, C] -> [{operator}, -, B, A, C]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([operator, [['-', [B, A]], C]])
                            # print(stack)
                            i -= 1
                            continue

                        # Check for the pattern [*, A, neg, B] -> [neg, *, A, B] or [/, A, neg, B] -> [neg, /, A, B]
                        if verbose:
                            print(f'Applying [{operator}, A, neg, B] -> [neg, {operator}, A, B]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['neg', [[operator, [operands[0], operands[1][1][0]]]]])
                        i -= 1
                        continue

                    # Same but with the operands swapped
                    if (len(operands[0]) == 2 and operands[0][0] == 'neg'):
                        # Check for the pattern [*, neg, C, -, A, B] -> [*, C, -, B, A] or [/, neg, C, -, A, B] -> [/, C, -, B, A]
                        if operands[0][0] == '-':
                            if verbose:
                                print(f'Applying [{operator}, neg, C, -, A, B] -> [{operator}, C, -, B, A]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([operator, [operands[0][1][0], ['-', [operands[1][1][1], operands[1][1][0]]]]])
                            i -= 1
                            continue

                        # Check for the pattern [*, neg, A, B] -> [neg, *, A, B] or [/, neg, A, B] -> [neg, /, A, B]
                        if verbose:
                            print(f'Applying [{operator}, neg, A, B] -> [neg, {operator}, A, B]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['neg', [[operator, [operands[0][1][0], operands[1]]]]])
                        # print(stack)
                        i -= 1
                        continue

                if operator in ['+', '*', '/']:
                    # Check for the pattern [/, abs, A, abs, B] or [*, abs, A, abs, B] or [+, abs, A, abs, B]
                    # -> [abs, /, A, B] or [abs, *, A, B] or [abs, +, A, B]
                    if len(operands[0]) == 2 and len(operands[1]) == 2 and operands[0][0] == 'abs' and operands[1][0] == 'abs':
                        if verbose:
                            print(f'Applying [{operator}, abs, A, abs, B] -> [abs, {operator}, A, B]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['abs', [[operator, [operands[0][1][0], operands[1][1][0]]]]])
                        i -= 1
                        continue

                if operator == 'inv':
                    # Check for the pattern [inv, exp, A] -> [exp, neg, A]
                    if len(operands[0]) == 2 and operands[0][0] == 'exp':
                        if verbose:
                            print('Applying [inv, exp, A] -> [exp, neg, A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['exp', [['neg', [operands[0][1][0]]]]])
                        i -= 1
                        continue

                if operator == 'exp':
                    # Check for the pattern [exp, +, A, A] -> [pow2, exp, A]
                    if len(operands[0]) == 2 and operands[0][0] == '+' and operands[0][1][0] == operands[0][1][1]:
                        if verbose:
                            print('Applying [exp, +, A, A] -> [pow2, exp, A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['pow2', [[operator, [operands[0][1][0]]]]])
                        i -= 1
                        continue

                if operator.startswith('pow1_'):
                    negative_exponent = int(operator[5:])
                    if negative_exponent % 2 == 0:
                        # Check for the pattern [pow1_<even_integer>, pow<even_integer>, A] -> [abs, A]
                        if len(operands[0]) == 2 and operands[0][0].startswith('pow') and '_' not in operands[0][0] and negative_exponent == int(operands[0][0][3:]):
                            if verbose:
                                print(f'Applying [pow1_{negative_exponent}, pow{negative_exponent}, A] -> [abs, A] for even integer {negative_exponent}')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append(['abs', [operands[0][1][0]]])
                            i -= 1
                            continue

                if operator.startswith('pow'):
                    # Check for the pattern [pow<i>, inv, A] -> [inv, pow<i>, A]
                    if len(operands[0]) == 2 and operands[0][0] == 'inv':
                        if verbose:
                            print('Applying [pow<i>, inv, A] -> [inv, pow<i>, A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(['inv', [[operator, [operands[0][1]][0]]]])
                        i -= 1
                        continue

                    # Identify chains
                    if '_' in operator:
                        p = 1
                        q = int(operator[5:])
                    else:
                        p = int(operator[3:])
                        q = 1
                    operator_chain = [operator]
                    operator_chain_factors = [Fraction(p, q)]
                    current_operand = operands[0]

                    reciprocal_operator_allowed = True

                    while len(current_operand) == 2:
                        if re.match(r'pow1_\d+', current_operand[0]) and reciprocal_operator_allowed:
                            p = 1
                            q = int(re.match(r'pow1_(\d+)', current_operand[0]).group(1))  # type: ignore
                            operator_chain.append(current_operand[0])
                            operator_chain_factors.append(Fraction(p, q))
                            current_operand = current_operand[1]
                        elif '_' not in current_operand[0] and re.match(r'pow\d+', current_operand[0]):
                            p = int(re.match(r'pow(\d+)', current_operand[0]).group(1))  # type: ignore
                            q = 1
                            operator_chain.append(current_operand[0])
                            operator_chain_factors.append(Fraction(p, q))
                            current_operand = current_operand[1]
                            if p % 2 == 0:
                                reciprocal_operator_allowed = False
                        else:
                            break

                    if len(operator_chain_factors) > 1:
                        total_fraction: Fraction = prod(operator_chain_factors)  # type: ignore

                        # print(operator_chain_factors, total_fraction)

                        # Factorize p into at most self.max_power or self.max_fractional_power
                        p_factors = self.factorize_to_at_most(total_fraction.numerator, self.max_power)
                        q_factors = self.factorize_to_at_most(total_fraction.denominator, self.max_fractional_power)

                        # Construct the new operators
                        new_operators = []
                        for p in p_factors:
                            if p == 1:
                                continue
                            new_operators.append(f'pow{p}')

                        for q in q_factors:
                            if q == 1:
                                continue
                            new_operators.append(f'pow1_{q}')

                        if len(new_operators) == 0:
                            new_chain = current_operand[0]
                            # print(f'0: {new_chain}')
                        else:
                            new_chain = [new_operators[-1], current_operand]
                            # print(f'1: {new_chain}')
                            for op in new_operators[-2::-1]:
                                new_chain = [op, [new_chain]]
                                # print(f'2: {new_chain}')

                        if verbose:
                            print('Applying power chain simplification')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(new_chain)
                        # print(stack)
                        i -= 1
                        continue

                if operator == 'abs':
                    if len(operands[0]) == 2:
                        # Check for the pattern [abs, <positive_operator>, A] -> [<positive_operator>, A]
                        if operands[0][0] in self.positive_operators:
                            if verbose:
                                print(f'Applying [abs, {operands[0][0]}, A] -> [{operands[0][0]}, A] for positive operator {operands[0][0]}')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([operands[0][0], [operands[0][1][0]]])
                            i -= 1
                            continue

                        # Check for the pattern [abs, inv, A] -> [inv, abs, A]
                        if operands[0][0] == 'inv':
                            if verbose:
                                print('Applying [abs, inv, A] -> [inv, abs, A]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append(['inv', [[operator, [operands[0][1][0]]]]])
                            i -= 1
                            continue

                if operator in self.monotonous_increasing_operators and operator in self.antisymmetric_operators:
                    if len(operands[0]) == 2:
                        if operands[0][0] == 'abs':
                            if verbose:
                                print(f'Applying [{operator}, abs, A] -> [abs, {operator}, A] for antisymmetric monotonous increasing operator {operator}')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append(['abs', [[operator, [operands[0][1][0]]]]])
                            i -= 1
                            continue

                if operator in self.symmetric_operators and len(operands[0]) == 2:
                    # Check for the pattern [<symmetric_operator>, neg, A] -> [<symmetric_operator>, A]
                    if operands[0][0] == 'neg':
                        if verbose:
                            print(f'Applying [{operator}, neg, A] -> [{operator}, A] for symmetric operator {operator}')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append([operator, [operands[0][1][0]]])
                        i -= 1
                        continue

                    if operands[0][0] == '-':
                        if verbose:
                            print('Sorting operands of - in symmetric operator')
                        unique_operands_to_sort = operands[0][1]
                        unique_sorted_operands = sorted(unique_operands_to_sort, key=self.operand_key)
                        if unique_operands_to_sort != unique_sorted_operands:
                            operands[0][1] = unique_sorted_operands
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([operator, operands])
                            i -= 1
                            continue

                    if operands[0][0] == 'abs':
                        if verbose:
                            print(f'Applying [{operator}, abs, A] -> [{operator}, A]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append([operator, operands[0][1]])
                        i -= 1
                        continue

                if operator in self.operator_inverses:
                    operator_inverse = self.operator_inverses[operator]

                    if arity == 1:
                        # Check for the pattern [inv, inv, A] -> [A] or [neg, neg, A] -> [A]
                        if operands[0][0] == operator_inverse:
                            if verbose:
                                print(operator, operator_inverse, operands)
                                print(f'Applying [{operator}, {operator_inverse}, A] -> [A]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append(operands[0][1][0])
                            i -= 1
                            continue

                # Check for the pattern [<antisymmetric_operator>, neg, A] -> [neg, <antisymmetric_operator>, A]
                if operator in self.antisymmetric_operators and isinstance(operands[0], list) and operands[0][0] == 'neg':
                    if verbose:
                        print(f'Applying [{operator}, neg, A] -> [neg, {operator}, A] for antisymmetric operator {operator}')
                    _ = [stack.pop() for _ in range(arity)]
                    stack.append(['neg', [[operator, [operands[0][1][0]]]]])
                    # print(stack)
                    i -= 1
                    continue

                if operator == 'neg':
                    # Identify chains of antisymmetric operators
                    operator_chain = []
                    current_operand = operands[0]

                    while len(current_operand) == 2 and current_operand[0] in self.antisymmetric_operators:
                        operator_chain.append(current_operand[0])
                        current_operand = current_operand[1][0]

                    # # BFS through chains of sign-preserving ['*', '/'] operators and flip the operands of the first encountered '-' operator
                    # current_operand_copy = deepcopy(current_operand)
                    # queue = [current_operand_copy]
                    # while queue:
                    #     current_operand_copy = queue.pop(0)
                    #     if len(current_operand_copy) == 2 and current_operand_copy[0] in ['*', '/']:
                    #         queue.append(current_operand_copy[1][0])
                    #         queue.append(current_operand_copy[1][1])
                    #     elif len(current_operand_copy) == 2 and current_operand_copy[0] in self.antisymmetric_operators:
                    #         queue.append(current_operand_copy[1][0])
                    #     elif len(current_operand_copy) == 2 and current_operand_copy[0] == '-':
                    #         break

                    # print('Current operand:', current_operand_copy)

                    if len(current_operand) == 2 and current_operand[0] in ['*', '/']:
                        # Check for the patterns [neg, X, *, -, A, B, C] -> [X, *, -, B, A, C] or [neg, X, /, -, A, B, C] -> [X, /, -, B, A, C] for chain of antisymmetric operators X
                        if current_operand[1][0][0] == '-':
                            if verbose:
                                print(f'Applying [neg, X, {current_operand[0]}, -, A, B, C] -> [X, {current_operand[0]}, -, B, A, C] for chain of antisymmetric operators X')
                            A = current_operand[1][0][1][0]
                            B = current_operand[1][0][1][1]
                            C = current_operand[1][1]

                            new_operand = [current_operand[0], [['-', [B, A]], C]]
                            for op in operator_chain[::-1]:
                                new_operand = [op, [new_operand]]

                            _ = [stack.pop() for _ in range(arity)]
                            stack.append(new_operand)
                            i -= 1
                            continue

                        # Check for the patterns [neg, X, *, A, -, B, C] -> [X,*, A, -, C, B] or [neg, X, /, A, -, B, C] -> [X, /, A, -, C, B] for chain of antisymmetric operators X
                        if current_operand[1][1][0] == '-':
                            if verbose:
                                print(f'Applying [neg, X, {current_operand[0]}, A, -, B, C] -> [X, {current_operand[0]}, A, -, C, B] for chain of antisymmetric operators X')
                            A = current_operand[1][0]
                            B = current_operand[1][1][1][0]
                            C = current_operand[1][1][1][1]

                            new_operand = [current_operand[0], [A, ['-', [C, B]]]]
                            for op in operator_chain[::-1]:
                                new_operand = [op, [new_operand]]

                            _ = [stack.pop() for _ in range(arity)]
                            stack.append(new_operand)
                            i -= 1
                            continue

                    # Check for the pattern [neg, <antisymmetric_operator>, -, A, B] -> [<antisymmetric_operator>, -, B, A]
                    if len(current_operand) == 2 and current_operand[0] == '-':
                        if verbose:
                            print('Applying [neg, X, -, A, B] -> [X,  -, B, A] for antisymmetric operators X')
                        A = current_operand[1][0]
                        B = current_operand[1][1]

                        new_operand = ['-', [B, A]]
                        for op in operator_chain[::-1]:
                            new_operand = [op, [new_operand]]

                        _ = [stack.pop() for _ in range(arity)]
                        stack.append(new_operand)
                        i -= 1
                        continue

                if operator in self.inverse_base:
                    unary_inverse, binary_inverse, neutral_element = self.inverse_base[operator]

                    if len(operands[0]) == 2 and len(operands[1]) == 2:
                        # Check for the pattern [*, inv, A, inv, B] -> [inv, *, A, B] or [+, neg, A, neg, B] -> [neg, +, A, B]
                        if operands[0][0] == unary_inverse and operands[1][0] == unary_inverse:
                            if verbose:
                                print(f'Applying [{operator}, {unary_inverse}, A, {unary_inverse}, B] -> [{unary_inverse}, {operator}, A, B]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([unary_inverse, [[operator, [operands[0][1][0], operands[1][1][0]]]]])
                            i -= 1
                            continue

                        # Check for the pattern [*, /, A, B, /, B, A] -> <1> or [+, -, A, B, -, B, A] -> <0>
                        if operands[0][0] == binary_inverse and operands[1][0] == binary_inverse and operands[0][1][0] == operands[1][1][1] and operands[0][1][1] == operands[1][1][0]:
                            # print("Term:", operator, operands)
                            if verbose:
                                print(f'Applying [{operator}, {binary_inverse}, A, B, {binary_inverse}, B, A] -> [{neutral_element}]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([neutral_element])
                            i -= 1
                            continue

                        # Check for the pattern [*, inv, A, /, B, C] -> [/, /, B, A, C] or [+, neg, A, -, B, C] -> [-, -, B, A, C]
                        if operands[0][0] == unary_inverse and operands[1][0] == binary_inverse:
                            if verbose:
                                print(f'Applying [{operator}, {unary_inverse}, A, {binary_inverse}, B, C] -> [{binary_inverse}, {binary_inverse}, B, A, C]')
                            A = operands[0][1][0]
                            B = operands[1][1][0]
                            C = operands[1][1][1]
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([binary_inverse, [[binary_inverse, [B, A]], C]])
                            i -= 1
                            continue

                    if len(operands[0]) == 2:
                        # Check for the pattern [*, inv, A, B] -> [/, B, A] or [+, neg, A, B] -> [-, B, A]
                        if operands[0][0] == unary_inverse:
                            if verbose:
                                print(f'Applying [{operator}, {unary_inverse}, A, B] -> [{binary_inverse}, B, A]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([binary_inverse, [operands[1], operands[0][1][0]]])
                            # print(stack)
                            i -= 1
                            continue

                        if len(operands[0][1]) == 2:
                            # Check for the pattern [* /, B, A, A] -> B or [+, -, B, A, A] -> B
                            if operands[1] == operands[0][1][1] and operands[0][0] == binary_inverse:
                                if verbose:
                                    print(f'Applying [{operator}, {binary_inverse}, B, A, A] -> B')
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append(operands[0][1][0])
                                i -= 1
                                continue

                    if len(operands[1]) == 2:
                        # Check for the pattern [*, A, inv, B] -> [/, A, B] or [+, A, neg, B] -> [-, A, B]
                        if operands[1][0] == unary_inverse:
                            if verbose:
                                print(f'Applying [{operator}, A, {unary_inverse}, B] -> [{binary_inverse}, A, B]')
                            _ = [stack.pop() for _ in range(arity)]
                            # print([binary_inverse, [operands[0], operands[1][1][0]]])
                            stack.append([binary_inverse, [operands[0], operands[1][1][0]]])
                            i -= 1
                            continue

                        if len(operands[1][1]) == 2:
                            # Check for the pattern [+, A, -, B, A] -> B or [*, A, /, B, A] -> B
                            if operands[0] == operands[1][1][1] and operands[1][0] == binary_inverse:
                                if verbose:
                                    print(f'Applying [{operator}, A, {binary_inverse}, B, A] -> B')
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append(operands[1][1][0])
                                i -= 1
                                continue

                            # Check for the pattern [+, A, +, B, -, B, A] -> [+, B, B] or [*, A, *, B, /, B, A] -> [*, B, B]
                            if len(operands[1][1][1]) == 2 and len(operands[1][1][1][1]) == 2 and \
                                    operands[1][0] == operator and operands[1][1][1][0] == binary_inverse and operands[1][1][1][1][1] == operands[0] and operands[1][1][1][1][0] == operands[1][1][0]:
                                if verbose:
                                    print(f'Applying [{operator}, A, {operator}, B, {binary_inverse}, B, A] -> [{operator}, B, B]')
                                B = operands[1][1][0]
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append([operator, [B, B]])
                                i -= 1
                                continue

                        # Check for the pattern [*, A, /, B, C]
                        if operands[1][0] == binary_inverse:
                            A = operands[0]
                            B = operands[1][1][0]
                            C = operands[1][1][1]
                            potential_subtrees = [
                                [operator, [A, [binary_inverse, [B, C]]]],
                                [operator, [B, [binary_inverse, [A, C]]]],
                            ]
                            sorted_potential_subtrees = sorted(potential_subtrees, key=self.operand_key)

                            if potential_subtrees[0] != sorted_potential_subtrees[0]:
                                if verbose:
                                    print(f'Swapping positions from [{operator}, A, {binary_inverse}, B, C]')
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append(sorted_potential_subtrees[0])
                                i -= 1
                                continue

                if operator in self.inverse_binary:
                    base, unary_inverse, neutral_element = self.inverse_binary[operator]

                    if len(operands[0]) == 2:
                        if len(operands[0][1]) == 2:
                            # Check for the pattern [/, *, A, B, A] -> B or [-, +, A, B, A] -> B
                            if operands[1] == operands[0][1][0] and operands[0][0] == base:
                                if verbose:
                                    print(f'Applying [{operator}, {base}, A, B, A] -> B')
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append(operands[0][1][1])
                                # print(stack)
                                i -= 1
                                continue

                            # Check for the pattern [/, *, A, B, C] -> [*, A, /, B, C] or [-, +, A, B, C] -> [+, A, -, B, C]
                            if operands[0][0] == base:
                                if verbose:
                                    # print(operator, operands)
                                    print(f'Applying [{operator}, {base}, A, B, C] -> [{base}, A, {operator}, B, C]')
                                A = operands[0][1][0]
                                B = operands[0][1][1]
                                C = operands[1]
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append([base, [A, [operator, [B, C]]]])
                                i -= 1
                                continue

                            # Check for the pattern [/, /, A, B, C] or [-, -, A, B, C] and sort B and C
                            if operands[0][0] == operator:
                                unique_operands_to_sort = [operands[0][1][1], operands[1]]
                                unique_sorted_operands = sorted(unique_operands_to_sort, key=self.operand_key)
                                if unique_operands_to_sort != unique_sorted_operands:
                                    if verbose:
                                        print(f'Sorting B and C in [{operator}, {operator}, A, B, C]')
                                    operands[0][1][1] = unique_sorted_operands[0]
                                    operands[1] = unique_sorted_operands[1]
                                    _ = [stack.pop() for _ in range(arity)]
                                    stack.append([operator, operands])
                                    i -= 1
                                    continue

                        # Check for the pattern [/, inv, A, B] ->  [inv, *, A, B] or [-, neg, A, B] ->  [neg, +, A, B]
                        if operands[0][0] == unary_inverse:
                            if verbose:
                                print(f'Applying [{operator}, {unary_inverse}, A, B] -> [{unary_inverse}, {base}, A, B]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([unary_inverse, [[base, [operands[0][1][0], operands[1]]]]])
                            i -= 1
                            continue

                    if len(operands[1]) == 2:
                        # Check for the pattern [/, A, inv, B] ->  [*, A, B] or [-, A, neg, B] ->  [+, A, B]
                        if operands[1][0] == unary_inverse:
                            if verbose:
                                print(f'Applying [{operator}, A, {unary_inverse}, B] -> [{base}, A, B]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([base, [operands[0], operands[1][1][0]]])
                            i -= 1
                            continue

                        if len(operands[1][1]) == 2:
                            # Check for the pattern [/, A, *, A, B] -> [inv, B] or [-, A, +, A, B] -> [neg, B]
                            if operands[0] == operands[1][1][0]:
                                if verbose:
                                    print(f'Applying [{operator}, A, {base}, A, B] -> [{unary_inverse}, B]')
                                _ = [stack.pop() for _ in range(arity)]
                                # print([unary_inverse, [operands[1][1][1]]])
                                stack.append([unary_inverse, [operands[1][1][1]]])
                                i -= 1
                                continue

                            # Check for the pattern [/, A, /, B, C] -> [*, A, /, C, B] or [-, A, -, B, C] -> [+, A, -, C, B]
                            if operands[1][0] == operator:
                                if verbose:
                                    print(f'Applying [{operator}, A, {operator}, B, C] -> [{base}, A, {operator}, C, B]')
                                _ = [stack.pop() for _ in range(arity)]
                                stack.append([base, [operands[0], [operator, [operands[1][1][1], operands[1][1][0]]]]])
                                i -= 1
                                continue

                    # Check for the patterns [-, A, A] -> [0] and [/, A, A] -> [1]
                    if operands[0] == operands[1]:
                        if verbose:
                            print(f'Applying [{operator}, A, A] -> [{neutral_element}]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append([neutral_element])
                        # print(stack)
                        i -= 1
                        continue

                if operator in self.inverse_unary:
                    base, binary_inverse, neutral_element = self.inverse_unary[operator]

                    if len(operands[0]) == 2:
                        # Check for the pattern [inv, /, A, B] -> [/, B, A] or [neg, -, A, B] -> [-, B, A]
                        if operands[0][0] == binary_inverse:
                            if verbose:
                                print(f'Applying [{operator}, {binary_inverse}, A, B] -> [{binary_inverse}, B, A]')
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([binary_inverse, [operands[0][1][1], operands[0][1][0]]])
                            i -= 1
                            continue

                        # Check for the pattern [inv, *, A, /, B, C]  -> [/, /, C, B, A] or [neg, +, A, -, B, C]  -> [-, -, C, B, A]  # TODO: Combine this with above (in general for commutative operators)
                        if operands[0][0] == base and operands[0][1][1][0] == binary_inverse:
                            if verbose:
                                print(f'Applying [{operator}, {base}, A, {binary_inverse}, B, C] -> [{binary_inverse}, {binary_inverse}, C, B, A]')
                            A = operands[0][1][0]
                            B = operands[0][1][1][1][0]
                            C = operands[0][1][1][1][1]
                            _ = [stack.pop() for _ in range(arity)]
                            stack.append([binary_inverse, [[binary_inverse, [C, B]], A]])
                            i -= 1
                            continue

                if operator in self.commutative_operators:
                    # Check for the pattern [*, *, A, B, C] -> [*, A, *, B, C] or [+, +, A, B, C] -> [+, A, +, B, C]
                    if len(operands[0]) == 2 and operator == operands[0][0]:
                        if verbose:
                            print(f'Applying [{operator}, {operator}, A, B, C] -> [{operator}, A, {operator}, B, C]')
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append([operator, [operands[0][1][0], [operator, [operands[0][1][1], operands[1]]]]])
                        # print(stack)
                        i -= 1
                        continue

                    if verbose:
                        # print(f'Sorting operands of commutative subtree with operator {operator}: {[flatten_nested_list(op)[::-1] for op in operands]}', end='')
                        print(f'Sorting operands of commutative subtree with operator {operator}: {operands}', end='')

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
                                # Steop: Leaf
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

                    if verbose:
                        # print(f' -> {[flatten_nested_list(op)[::-1] for op in sorted_operands]}')
                        print(f' -> {sorted_operands}')

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
    def operators_to_realizations(self, prefix_expression: list[str]) -> list[str]:
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

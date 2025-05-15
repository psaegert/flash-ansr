from typing import Any
import re
from copy import deepcopy
from math import prod

from fractions import Fraction
from flash_ansr.expressions.utils import codify, flatten_nested_list


def _simplify_flash(self: Any, expression: list[str], verbose: bool = False, debug: bool = False) -> list[str]:
    '''
    Simplify an expression

    Parameters
    ----------
    self: ExpressionSpace
        The current instance
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

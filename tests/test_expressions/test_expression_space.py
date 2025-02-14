from flash_ansr import ExpressionSpace, get_path
from flash_ansr.expressions.utils import codify, num_to_constants

import numpy as np

import unittest


class TestExpressionSpace(unittest.TestCase):
    def test_prefix_to_infix(self):
        test_expression = ["*", "sin", "**", "1.234", "x1", "-423"]
        space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

        infix_equation_string = space.prefix_to_infix(test_expression)

        assert infix_equation_string == 'sin(1.234 ** x1) * -423'

    def test_from_config_dict(self):
        space = ExpressionSpace(operators={
            '+': {
                "realization": "+",
                "alias": ["add", "plus"],
                "inverse": "-",
                "arity": 2,
                "weight": 10,
                "precedence": 1,
                "commutative": True,
                "symmetry": 0,
                "positive": False,
                "monotonicity": 0
            }},
            variables=1)

        assert space.variables == ["x1"]
        assert space.operator_arity == {'+': 2}

    def test_from_config_file(self):
        space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

        assert space.variables == ["x1", "x2", "x3"]
        assert space.operator_arity['*'] == 2

    def test_check_valid(self):
        space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

        assert space.is_valid(["*", "sin", "pow2", "<num>", "x1"], verbose=True) is True

    def test_check_valid_invalid(self):
        space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

        assert space.is_valid(["*", "sin", "pow2" "<num>", "x1", "<num>", "<num>"], verbose=True) is False

    def test_parse_infix_expression(self):
        space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

        assert isinstance(space.parse_expression("sin(2.1**x1) * -4370"), list)

    def test_check_valid_invalid_token(self):
        space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

        assert space.is_valid(["*", "sin", "numpy.does_not_exist", "<num>", "x1", "<num>"], verbose=True) is False

    def test_check_valid_invalid_too_many_operands(self):
        space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

        assert space.is_valid(["*", "sin", "numpy.does_not_exist", "<num>", "x1", "<num>", "x1", "<num>", "<num>"], verbose=True) is False

    def test_check_valid_invalid_too_few_operands(self):
        space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

        assert space.is_valid(["*", "sin", "<num>"], verbose=True) is False

    def test_collapse_numeric_subtrees(self):
        space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

        expression = ['+', 'x1', '*', '<num>', '<num>']

        assert space.is_valid(expression)

        collapsed_expression = space.simplify(expression)

        assert collapsed_expression == ['+', '<num>', 'x1']

    def test_infix_to_prefix(self):
        space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

        infix_expressions = [
            ('sin(2.1**x1) * -4370', ['*', 'sin', '**', '2.1', 'x1', 'neg', '4370']),
            ('x**2', ['**', 'x', '2']),
            ('-x**2', ['neg', '**', 'x', '2']),
            ('(-x)**2', ['**', 'neg', 'x', '2']),
            ('-(x**2)', ['neg', '**', 'x', '2']),
            ('sin(x1) + 1', ['+', 'sin', 'x1', '1']),
            ('x1**2 + 1', ['+', '**', 'x1', '2', '1']),
            ('x1**2 + 2*x1 + 1', ['+', '**', 'x1', '2', '+', '*', '2', 'x1', '1']),
            ('exp(- (x - 1.1)**2 / 1.2)', ['exp', '/', 'neg', '**', '-', 'x', '1.1', '2', '1.2'])
        ]

        for infix_expression, target_prefix_expression in infix_expressions:
            print(infix_expression, target_prefix_expression)
            prefix_expression = space.infix_to_prefix(infix_expression)
            assert prefix_expression == target_prefix_expression
            print()


class TestSimplify(unittest.TestCase):
    def setUp(self):
        self.space = ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml'))

    def test_simplify(self):
        test_expressions = [
            (['+', 'x1', '*', '<num>', '<num>'], ['+', '<num>', 'x1']),
            (['+', 'x1', '*', '1.2', '2'], ['+', 'x1', '2.4']),
            (['+', '-', 'x1', 'x1', 'x2'], ['x2']),
            (['*', '-', 'x1', 'x1', 'x2'], ['<num>']),
            (['*', 'x1', 'x1'], ['pow2', 'x1']),
            (['*', 'x1', 'neg', 'x1'], ['neg', 'pow2', 'x1']),
            (['*', 'x2', '+', 'x1', 'x1'], ['*', 'x1', '+', 'x2', 'x2']),
            (['*', 'x1', '+', 'x1', 'x1'], ['+', 'pow2', 'x1', 'pow2', 'x1']),
            (['*', '-', 'x1', 'x2', '-', 'x2', 'x1'], ['neg', 'pow2', '-', 'x1', 'x2']),
            (['*', '-', 'x3', 'x2', '-', 'x2', 'x1'], ['*', '-', 'x1', 'x2', '-', 'x2', 'x3']),
            (['/', 'x1', '/', 'x2', 'x2'], ['x1']),
            (['/', '/', 'x2', 'x2', 'x1'], ['inv', 'x1']),
            (['/', 'abs', 'x1', 'x1'], ['/', 'x1', 'abs', 'x1']),
            # [/, +, A, A, +, B, B] -> [/, A, B]
            (['/', '+', 'x1', 'x1', '+', 'x2', 'x2'], ['/', 'x1', 'x2']),
            #  [/, -, A, B, -, B, A] -> [neg, <1>]
            (['/', '-', 'x1', 'x2', '-', 'x2', 'x1'], ['<num>']),
            # Swapping positions from [{operator}, -, A, B, -, C, D]
            (['/', '-', 'x3', 'x2', '-', 'x2', 'x1'], ['/', '-', 'x2', 'x3', '-', 'x1', 'x2']),
            # [+, *, A, B, *, A, B] -> [*, A, +, B, B]
            (['+', '*', 'x1', 'x2', '*', 'x1', 'x2'], ['*', 'x1', '+', 'x2', 'x2']),
            # [+, /, A, B, /, A, B] -> [/, +, A, A, B]
            (['+', '/', 'x1', 'x2', '/', 'x1', 'x2'], ['/', '+', 'x1', 'x1', 'x2']),
            # [-, A, 0] -> [A]
            (['-', 'x1', '-', 'x2', 'x2'], ['x1']),
            (['-', '-', 'x2', 'x2', 'x1'], ['neg', 'x1']),
            # [*, -, A, B, neg, C] -> [*, -, B, A, C]
            (['*', '-', 'x1', 'x2', 'neg', 'x3'], ['*', 'x3', '-', 'x2', 'x1']),
            # [*, A, neg, B] -> [neg, *, A, B]
            (['*', 'x1', 'neg', 'x2'], ['neg', '*', 'x1', 'x2']),
            # [*, neg, C, -, A, B] -> [*, C, -, B, A]
            (['*', 'neg', 'x3', '-', 'x1', 'x2'], ['*', 'x3', '-', 'x2', 'x1']),
            # [+, abs, A, abs, A] -> [abs, +, A, A]
            (['+', 'abs', 'x1', 'abs', 'x1'], ['abs', '+', 'x1', 'x1']),
            # [*, abs, A, abs, B] -> [abs, *, A, B]
            (['*', 'abs', 'x1', 'abs', 'x2'], ['abs', '*', 'x1', 'x2']),
            # [inv, exp, A] -> [exp, neg, A]
            (['inv', 'exp', 'x1'], ['exp', 'neg', 'x1']),
            # [exp, +, A, A] -> [pow2, exp, A]
            (['exp', '+', 'x1', 'x1'], ['pow2', 'exp', 'x1']),
            # [pow1_<even_integer>, pow<even_integer>, A] -> [abs, A]
            (['pow1_2', 'pow2', 'x1'], ['abs', 'x1']),
            # [pow<i>, inv, A] -> [inv, pow<i>, A]
            (['pow2', 'inv', 'x1'], ['inv', 'pow2', 'x1']),
            # Power simplifications
            (['pow1_2', 'pow4', 'x1'], ['pow2', 'x1']),
            (['pow1_3', 'pow3', 'x1'], ['x1']),
            (['pow1_3', 'pow3', 'pow5', 'x1'], ['pow5', 'x1']),
            (['pow1_3', 'pow3', 'pow2', 'x1'], ['pow2', 'x1']),
            (['pow1_3', 'pow2', 'pow3', 'x1'], ['pow2', 'x1']),
            (['pow4', 'pow1_3', 'x1'], ['pow4', 'pow1_3', 'x1']),
            # [abs, <positive_operator>, A] -> [<positive_operator>, A]
            (['abs', 'exp', 'x1'], ['exp', 'x1']),
            # [abs, inv, A] -> [inv, abs, A]
            (['abs', 'inv', 'x1'], ['inv', 'abs', 'x1']),
            # [{operator}, abs, A] -> [abs, {operator}, A] for monotonous increasing operator {operator}
            (['pow3', 'abs', 'x1'], ['abs', 'pow3', 'x1']),
            (['exp', 'abs', 'x1'], ['exp', 'abs', 'x1']),
            # [/, abs, A, abs, B] -> [abs, /, A, B]
            (['/', 'abs', 'x1', 'abs', 'x2'], ['abs', '/', 'x1', 'x2']),
            # [<symmetric_operator>, neg, A] -> [<symmetric_operator>, A]
            (['pow2', 'neg', 'x1'], ['pow2', 'x1']),
            # Sorting operands of - in symmetric operator
            (['pow2', '-', 'x1', 'x2'], ['pow2', '-', 'x1', 'x2']),
            (['pow2', '-', 'x2', 'x1'], ['pow2', '-', 'x1', 'x2']),
            # [{operator}, abs, A] -> [{operator}, A] for symmetric operator {operator}
            (['cos', 'abs', 'x1'], ['cos', 'x1']),
            # [inv, inv, A] -> [A]
            (['inv', 'inv', 'x1'], ['x1']),
            # [<antisymmetric_operator>, neg, A]
            (['sin', 'neg', 'x1'], ['neg', 'sin', 'x1']),
            # [neg, X, /, -, A, B, C] -> [X, /, -, B, A, C] for chain of antisymmetric operators X
            (['neg', 'sin', '/', '-', 'x1', 'x2', 'x3'], ['sin', '/', '-', 'x2', 'x1', 'x3']),
            # [neg, <antisymmetric_operator>, -, A, B] -> [<antisymmetric_operator>, -, B, A]
            (['neg', 'sin', '-', 'x1', 'x2'], ['sin', '-', 'x2', 'x1']),
            # [*, inv, A, inv, B] -> [inv, *, A, B]
            (['*', 'inv', 'x1', 'inv', 'x2'], ['inv', '*', 'x1', 'x2']),
            # [*, /, A, B, /, B, A] -> <1>
            (['*', '/', 'x1', 'x2', '/', 'x2', 'x1'], ['<num>']),
            # [*, inv, A, /, B, C] -> [/, /, B, A, C]
            (['*', 'inv', 'x1', '/', 'x2', 'x3'], ['/', '/', 'x2', 'x1', 'x3']),
            # [*, inv, A, B] -> [/, B, A]
            (['*', 'inv', 'x1', 'x2'], ['/', 'x2', 'x1']),
            # [* /, B, A, A] -> B
            (['*', '/', 'x2', 'x1', 'x1'], ['x2']),
            # [*, A, inv, B] -> [/, A, B]
            (['*', 'x1', 'inv', 'x2'], ['/', 'x1', 'x2']),
            # [+, A, -, B, A] -> B
            (['+', 'x1', '-', 'x2', 'x1'], ['x2']),
            # [+, A, +, B, -, B, A] -> [+, B, B]
            (['+', 'x1', '+', 'x2', '-', 'x2', 'x1'], ['+', 'x2', 'x2']),
            # [*, A, /, B, C] sort A and B
            (['*', 'x2', '/', 'x1', 'x3'], ['*', 'x1', '/', 'x2', 'x3']),
            # [-, +, A, B, A] -> B
            (['-', '+', 'x1', 'x2', 'x1'], ['x2']),
            # [-, +, B, A, A] -> B
            (['-', '+', 'x2', 'x1', 'x1'], ['x2']),
            # [/, *, A, B, C] -> [*, B, /, A, C]
            (['/', '*', 'x1', 'x2', 'x3'], ['*', 'x1', '/', 'x2', 'x3']),
            # [-, -, A, B, C] sort B and C
            (['-', '-', 'x1', 'x3', 'x2'], ['-', '-', 'x1', 'x2', 'x3']),
            # [/, inv, A, B] ->  [inv, *, A, B]
            (['/', 'inv', 'x1', 'x2'], ['inv', '*', 'x1', 'x2']),
            # [/, A, inv, B] ->  [*, A, B]
            (['/', 'x1', 'inv', 'x2'], ['*', 'x1', 'x2']),
            # [/, A, *, A, B] -> [inv, B]
            (['/', 'x1', '*', 'x1', 'x2'], ['inv', 'x2']),
            # [/, A, *, B, A] -> [inv, B]
            (['/', 'x1', '*', 'x2', 'x1'], ['inv', 'x2']),
            # [/, A, /, B, C] -> [*, A, /, C, B]
            (['/', 'x1', '/', 'x2', 'x3'], ['*', 'x1', '/', 'x3', 'x2']),
            # [inv, /, A, B] -> [/, B, A]
            (['inv', '/', 'x1', 'x2'], ['/', 'x2', 'x1']),
            # [inv, *, /, B, C, A]  -> [/, /, C, B, A]
            (['inv', '*', '/', 'x2', 'x3', 'x1'], ['/', '/', 'x3', 'x1', 'x2']),  # Implicit sorting
            # [inv, *, A, /, B, C]  -> [/, /, C, B, A]
            (['inv', '*', 'x1', '/', 'x2', 'x3'], ['/', '/', 'x3', 'x1', 'x2']),  # Implicit sorting
            # [+, +, A, B, C] -> [+, A, +, B, C]
            (['+', '+', 'x1', 'x2', 'x3'], ['+', 'x1', '+', 'x2', 'x3']),
            # Misc
            (['*', '/', 'x1', 'x3', '/', 'x2', 'x1'], ['/', 'x2', 'x3']),
            (['*', 'x1', '*', 'x2', '/', 'x2', 'x1'], ['pow2', 'x2']),
            (['*', 'x1', '*', 'x3', '/', 'x2', 'x1'], ['*', 'x1', '*', 'x2', '/', 'x3', 'x1']),
            (['+', '*', 'x1', 'x2', '*', 'x1', 'x2'], ['*', 'x1', '+', 'x2', 'x2']),
            (['+', '*', 'x1', 'x2', '*', 'x1', 'x3'], ['*', 'x1', '+', 'x2', 'x3']),
            (['+', '/', 'x1', 'x2', '/', 'x3', 'x2'], ['/', '+', 'x1', 'x3', 'x2']),
        ]

        X = np.random.uniform(-10, 10, (512, 3))

        exceptions = [
            ['pow1_3', 'pow3', 'x1'],
            ['pow1_3', 'pow3', 'pow5', 'x1'],
        ]

        for expression, target_simplified_expression in test_expressions:
            print(expression, target_simplified_expression)
            assert self.space.is_valid(expression)

            simplified_expression = self.space.simplify(expression, verbose=True, debug=False)

            assert self.space.is_valid(simplified_expression)
            assert simplified_expression == target_simplified_expression

            target_executable_prefix_expression = self.space.operators_to_realizations(expression)
            target_prefix_expression_with_constants, target_constants = num_to_constants(target_executable_prefix_expression)

            executable_prefix_expression = self.space.operators_to_realizations(simplified_expression)
            prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression)

            if len(constants) == 0 and len(target_constants) == 0 and expression not in exceptions:
                code_string = self.space.prefix_to_infix(target_prefix_expression_with_constants, realization=True)
                code = codify(code_string, self.space.variables)
                f = self.space.code_to_lambda(code)
                raw_evaluation_signature = tuple(f(*X.T).round(4))

                code_string = self.space.prefix_to_infix(prefix_expression_with_constants, realization=True)
                code = codify(code_string, self.space.variables)
                f = self.space.code_to_lambda(code)
                evaluation_signature = tuple(f(*X.T).round(4))
                print(np.allclose(raw_evaluation_signature, evaluation_signature, equal_nan=True))
                assert np.allclose(raw_evaluation_signature, evaluation_signature, equal_nan=True)

            print()

from simplipy import SimpliPyEngine

import unittest


class TestSimpliPyEngine(unittest.TestCase):
    def test_prefix_to_infix(self):
        test_expression = ["*", "sin", "**", "1.234", "x1", "-423"]
        space = SimpliPyEngine.load('dev_7-3')

        infix_equation_string = space.prefix_to_infix(test_expression)

        assert infix_equation_string == 'sin(1.234 ** x1) * -423'

    def test_from_config_dict(self):
        space = SimpliPyEngine(operators={
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
            }})

        assert space.operator_arity == {'+': 2}

    def test_from_config_file(self):
        space = SimpliPyEngine.load('dev_7-3', install=True)

        assert space.operator_arity['*'] == 2

    def test_check_valid(self):
        space = SimpliPyEngine.load('dev_7-3', install=True)

        assert space.is_valid(["*", "sin", "pow2", "<constant>", "x1"], verbose=True) is True

    def test_check_valid_invalid(self):
        space = SimpliPyEngine.load('dev_7-3', install=True)

        assert space.is_valid(["*", "sin", "pow2" "<constant>", "x1", "<constant>", "<constant>"], verbose=True) is False

    def test_parse_infix_expression(self):
        space = SimpliPyEngine.load('dev_7-3', install=True)

        assert isinstance(space.parse("sin(2.1**x1) * -4370"), list)

    def test_check_valid_invalid_token(self):
        space = SimpliPyEngine.load('dev_7-3', install=True)

        assert space.is_valid(["*", "sin", "numpy.does_not_exist", "<constant>", "x1", "<constant>"], verbose=True) is False

    def test_check_valid_invalid_too_many_operands(self):
        space = SimpliPyEngine.load('dev_7-3', install=True)

        assert space.is_valid(["*", "sin", "numpy.does_not_exist", "<constant>", "x1", "<constant>", "x1", "<constant>", "<constant>"], verbose=True) is False

    def test_check_valid_invalid_too_few_operands(self):
        space = SimpliPyEngine.load('dev_7-3', install=True)

        assert space.is_valid(["*", "sin", "<constant>"], verbose=True) is False

    def test_collapse_numeric_subtrees(self):
        space = SimpliPyEngine.load('dev_7-3', install=True)

        expression = ['+', 'x1', '*', '<constant>', '<constant>']

        assert space.is_valid(expression)

        collapsed_expression = space.simplify(expression, max_pattern_length=4)

        assert collapsed_expression == ['+', '<constant>', 'x1']

    def test_infix_to_prefix(self):
        space = SimpliPyEngine.load('dev_7-3', install=True)

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

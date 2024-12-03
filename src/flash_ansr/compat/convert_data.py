from abc import abstractmethod
import warnings

import pandas as pd

from tqdm import tqdm

from flash_ansr import ExpressionSpace, SkeletonPool  # Parse expressions with ExpressionSpace.parse_infix_expression
from flash_ansr.expressions.utils import codify, num_to_constants


class TestSetParaser:
    @abstractmethod
    def parse_data(self, test_set_df: pd.DataFrame, expression_space: ExpressionSpace, base_skeleton_pool: SkeletonPool, verbose: bool = False) -> SkeletonPool:
        '''
        Parse the test set data and import it into the skeleton pool.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        expression_space : ExpressionSpace
            The expression space to use for parsing and simplifying the expressions.
        base_skeleton_pool : SkeletonPool
            An initial skeleton pool to add the parsed expressions to.
        verbose : bool, optional
            Whether to print progress information, by default False

        Returns
        -------
        SkeletonPool
            The skeleton pool with the parsed expressions added.
        '''
        pass


class ParserFactory:
    @staticmethod
    def get_parser(parser_name: str) -> TestSetParaser:
        match parser_name:
            case 'soose':
                return SOOSEParser()
            case 'feynman':
                return FeynmanParser()
            case 'nguyen':
                return NguyenParser()
            case _:
                raise ValueError(f'Unknown parser: {parser_name}')


class SOOSEParser(TestSetParaser):
    def parse_data(self, test_set_df: pd.DataFrame, expression_space: ExpressionSpace, base_skeleton_pool: SkeletonPool, verbose: bool = False) -> SkeletonPool:
        '''
        Parse the test set data and import it into the skeleton pool.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        expression_space : ExpressionSpace
            The expression space to use for parsing and simplifying the expressions.
        base_skeleton_pool : SkeletonPool
            An initial skeleton pool to add the parsed expressions to.
        verbose : bool, optional
            Whether to print progress information, by default False

        Returns
        -------
        SkeletonPool
            The skeleton pool with the parsed expressions added.
        '''
        n_invalid_expressions = 0
        n_too_many_variables = 0

        expression_dict = {}
        for expression in tqdm(test_set_df['eq'], disable=not verbose, desc='Parsing and Importing SOOOSE Data'):
            # Parse and simplify
            try:
                prefix_expression = expression_space.parse_expression(expression, mask_numbers=True, too_many_variables='raise')
            except ValueError:
                n_too_many_variables += 1
                print(prefix_expression)
                warnings.warn(f'Expression {expression} has too many variables despite checking the dataset')
                continue

            # Check valid
            if not expression_space.is_valid(prefix_expression, verbose=True):
                n_invalid_expressions += 1
                continue

            prefix_expression = expression_space.simplify(prefix_expression)

            # Codify
            prefix_expression_w_num = expression_space.operators_to_realizations(prefix_expression)
            prefix_expression_w_constants, constants = num_to_constants(prefix_expression_w_num)
            code_string = expression_space.prefix_to_infix(prefix_expression_w_constants, realization=True)
            code = codify(code_string, expression_space.variables + constants)

            # Import
            expression_hash = tuple(prefix_expression)
            expression_dict[expression_hash] = (code, constants)

        print(f'Number of invalid expressions: {n_invalid_expressions} ({n_invalid_expressions / len(test_set_df) * 100:.2f}%)')
        print(f'Number of expressions with too many variables: {n_too_many_variables} ({n_too_many_variables / len(test_set_df) * 100:.2f}%)')

        base_skeleton_pool.skeleton_codes = expression_dict
        base_skeleton_pool.skeletons = list(expression_dict.keys())

        return base_skeleton_pool


class FeynmanParser(TestSetParaser):
    def parse_data(self, test_set_df: pd.DataFrame, expression_space: ExpressionSpace, base_skeleton_pool: SkeletonPool, verbose: bool = False) -> SkeletonPool:
        '''
        Parse the test set data and import it into the skeleton pool.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        expression_space : ExpressionSpace
            The expression space to use for parsing and simplifying the expressions.
        base_skeleton_pool : SkeletonPool
            An initial skeleton pool to add the parsed expressions to.
        verbose : bool, optional
            Whether to print progress information, by default False

        Returns
        -------
        SkeletonPool
            The skeleton pool with the parsed expressions added.
        '''
        n_invalid_expressions = 0
        n_too_many_variables = 0

        expression_dict = {}
        for _, row in tqdm(test_set_df.iterrows(), disable=not verbose, desc='Parsing and Importing Feynman Data', total=len(test_set_df)):
            if row['# variables'] > len(expression_space.variables):
                n_too_many_variables += 1
                continue

            expression = str(row['Formula'])

            # Parse and simplify
            try:
                prefix_expression = expression_space.parse_expression(expression, mask_numbers=True, too_many_variables='raise')
            except ValueError:
                n_too_many_variables += 1
                warnings.warn(f'Expression {expression} has too many variables despite checking the dataset')
                continue

            # Check valid
            if not expression_space.is_valid(prefix_expression, verbose=True):
                continue

            prefix_expression = expression_space.simplify(prefix_expression)

            # Codify
            prefix_expression_w_num = expression_space.operators_to_realizations(prefix_expression)
            prefix_expression_w_constants, constants = num_to_constants(prefix_expression_w_num)
            code_string = expression_space.prefix_to_infix(prefix_expression_w_constants, realization=True)
            code = codify(code_string, expression_space.variables + constants)

            # Import
            expression_hash = tuple(prefix_expression)
            expression_dict[expression_hash] = (code, constants)

        print(f'Number of invalid expressions: {n_invalid_expressions} ({n_invalid_expressions / len(test_set_df) * 100:.2f}%)')
        print(f'Number of expressions with too many variables: {n_too_many_variables} ({n_too_many_variables / len(test_set_df) * 100:.2f}%)')

        base_skeleton_pool.skeleton_codes = expression_dict
        base_skeleton_pool.skeletons = list(expression_dict.keys())

        return base_skeleton_pool


class NguyenParser(TestSetParaser):
    def parse_data(self, test_set_df: pd.DataFrame, expression_space: ExpressionSpace, base_skeleton_pool: SkeletonPool, verbose: bool = False) -> SkeletonPool:
        '''
        Parse the test set data and import it into the skeleton pool.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        expression_space : ExpressionSpace
            The expression space to use for parsing and simplifying the expressions.
        base_skeleton_pool : SkeletonPool
            An initial skeleton pool to add the parsed expressions to.
        verbose : bool, optional
            Whether to print progress information, by default False

        Returns
        -------
        SkeletonPool
            The skeleton pool with the parsed expressions added.
        '''
        n_invalid_expressions = 0
        n_too_many_variables = 0

        expression_dict = {}

        for _, row in tqdm(test_set_df.iterrows(), disable=not verbose, desc='Parsing and Importing Nguyen Data', total=len(test_set_df)):
            expression = str(row['Equation'])

            # Parse and simplify
            try:
                prefix_expression = expression_space.parse_expression(expression, mask_numbers=True, too_many_variables='raise')
            except ValueError:
                n_too_many_variables += 1
                warnings.warn(f'Expression {expression} has too many variables despite checking the dataset')
                continue

            # Check valid
            if not expression_space.is_valid(prefix_expression, verbose=True):
                n_invalid_expressions += 1
                continue

            prefix_expression = expression_space.simplify(prefix_expression)

            # Codify
            prefix_expression_w_num = expression_space.operators_to_realizations(prefix_expression)
            prefix_expression_w_constants, constants = num_to_constants(prefix_expression_w_num)
            code_string = expression_space.prefix_to_infix(prefix_expression_w_constants, realization=True)
            code = codify(code_string, expression_space.variables + constants)

            # Import
            expression_hash = tuple(prefix_expression)
            expression_dict[expression_hash] = (code, constants)

        print(f'Number of invalid expressions: {n_invalid_expressions} ({n_invalid_expressions / len(test_set_df) * 100:.2f}%)')
        print(f'Number of expressions with too many variables: {n_too_many_variables} ({n_too_many_variables / len(test_set_df) * 100:.2f}%)')

        base_skeleton_pool.skeleton_codes = expression_dict
        base_skeleton_pool.skeletons = list(expression_dict.keys())

        return base_skeleton_pool

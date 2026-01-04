from abc import abstractmethod
import warnings

import pandas as pd

from tqdm import tqdm

from simplipy import SimpliPyEngine
from simplipy.utils import remap_expression

from flash_ansr import SkeletonPool  # Parse expressions with SimpliPyEngine.parse_infix_expression
from flash_ansr.expressions.compilation import codify
from flash_ansr.expressions.token_ops import identify_constants


class TestSetParaser:
    @abstractmethod
    def parse_data(self, test_set_df: pd.DataFrame, simplipy_engine: SimpliPyEngine, base_skeleton_pool: SkeletonPool, verbose: bool = False) -> SkeletonPool:
        '''
        Parse the test set data and import it into the skeleton pool.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        simplipy_engine : SimpliPyEngine
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
            case 'fastsrb':
                return FastSRBParser()
            case _:
                raise ValueError(f'Unknown parser: {parser_name}')


def is_number(token: str) -> bool:
    '''
    Check if a token is a number.

    Parameters
    ----------
    token : str
        The token to check.

    Returns
    -------
    bool
        True if the token is a number, False otherwise.
    '''
    try:
        float(token)
        return True
    except ValueError:
        return False


class SOOSEParser(TestSetParaser):
    def parse_data(self, test_set_df: pd.DataFrame, simplipy_engine: SimpliPyEngine, base_skeleton_pool: SkeletonPool, verbose: bool = False) -> SkeletonPool:
        '''
        Parse the test set data and import it into the skeleton pool.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        simplipy_engine : SimpliPyEngine
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
        for expression in tqdm(test_set_df['eq'], disable=not verbose, desc='Parsing and Importing SOOOSE Data', smoothing=0.0):
            # Parse and simplify
            prefix_expression = simplipy_engine.parse(expression, mask_numbers=True)

            # Check valid
            if not simplipy_engine.is_valid(prefix_expression, verbose=True):
                n_invalid_expressions += 1
                continue

            prefix_expression = simplipy_engine.simplify(prefix_expression, max_pattern_length=4)

            # Standardize variable names
            found_variables = [token for token in prefix_expression if token not in simplipy_engine.operators and not is_number(token) and token != '<constant>']
            prefix_expression, mapping = remap_expression(prefix_expression, found_variables, variable_mapping=None, variable_prefix="x", enumeration_offset=1)

            if len(mapping) > len(base_skeleton_pool.variables):
                n_too_many_variables += 1
                warnings.warn(f'\nExpression {expression} has too many variables for the skeleton pool. Expected at most {len(base_skeleton_pool.variables)} but got {len(mapping)} from mapping {mapping}')
                continue

            # Codify
            prefix_expression_w_num = simplipy_engine.operators_to_realizations(prefix_expression)
            prefix_expression_w_constants, constants = identify_constants(prefix_expression_w_num, inplace=True)
            code_string = simplipy_engine.prefix_to_infix(prefix_expression_w_constants, realization=True)
            code = codify(code_string, base_skeleton_pool.variables + constants)

            # Import
            expression_hash = tuple(prefix_expression)
            expression_dict[expression_hash] = (code, constants)

        print(f'Number of invalid expressions: {n_invalid_expressions} ({n_invalid_expressions / len(test_set_df) * 100:.2f}%)')
        print(f'Number of expressions with too many variables: {n_too_many_variables} ({n_too_many_variables / len(test_set_df) * 100:.2f}%)')

        base_skeleton_pool.skeleton_codes = expression_dict
        base_skeleton_pool.skeletons = list(expression_dict.keys())

        return base_skeleton_pool


class FeynmanParser(TestSetParaser):
    def parse_data(self, test_set_df: pd.DataFrame, simplipy_engine: SimpliPyEngine, base_skeleton_pool: SkeletonPool, verbose: bool = False) -> SkeletonPool:
        '''
        Parse the test set data and import it into the skeleton pool.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        simplipy_engine : SimpliPyEngine
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
        for _, row in tqdm(test_set_df.iterrows(), disable=not verbose, desc='Parsing and Importing Feynman Data', total=len(test_set_df), smoothing=0.0):
            if row['# variables'] > len(base_skeleton_pool.variables):
                n_too_many_variables += 1
                continue

            expression = str(row['Formula'])

            # Parse and simplify
            prefix_expression = simplipy_engine.parse(expression, mask_numbers=True)

            # Check valid
            if not simplipy_engine.is_valid(prefix_expression, verbose=True):
                continue

            prefix_expression = simplipy_engine.simplify(prefix_expression, max_pattern_length=4)

            # Standardize variable names
            found_variables = [token for token in prefix_expression if token not in simplipy_engine.operators and not is_number(token) and token != '<constant>']
            prefix_expression, mapping = remap_expression(prefix_expression, found_variables, variable_mapping=None, variable_prefix="x", enumeration_offset=1)

            if len(mapping) > len(base_skeleton_pool.variables):
                n_too_many_variables += 1
                warnings.warn(f'\nExpression {expression} has too many variables for the skeleton pool. Expected at most {len(base_skeleton_pool.variables)} but got {len(mapping)} from mapping {mapping}')
                continue

            # Codify
            prefix_expression_w_num = simplipy_engine.operators_to_realizations(prefix_expression)
            prefix_expression_w_constants, constants = identify_constants(prefix_expression_w_num, inplace=True)
            code_string = simplipy_engine.prefix_to_infix(prefix_expression_w_constants, realization=True)
            code = codify(code_string, base_skeleton_pool.variables + constants)

            # Import
            expression_hash = tuple(prefix_expression)
            expression_dict[expression_hash] = (code, constants)

        print(f'Number of invalid expressions: {n_invalid_expressions} ({n_invalid_expressions / len(test_set_df) * 100:.2f}%)')
        print(f'Number of expressions with too many variables: {n_too_many_variables} ({n_too_many_variables / len(test_set_df) * 100:.2f}%)')

        base_skeleton_pool.skeleton_codes = expression_dict
        base_skeleton_pool.skeletons = list(expression_dict.keys())

        return base_skeleton_pool


class NguyenParser(TestSetParaser):
    def parse_data(self, test_set_df: pd.DataFrame, simplipy_engine: SimpliPyEngine, base_skeleton_pool: SkeletonPool, verbose: bool = False) -> SkeletonPool:
        '''
        Parse the test set data and import it into the skeleton pool.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        simplipy_engine : SimpliPyEngine
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

        for _, row in tqdm(test_set_df.iterrows(), disable=not verbose, desc='Parsing and Importing Nguyen Data', total=len(test_set_df), smoothing=0.0):
            expression = str(row['Equation'])

            # Parse and simplify
            prefix_expression = simplipy_engine.parse(expression, mask_numbers=True)

            # Check valid
            if not simplipy_engine.is_valid(prefix_expression, verbose=True):
                n_invalid_expressions += 1
                continue

            prefix_expression = simplipy_engine.simplify(prefix_expression, max_pattern_length=4)

            # Standardize variable names
            found_variables = [token for token in prefix_expression if token not in simplipy_engine.operators and not is_number(token) and token != '<constant>']
            prefix_expression, mapping = remap_expression(prefix_expression, found_variables, variable_mapping=None, variable_prefix="x", enumeration_offset=1)

            if len(mapping) > len(base_skeleton_pool.variables):
                n_too_many_variables += 1
                warnings.warn(f'\nExpression {expression} has too many variables for the skeleton pool. Expected at most {len(base_skeleton_pool.variables)} but got {len(mapping)} from mapping {mapping}')
                continue

            # Codify
            prefix_expression_w_num = simplipy_engine.operators_to_realizations(prefix_expression)
            prefix_expression_w_constants, constants = identify_constants(prefix_expression_w_num, inplace=True)
            code_string = simplipy_engine.prefix_to_infix(prefix_expression_w_constants, realization=True)
            code = codify(code_string, base_skeleton_pool.variables + constants)

            # Import
            expression_hash = tuple(prefix_expression)
            expression_dict[expression_hash] = (code, constants)

        print(f'Number of invalid expressions: {n_invalid_expressions} ({n_invalid_expressions / len(test_set_df) * 100:.2f}%)')
        print(f'Number of expressions with too many variables: {n_too_many_variables} ({n_too_many_variables / len(test_set_df) * 100:.2f}%)')

        base_skeleton_pool.skeleton_codes = expression_dict
        base_skeleton_pool.skeletons = list(expression_dict.keys())

        return base_skeleton_pool


class FastSRBParser(TestSetParaser):
    def parse_data(self, test_set_df: pd.DataFrame, simplipy_engine: SimpliPyEngine, base_skeleton_pool: SkeletonPool, verbose: bool = False) -> SkeletonPool:
        '''
        Parse the FastSRB benchmark data and import it into the skeleton pool.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the prepared equations to evaluate.
        simplipy_engine : SimpliPyEngine
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
        n_missing_prepared = 0

        expression_dict: dict[tuple[str, ...], tuple] = {}

        for idx, row in tqdm(test_set_df.iterrows(), disable=not verbose, desc='Parsing and Importing FastSRB Data', total=len(test_set_df), smoothing=0.0):
            prepared_expression = row.get('prepared').replace('^', '**')

            if not isinstance(prepared_expression, str) or prepared_expression.strip() == '':
                n_missing_prepared += 1
                continue

            try:
                prefix_expression = simplipy_engine.parse(prepared_expression, mask_numbers=True)
            except (ValueError, TypeError):
                n_invalid_expressions += 1
                continue

            if not simplipy_engine.is_valid(prefix_expression, verbose=True):
                print(prepared_expression)
                n_invalid_expressions += 1
                continue

            prefix_expression = simplipy_engine.simplify(prefix_expression, max_pattern_length=4)

            found_variables = [token for token in prefix_expression if token not in simplipy_engine.operators and not is_number(token) and token != '<constant>']
            prefix_expression, mapping = remap_expression(prefix_expression, found_variables, variable_mapping=None, variable_prefix='x', enumeration_offset=1)

            if len(mapping) > len(base_skeleton_pool.variables):
                n_too_many_variables += 1
                warnings.warn(f"\nExpression at index {idx} has too many variables for the skeleton pool. Expected at most {len(base_skeleton_pool.variables)} but got {len(mapping)} from mapping {mapping}")
                continue

            prefix_expression_w_num = simplipy_engine.operators_to_realizations(prefix_expression)
            prefix_expression_w_constants, constants = identify_constants(prefix_expression_w_num, inplace=True)
            code_string = simplipy_engine.prefix_to_infix(prefix_expression_w_constants, realization=True)
            code = codify(code_string, base_skeleton_pool.variables + constants)

            expression_hash = tuple(prefix_expression)
            expression_dict[expression_hash] = (code, constants)

        denominator = len(test_set_df) if len(test_set_df) > 0 else 1
        print(f'Number of invalid expressions: {n_invalid_expressions} ({n_invalid_expressions / denominator * 100:.2f}%)')
        print(f'Number of expressions with too many variables: {n_too_many_variables} ({n_too_many_variables / denominator * 100:.2f}%)')
        print(f'Number of entries missing prepared expressions: {n_missing_prepared} ({n_missing_prepared / denominator * 100:.2f}%)')

        base_skeleton_pool.skeleton_codes = expression_dict
        base_skeleton_pool.skeletons = list(expression_dict.keys())

        return base_skeleton_pool

from abc import abstractmethod
import warnings

import pandas as pd

from tqdm import tqdm

from simplipy import SimpliPyEngine
from simplipy.utils import codify, explicit_constant_placeholders as identify_constants, remap_expression

from symbolic_data import LampleChartonCatalog  # Parse expressions with SimpliPyEngine.parse_infix_expression


class _InvalidExpression(Exception):
    """A parsed expression failed the engine's validity check (skip + count as invalid)."""


class _TooManyVariables(Exception):
    """A parsed expression uses more variables than the catalog supports (skip + count)."""

    def __init__(self, mapping: dict) -> None:
        super().__init__()
        self.mapping = mapping


class TestSetParser:
    @abstractmethod
    def parse_data(self, test_set_df: pd.DataFrame, simplipy_engine: SimpliPyEngine, base_catalog: LampleChartonCatalog, verbose: bool = False, *, skip_unparseable: bool = False) -> LampleChartonCatalog:
        '''
        Parse the test set data and import it into the catalog.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        simplipy_engine : SimpliPyEngine
            The expression space to use for parsing and simplifying the expressions.
        base_catalog : LampleChartonCatalog
            An initial catalog to add the parsed expressions to.
        verbose : bool, optional
            Whether to print progress information, by default False
        skip_unparseable : bool, optional
            How to handle a malformed (unparseable) expression. ``False`` (default) is fail-loud: the
            parse error propagates and aborts the import. ``True`` counts it in the invalid tally and
            skips it (the lenient mode for known-noisy inputs, e.g. an external benchmark file).

        Returns
        -------
        LampleChartonCatalog
            The catalog with the parsed expressions added.
        '''

    @staticmethod
    def _process_expression(expression: str, simplipy_engine: SimpliPyEngine, base_catalog: LampleChartonCatalog, *, skip_unparseable: bool = False) -> tuple[tuple[str, ...], tuple]:
        '''Shared per-expression pipeline: parse -> validate -> simplify -> canonicalize variables ->
        codify. Returns ``(expression_hash, (code, constants))``.

        A parse failure (malformed input) is **fail-loud by default**: the underlying
        ``ValueError`` / ``TypeError`` propagates and aborts the import, since malformed input in a
        curated test set signals a data problem worth surfacing. Pass ``skip_unparseable=True`` to
        instead count + skip it (raised as :class:`_InvalidExpression`). Engine-invalid (parsed but not
        representable) and too-many-variable expressions are *always* count + skip filters
        (:class:`_InvalidExpression` / :class:`_TooManyVariables`) -- they are the designed, reported
        attrition of importing an external set into a specific catalog vocabulary, not errors.'''
        try:
            prefix_expression = simplipy_engine.parse(expression, mask_numbers=True)
        except (ValueError, TypeError) as exc:
            if skip_unparseable:
                raise _InvalidExpression() from exc
            raise
        if not simplipy_engine.is_valid(prefix_expression, verbose=True):
            raise _InvalidExpression()
        prefix_expression = simplipy_engine.simplify(prefix_expression, max_pattern_length=4)

        found_variables = [token for token in prefix_expression if token not in simplipy_engine.operators and not is_number(token) and token != '<constant>']
        prefix_expression, mapping = remap_expression(prefix_expression, found_variables, variable_mapping=None, variable_prefix="x", enumeration_offset=1)
        if len(mapping) > len(base_catalog.variables):
            raise _TooManyVariables(mapping)

        prefix_expression_w_num = simplipy_engine.operators_to_realizations(prefix_expression)
        prefix_expression_w_constants, constants = identify_constants(prefix_expression_w_num, inplace=True)
        code_string = simplipy_engine.prefix_to_infix(prefix_expression_w_constants, realization=True)
        code = codify(code_string, base_catalog.variables + constants)
        return tuple(prefix_expression), (code, constants)

    @staticmethod
    def _finalize(base_catalog: LampleChartonCatalog, expression_dict: dict, n_invalid: int, n_too_many: int, total: int, extra_lines: 'tuple[tuple[str, int], ...]' = ()) -> LampleChartonCatalog:
        '''Shared reporting + import step: print the invalid / too-many-variable rates (plus any
        parser-specific ``extra_lines`` of ``(description, count)``) and attach the parsed skeletons
        to the catalog.'''
        print(f'Number of invalid expressions: {n_invalid} ({n_invalid / max(total, 1) * 100:.2f}%)')
        print(f'Number of expressions with too many variables: {n_too_many} ({n_too_many / max(total, 1) * 100:.2f}%)')
        for description, count in extra_lines:
            print(f'{description}: {count} ({count / max(total, 1) * 100:.2f}%)')
        base_catalog.skeleton_codes = expression_dict
        base_catalog.skeletons = list(expression_dict.keys())
        return base_catalog


class ParserFactory:
    '''Factory that maps a benchmark name to its :class:`TestSetParser` implementation.'''

    @staticmethod
    def get_parser(parser_name: str) -> TestSetParser:
        '''
        Return the parser instance for a named benchmark test set.

        Parameters
        ----------
        parser_name : str
            Benchmark identifier. One of ``'soose'``, ``'feynman'``, ``'nguyen'`` or ``'fastsrb'``.

        Returns
        -------
        TestSetParser
            A new parser instance for the requested benchmark.

        Raises
        ------
        ValueError
            If ``parser_name`` does not match a known benchmark.
        '''
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


class SOOSEParser(TestSetParser):
    def parse_data(self, test_set_df: pd.DataFrame, simplipy_engine: SimpliPyEngine, base_catalog: LampleChartonCatalog, verbose: bool = False, *, skip_unparseable: bool = False) -> LampleChartonCatalog:
        '''
        Parse the test set data and import it into the catalog.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        simplipy_engine : SimpliPyEngine
            The expression space to use for parsing and simplifying the expressions.
        base_catalog : LampleChartonCatalog
            An initial catalog to add the parsed expressions to.
        verbose : bool, optional
            Whether to print progress information, by default False

        Returns
        -------
        LampleChartonCatalog
            The catalog with the parsed expressions added.
        '''
        n_invalid_expressions = 0
        n_too_many_variables = 0

        expression_dict = {}
        for expression in tqdm(test_set_df['eq'], disable=not verbose, desc='Parsing and Importing SOOSE Data', smoothing=0.0):
            try:
                expression_hash, entry = self._process_expression(expression, simplipy_engine, base_catalog, skip_unparseable=skip_unparseable)
            except _InvalidExpression:
                n_invalid_expressions += 1
                continue
            except _TooManyVariables as exc:
                n_too_many_variables += 1
                warnings.warn(f'\nExpression {expression} has too many variables for the catalog. Expected at most {len(base_catalog.variables)} but got {len(exc.mapping)} from mapping {exc.mapping}')
                continue

            expression_dict[expression_hash] = entry

        return self._finalize(base_catalog, expression_dict, n_invalid_expressions, n_too_many_variables, len(test_set_df))


class FeynmanParser(TestSetParser):
    def parse_data(self, test_set_df: pd.DataFrame, simplipy_engine: SimpliPyEngine, base_catalog: LampleChartonCatalog, verbose: bool = False, *, skip_unparseable: bool = False) -> LampleChartonCatalog:
        '''
        Parse the test set data and import it into the catalog.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        simplipy_engine : SimpliPyEngine
            The expression space to use for parsing and simplifying the expressions.
        base_catalog : LampleChartonCatalog
            An initial catalog to add the parsed expressions to.
        verbose : bool, optional
            Whether to print progress information, by default False

        Returns
        -------
        LampleChartonCatalog
            The catalog with the parsed expressions added.
        '''
        n_invalid_expressions = 0
        n_too_many_variables = 0

        expression_dict = {}
        for _, row in tqdm(test_set_df.iterrows(), disable=not verbose, desc='Parsing and Importing Feynman Data', total=len(test_set_df), smoothing=0.0):
            # Cheap pre-guard: skip formulae whose declared variable count already exceeds the catalog,
            # before paying for a parse (counted as too-many, no warning since we never parsed it).
            if row['# variables'] > len(base_catalog.variables):
                n_too_many_variables += 1
                continue

            expression = str(row['Formula'])

            try:
                expression_hash, entry = self._process_expression(expression, simplipy_engine, base_catalog, skip_unparseable=skip_unparseable)
            except _InvalidExpression:
                n_invalid_expressions += 1
                continue
            except _TooManyVariables as exc:
                n_too_many_variables += 1
                warnings.warn(f'\nExpression {expression} has too many variables for the catalog. Expected at most {len(base_catalog.variables)} but got {len(exc.mapping)} from mapping {exc.mapping}')
                continue

            expression_dict[expression_hash] = entry

        return self._finalize(base_catalog, expression_dict, n_invalid_expressions, n_too_many_variables, len(test_set_df))


class NguyenParser(TestSetParser):
    def parse_data(self, test_set_df: pd.DataFrame, simplipy_engine: SimpliPyEngine, base_catalog: LampleChartonCatalog, verbose: bool = False, *, skip_unparseable: bool = False) -> LampleChartonCatalog:
        '''
        Parse the test set data and import it into the catalog.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the equations to evaluate.
        simplipy_engine : SimpliPyEngine
            The expression space to use for parsing and simplifying the expressions.
        base_catalog : LampleChartonCatalog
            An initial catalog to add the parsed expressions to.
        verbose : bool, optional
            Whether to print progress information, by default False

        Returns
        -------
        LampleChartonCatalog
            The catalog with the parsed expressions added.
        '''
        n_invalid_expressions = 0
        n_too_many_variables = 0

        expression_dict = {}

        for _, row in tqdm(test_set_df.iterrows(), disable=not verbose, desc='Parsing and Importing Nguyen Data', total=len(test_set_df), smoothing=0.0):
            expression = str(row['Equation'])

            try:
                expression_hash, entry = self._process_expression(expression, simplipy_engine, base_catalog, skip_unparseable=skip_unparseable)
            except _InvalidExpression:
                n_invalid_expressions += 1
                continue
            except _TooManyVariables as exc:
                n_too_many_variables += 1
                warnings.warn(f'\nExpression {expression} has too many variables for the catalog. Expected at most {len(base_catalog.variables)} but got {len(exc.mapping)} from mapping {exc.mapping}')
                continue

            expression_dict[expression_hash] = entry

        return self._finalize(base_catalog, expression_dict, n_invalid_expressions, n_too_many_variables, len(test_set_df))


class FastSRBParser(TestSetParser):
    def parse_data(self, test_set_df: pd.DataFrame, simplipy_engine: SimpliPyEngine, base_catalog: LampleChartonCatalog, verbose: bool = False, *, skip_unparseable: bool = False) -> LampleChartonCatalog:
        '''
        Parse the FastSRB benchmark data and import it into the catalog.

        Parameters
        ----------
        test_set_df : pd.DataFrame
            The test set data containing the prepared equations to evaluate.
        simplipy_engine : SimpliPyEngine
            The expression space to use for parsing and simplifying the expressions.
        base_catalog : LampleChartonCatalog
            An initial catalog to add the parsed expressions to.
        verbose : bool, optional
            Whether to print progress information, by default False

        Returns
        -------
        LampleChartonCatalog
            The catalog with the parsed expressions added.
        '''

        n_invalid_expressions = 0
        n_too_many_variables = 0
        n_missing_prepared = 0

        expression_dict: dict[tuple[str, ...], tuple] = {}

        for idx, row in tqdm(test_set_df.iterrows(), disable=not verbose, desc='Parsing and Importing FastSRB Data', total=len(test_set_df), smoothing=0.0):
            prepared_expression = row.get('prepared')

            if not isinstance(prepared_expression, str) or prepared_expression.strip() == '':
                n_missing_prepared += 1
                continue

            prepared_expression = prepared_expression.replace('^', '**')

            try:
                expression_hash, entry = self._process_expression(prepared_expression, simplipy_engine, base_catalog, skip_unparseable=skip_unparseable)
            except _InvalidExpression:
                n_invalid_expressions += 1
                continue
            except _TooManyVariables as exc:
                n_too_many_variables += 1
                warnings.warn(f"\nExpression at index {idx} has too many variables for the catalog. Expected at most {len(base_catalog.variables)} but got {len(exc.mapping)} from mapping {exc.mapping}")
                continue

            expression_dict[expression_hash] = entry

        return self._finalize(
            base_catalog, expression_dict, n_invalid_expressions, n_too_many_variables, len(test_set_df),
            extra_lines=(('Number of entries missing prepared expressions', n_missing_prepared),),
        )

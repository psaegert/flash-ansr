import importlib
from typing import Literal, Callable, Any
import warnings

import numpy as np
import torch
from scipy.optimize import curve_fit, minimize, OptimizeWarning, least_squares

from simplipy import SimpliPyEngine

from flash_ansr.expressions.compilation import codify
from flash_ansr.expressions.token_ops import identify_constants, apply_variable_mapping
from flash_ansr.utils.tensor_ops import pad_input_set


class ConvergenceError(Exception):
    pass


class Refiner:
    '''
    Refine the constants of an expression to fit the data

    Parameters
    ----------
    simplipy_engine : SimpliPyEngine
        The expression space to use for the refiner
    '''
    input_expression: list[str]
    executable_prefix_expression: list[str]
    prefix_expression_with_constants: list[str]
    constants_symbols: list[str]
    code_string: str
    expression_code: Callable
    expression_lambda: Callable
    constants_cov: np.ndarray | None

    def __init__(self, simplipy_engine: SimpliPyEngine, n_variables: int):
        '''
        Initialize the Refiner with the expression or skeleton to be refined
        '''
        self.simplipy_engine = simplipy_engine
        self.n_variables: int = n_variables

        self.import_modules()

        self.loss = np.inf
        self.valid_fit: bool = False

        self._all_constants_values: list[tuple[np.ndarray, np.ndarray, float]] = []

    def import_modules(self) -> None:
        '''
        Import the modules required for the expression
        '''
        # TODO: Check if this is necessary
        for module in self.simplipy_engine.modules:
            if module not in globals():
                globals()[module] = importlib.import_module(module)

    def _initialize_expression(self, expression: list[str], n_inputs: int) -> None:
        '''
        Prepare internal state for the given expression without fitting constants.
        '''
        if not self.simplipy_engine.is_valid(expression, verbose=True):
            raise ValueError("The expression is not valid")

        self.input_expression = expression
        self.executable_prefix_expression = self.simplipy_engine.operators_to_realizations(self.input_expression)
        self.prefix_expression_with_constants, self.constants_symbols = identify_constants(self.input_expression)
        self.code_string = self.simplipy_engine.prefix_to_infix(self.prefix_expression_with_constants, realization=True)

        self.expression_code = codify(
            code_string=self.code_string,
            variables=[f'x{i + 1}' for i in range(n_inputs)] + self.constants_symbols
        )

        # Since the SimpliPyEngine is already initialized, we can use the same global scope
        self.expression_lambda = self.simplipy_engine.code_to_lambda(self.expression_code)

    def _assign_fits(self, fits: list[tuple[np.ndarray, np.ndarray | None, float]]) -> None:
        '''
        Consume serialized fit results and update the refiner state.
        '''
        expected_constants = len(self.constants_symbols)

        filtered: list[tuple[np.ndarray, np.ndarray, float]] = []
        for constants, cov, loss in fits:
            constants_array = np.asarray(constants, dtype=float)

            if expected_constants != len(constants_array):
                continue

            cov_array = np.asarray(cov) if cov is not None else np.asarray([], dtype=float)
            filtered.append((constants_array, cov_array, float(loss)))

        self._all_constants_values = sorted(filtered, key=lambda item: item[-1])
        self.valid_fit = any(np.isfinite(entry[-1]) for entry in self._all_constants_values)
        if self._all_constants_values:
            self.loss = float(self._all_constants_values[0][-1])
        else:
            self.loss = np.inf

    @classmethod
    def from_serialized(
            cls,
            simplipy_engine: SimpliPyEngine,
            n_variables: int,
            expression: list[str],
            n_inputs: int,
            fits: list[tuple[np.ndarray, np.ndarray | None, float]]) -> 'Refiner':
        '''
        Reconstruct a Refiner from cached fit results.
        '''
        refiner = cls(simplipy_engine=simplipy_engine, n_variables=n_variables)
        refiner._initialize_expression(expression, n_inputs)
        refiner._assign_fits(fits)
        return refiner

    def fit(
            self,
            expression: list[str],
            X: np.ndarray,
            y: np.ndarray,
            p0: np.ndarray | None = None,
            p0_noise: Literal['uniform', 'normal'] | None = 'normal',
            p0_noise_kwargs: dict | None = None,
            n_restarts: int = 1,
            method: Literal[
                'curve_fit_lm',
                'minimize_bfgs',
                'minimize_lbfgsb',
                'minimize_neldermead',
                'minimize_powell',
                'least_squares_trf',
                'least_squares_dogbox',
            ] = 'curve_fit_lm',
            no_constants_error: Literal['raise', 'ignore'] = 'ignore',
            optimizer_kwargs: dict | None = None,
            converge_error: Literal['raise', 'ignore'] = 'ignore') -> 'Refiner':
        '''
        Fit the constants of the expression to the data

        Parameters
        ----------
        expression : list
            The expression to fit in prefix notation
        X : np.ndarray
            The input data
        y : np.ndarray
            The output data
        p0 : np.ndarray, optional
            Initial guess for the constants
        p0_noise : str, optional
            Add noise to the initial guess. One of
            - 'uniform': Uniform noise
            - 'normal': Normal noise
        p0_noise_kwargs : dict, optional
            Keyword arguments for the noise generation
        n_restarts : int, optional
            Number of restarts for the optimization
        method : str, optional
            The optimization method to use. One of
            - 'curve_fit_lm': Use the curve_fit method with the Levenberg-Marquardt algorithm
            - 'minimize_bfgs': Use the minimize method with the BFGS algorithm
            - 'minimize_lbfgsb': Use the minimize method with the L-BFGS-B algorithm
            - 'minimize_neldermead': Use the minimize method with the Nelder-Mead algorithm
            - 'minimize_powell': Use the minimize method with the Powell algorithm
            - 'least_squares_trf': Use the least_squares solver with the trust region reflective algorithm
            - 'least_squares_dogbox': Use the least_squares solver with the dogbox algorithm
        no_constants_error : str, optional
            What to do if the expression does not contain any constants. One of
            - 'raise': Raise an error
            - 'ignore': Ignore the error
        optimizer_kwargs : dict, optional
            Keyword arguments for the optimizer
        converge_error : str, optional
            What to do if the optimization does not converge. One of
            - 'raise': Raise an error
            - 'ignore': Ignore the error

        Returns
        -------
        Refiner
            The refiner object
        '''
        self._initialize_expression(expression, X.shape[1])

        def pred_function(X: np.ndarray, *constants: np.ndarray | None) -> np.ndarray:
            if len(constants) == 0:
                y_pred = self.expression_lambda(*X.T)
            else:
                y_pred = self.expression_lambda(*X.T, *constants)

            n_samples = X.shape[0]

            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            if isinstance(y_pred, np.ndarray):
                if y_pred.ndim == 0:
                    y_pred = np.full((n_samples,), float(y_pred))
                else:
                    y_pred = y_pred.reshape(-1)
                    if y_pred.size == 1 and n_samples > 1:
                        y_pred = np.full((n_samples,), float(y_pred[0]))
                    elif y_pred.size != n_samples:
                        y_pred = np.resize(y_pred, n_samples)
            else:
                y_pred = np.full((n_samples,), y_pred)

            return np.asarray(y_pred, dtype=float)

        # Forget all previous results
        self._all_constants_values = []
        constants_values = None
        constants_cov = None

        if len(self.constants_symbols) == 0:
            if no_constants_error == 'raise':
                raise ValueError("The expression does not contain any constants")

            constants_values = np.array([])
            constants_cov = np.array([])
            try:
                diff = pred_function(X) - y[:, 0]
                if np.isnan(diff).any():
                    self.loss = np.nan
                else:
                    self.loss = np.mean(diff ** 2)  # type: ignore
            except OverflowError:
                self.loss = np.nan

            self._all_constants_values.append((constants_values, constants_cov, self.loss))  # type: ignore

            return self

        self._all_constants_values = []
        self.valid_fit = False

        for _ in range(n_restarts):
            try:
                constants, constants_cov = self._fit(pred_function, X, y, p0, p0_noise, p0_noise_kwargs, method, no_constants_error, optimizer_kwargs)
            except (ConvergenceError, OverflowError):
                self._all_constants_values.append((np.array([]), np.array([]), np.nan))
                continue

            try:
                diff = pred_function(X, *constants) - y[:, 0]
                if np.isnan(diff).any():
                    loss = np.nan
                else:
                    loss = np.mean(diff ** 2)  # type: ignore
            except (OverflowError, TypeError):
                loss = np.nan

            self._all_constants_values.append((constants, constants_cov, loss))

        expected_constants = len(self.constants_symbols)
        filtered_constants: list[tuple[np.ndarray, np.ndarray | None, float]] = []
        for constants, constants_cov, loss in self._all_constants_values:
            if len(constants) == expected_constants:
                filtered_constants.append((constants, constants_cov, loss))

        self._assign_fits(filtered_constants)

        if not self.valid_fit and converge_error == 'raise':
            raise ConvergenceError(f"The optimization did not converge after {n_restarts} restarts")

        return self

    def _fit(
            self,
            pred_function: Callable,
            X: np.ndarray,
            y: np.ndarray,
            p0: np.ndarray | None = None,
            p0_noise: Literal['uniform', 'normal'] | None = 'normal',
            p0_noise_kwargs: dict | None = None,
            method: Literal[
                'curve_fit_lm',
                'minimize_bfgs',
                'minimize_lbfgsb',
                'minimize_neldermead',
                'minimize_powell',
                'least_squares_trf',
                'least_squares_dogbox',
            ] = 'curve_fit_lm',
            no_constants_error: Literal['raise', 'ignore'] = 'ignore',
            optimizer_kwargs: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Fit the constants of the expression to the data

        Parameters
        ----------
        pred_function : Callable
            Function that predicts y from X and the constants
        X : np.ndarray
            The input data
        y : np.ndarray
            The output data
        p0 : np.ndarray, optional
            Initial guess for the constants
        p0_noise : str, optional
            Add noise to the initial guess. One of
            - 'uniform': Uniform noise
            - 'normal': Normal noise
        p0_noise_kwargs : dict, optional
            Keyword arguments for the noise generation
        method : str, optional
            The optimization method to use. One of
            - 'curve_fit_lm': Use the curve_fit method with the Levenberg-Marquardt algorithm
            - 'minimize_bfgs': Use the minimize method with the BFGS algorithm
            - 'minimize_lbfgsb': Use the minimize method with the L-BFGS-B algorithm
            - 'minimize_neldermead': Use the minimize method with the Nelder-Mead algorithm
            - 'minimize_powell': Use the minimize method with the Powell algorithm
            - 'least_squares_trf': Use the least_squares solver with the trust region reflective algorithm
            - 'least_squares_dogbox': Use the least_squares solver with the dogbox algorithm
        no_constants_error : str, optional
            What to do if the expression does not contain any constants. One of
            - 'raise': Raise an error
            - 'ignore': Ignore the error
        optimizer_kwargs : dict, optional
            Keyword arguments for the optimizer
        '''

        if len(self.constants_symbols) == 0:
            if no_constants_error == 'raise':
                raise ValueError("The expression does not contain any constants")
            return np.array([]), np.array([])

        p0_noise_kwargs = p0_noise_kwargs or {}
        optimizer_kwargs = optimizer_kwargs or {}

        if p0 is None:
            p0 = np.zeros(len(self.constants_symbols))

        # Initial guess for the constants
        if isinstance(p0_noise, str):
            match p0_noise:
                case 'uniform':
                    match p0_noise_kwargs.get('type', 'add'):
                        case 'add':
                            p0 += np.random.uniform(size=len(self.constants_symbols), low=p0_noise_kwargs.get('low', 0), high=p0_noise_kwargs.get('high', 1))
                        case 'multiply':
                            p0 *= np.random.uniform(size=len(self.constants_symbols), low=p0_noise_kwargs.get('low', 0), high=p0_noise_kwargs.get('high', 1))
                        case _:
                            raise ValueError(f"Invalid option for p0_noise: Expected one of 'add', 'multiply', got {p0_noise_kwargs.get('type', 'add')}")
                case 'normal':
                    match p0_noise_kwargs.get('type', 'add'):
                        case 'add':
                            p0 += np.random.normal(size=len(self.constants_symbols), loc=p0_noise_kwargs.get('loc', 0), scale=p0_noise_kwargs.get('scale', 1))
                        case 'multiply':
                            p0 *= np.random.normal(size=len(self.constants_symbols), loc=p0_noise_kwargs.get('loc', 0), scale=p0_noise_kwargs.get('scale', 1))
                        case _:
                            raise ValueError(f"Invalid option for p0_noise: Expected one of 'add', 'multiply', got {p0_noise_kwargs.get('type', 'add')}")
                case None:
                    pass
                case _:
                    raise ValueError(f"Invalid option for p0: Expected one of 'uniform', 'normal' or None, got {p0}")

        # Minimize the objective function
        try:
            valid_mask = np.all(np.isfinite(y), axis=-1)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            # Ignore OptimizeWarning warnings
            warnings.filterwarnings("ignore", category=OptimizeWarning)

            def objective(p: np.ndarray) -> float:
                return np.mean((pred_function(X_valid, *p) - y_valid.flatten()) ** 2)

            def residuals(p: np.ndarray) -> np.ndarray:
                return pred_function(X_valid, *p) - y_valid.flatten()

            def _safe_matrix(mat: Any) -> np.ndarray:
                if mat is None:
                    return np.asarray([])
                if hasattr(mat, 'todense'):
                    return np.asarray(mat.todense())
                try:
                    return np.asarray(mat)
                except Exception:
                    return np.asarray([])

            match method:
                case 'curve_fit_lm':
                    popt, pcov = curve_fit(pred_function, X_valid, y_valid.flatten(), p0, **optimizer_kwargs)
                case 'minimize_bfgs':
                    res = minimize(objective, p0, method='BFGS', **optimizer_kwargs)
                    popt = res.x
                    pcov = _safe_matrix(getattr(res, 'hess_inv', None))
                case 'minimize_lbfgsb':
                    res = minimize(objective, p0, method='L-BFGS-B', **optimizer_kwargs)
                    popt = res.x
                    pcov = _safe_matrix(getattr(res, 'hess_inv', None))
                case 'minimize_neldermead':
                    res = minimize(objective, p0, method='Nelder-Mead', **optimizer_kwargs)
                    popt = res.x
                    pcov = np.asarray([])
                case 'minimize_powell':
                    res = minimize(objective, p0, method='Powell', **optimizer_kwargs)
                    popt = res.x
                    pcov = np.asarray([])
                case 'least_squares_trf':
                    res = least_squares(residuals, p0, method='trf', **optimizer_kwargs)
                    popt = res.x
                    pcov = _safe_matrix(getattr(res, 'jac', None))
                case 'least_squares_dogbox':
                    res = least_squares(residuals, p0, method='dogbox', **optimizer_kwargs)
                    popt = res.x
                    pcov = _safe_matrix(getattr(res, 'jac', None))
                case _:
                    raise ValueError(f"Unknown optimization method: {method}")

        except (RuntimeError, TypeError, ValueError) as exc:
            raise ConvergenceError("The optimization did not converge") from exc

        return popt, pcov

    def predict(self, X: np.ndarray | torch.Tensor, nth_best_constants: int = 0) -> np.ndarray:
        '''
        Predict the output of the expression with the fitted constants

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            The input data
        nth_best_constants : int, optional
            An index specifying which fitted constants to use. By default 0 (the best constants)

        Returns
        -------
        np.ndarray
            The predicted output
        '''
        constants_values = self._all_constants_values[nth_best_constants][0]

        if len(constants_values) != len(self.constants_symbols):
            return np.full((X.shape[0], 1), np.nan)

        X = pad_input_set(X, self.n_variables)

        if len(self.constants_symbols) == 0 or len(constants_values) == 0:
            y = self.expression_lambda(*X.T)
        else:
            y = self.expression_lambda(*X.T, *constants_values)  # type: ignore

        if not isinstance(y, (np.ndarray, torch.Tensor)):
            if isinstance(X, torch.Tensor):
                y = torch.full((X.shape[0], 1), y)
            else:
                y = np.full((X.shape[0], 1), y)

        if len(y) == 1:
            # Repeat y to match the shape of x
            if isinstance(X, torch.Tensor):
                y = torch.repeat_interleave(y, X.shape[0], dim=0)
            else:
                y = np.repeat(y, X.shape[0])

        return y.reshape(-1, 1)

    def transform(self, expression: list[str], nth_best_constants: int = 0, return_prefix: bool = False, precision: int = 2, variable_mapping: dict | None = None, **kwargs: Any) -> list[str] | str:
        '''
        Insert the fitted constants to the expression

        Parameters
        ----------
        expression : list
            The expression to transform
        nth_best_constants : int, optional
            An index specifying which fitted constants to use. By default 0 (the best constants)
        return_prefix : bool, optional
            Whether to return the expression in prefix notation. By default False
        precision : int, optional
            The precision for rounding the constants. By default 2
        variable_mapping : dict, optional
            A dictionary mapping the variables to their names. By default None (use the default variable names)

        Returns
        -------
        list[str] or str
            The transformed expression
        '''
        constants_values = np.asarray(self._all_constants_values[nth_best_constants][0], dtype=float)

        expression_tokens = list(expression)
        if constants_values.size:
            rounded_constants = np.round(constants_values, precision)
            constant_iter = iter(rounded_constants.tolist())

            for idx, token in enumerate(expression_tokens):
                is_constant_token = (
                    token == "<constant>"
                    or token.startswith("C_")
                    or token in self.constants_symbols
                )

                if not is_constant_token:
                    continue

                try:
                    value = next(constant_iter)
                except StopIteration:
                    break

                expression_tokens[idx] = str(value)

        expression_with_values = expression_tokens

        if variable_mapping is not None:
            expression_with_values = apply_variable_mapping(expression_with_values, variable_mapping)

        if return_prefix:
            return expression_with_values

        expression_with_values_infix = self.simplipy_engine.prefix_to_infix(expression_with_values, **kwargs)
        return expression_with_values_infix

    def __str__(self) -> str:
        '''
        Return the string representation of the Refiner

        Returns
        -------
        str
            The string representation
        '''
        return f"Refiner(expression={self.input_expression}, best_constants={self._all_constants_values[0][0]}, best_loss={self._all_constants_values[0][2]})"

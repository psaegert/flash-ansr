import importlib
from typing import Literal, Callable, Any
import warnings

import numpy as np
import torch
from scipy.optimize import curve_fit, minimize, OptimizeWarning

from flash_ansr.expressions import ExpressionSpace
from flash_ansr.expressions.utils import codify, substitude_constants, num_to_constants, apply_variable_mapping


class ConvergenceError(Exception):
    pass


class Refiner:
    '''
    Refine the constants of an expression to fit the data

    Parameters
    ----------
    expression_space : ExpressionSpace
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

    def __init__(self, expression_space: ExpressionSpace):
        '''
        Initialize the Refiner with the expression or skeleton to be refined
        '''
        self.expression_space = expression_space

        self.import_modules()

        self.constants_values: np.ndarray | None = None
        self.loss: float | None = None

        self._all_constants_values: list[tuple[np.ndarray, np.ndarray, float]] = []

    def import_modules(self) -> None:
        '''
        Import the modules required for the expression
        '''
        # TODO: Check if this is necessary
        for module in self.expression_space.modules:
            if module not in globals():
                globals()[module] = importlib.import_module(module)

    def fit(
            self,
            expression: list[str],
            X: np.ndarray,
            y: np.ndarray,
            p0: np.ndarray | None = None,
            p0_noise: Literal['uniform', 'normal'] | None = 'normal',
            p0_noise_kwargs: dict | None = None,
            n_restarts: int = 1,
            method: Literal['curve_fit_lm', 'minimize_bfgs'] = 'curve_fit_lm',
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
        if not self.expression_space.is_valid(expression, verbose=True):
            raise ValueError("The expression is not valid")

        self.input_expression = expression
        self.executable_prefix_expression = self.expression_space.operators_to_realizations(self.input_expression)
        self.prefix_expression_with_constants, self.constants_symbols = num_to_constants(self.input_expression)
        self.code_string = self.expression_space.prefix_to_infix(self.prefix_expression_with_constants, realization=True)

        self.expression_code = codify(
            code_string=self.code_string,
            variables=self.expression_space.variables + self.constants_symbols
        )

        # Since the ExpressionSpace is already initialized, we can use the same global scope
        self.expression_lambda = self.expression_space.code_to_lambda(self.expression_code)

        def pred_function(X: np.ndarray, *constants: np.ndarray | None) -> float:
            if len(constants) == 0:
                y_pred = self.expression_lambda(*X.T)
            else:
                y_pred = self.expression_lambda(*X.T, *constants)

            return y_pred.flatten()

        # Forget all previous results
        self._all_constants_values = []
        self.constants_values = None
        self.constants_cov = None
        self.loss = None

        if len(self.constants_symbols) == 0:
            if no_constants_error == 'raise':
                raise ValueError("The expression does not contain any constants")

            self.constants_values = np.array([])
            self.constants_cov = np.array([])
            try:
                diff = pred_function(X) - y[:, 0]
                if np.isnan(diff).any():
                    self.loss = np.nan
                else:
                    self.loss = np.mean(diff ** 2)
            except OverflowError:
                self.loss = np.inf

            self._all_constants_values.append((self.constants_values, self.constants_cov, self.loss))  # type: ignore

            return self

        best_constants: np.ndarray | None = None
        best_constants_cov: np.ndarray | None = None
        best_loss = np.inf

        self._all_constants_values = []

        for _ in range(n_restarts):
            try:
                constants, constants_cov = self._fit(pred_function, X, y, p0, p0_noise, p0_noise_kwargs, method, no_constants_error, optimizer_kwargs)
            except (ConvergenceError, OverflowError):
                continue

            try:
                diff = pred_function(X, *constants) - y[:, 0]
                if np.isnan(diff).any():
                    loss = np.nan
                else:
                    loss = np.mean(diff ** 2)
            except OverflowError:
                loss = np.inf

            self._all_constants_values.append((constants, constants_cov, loss))

            if loss < best_loss:
                best_constants = constants
                best_constants_cov = constants_cov
                best_loss = loss

        # Sort the constants by loss
        self._all_constants_values = sorted(self._all_constants_values, key=lambda x: x[-1])

        if best_constants is None and converge_error == 'raise':
            raise ConvergenceError(f"The optimization did not converge after {n_restarts} restarts")

        self.constants_values = best_constants
        self.constants_cov = best_constants_cov  # type: ignore
        self.loss = best_loss

        return self

    def _fit(
            self,
            pred_function: Callable,
            X: np.ndarray,
            y: np.ndarray,
            p0: np.ndarray | None = None,
            p0_noise: Literal['uniform', 'normal'] | None = 'normal',
            p0_noise_kwargs: dict | None = None,
            method: Literal['curve_fit_lm', 'minimize_bfgs'] = 'curve_fit_lm',
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
            # Ignore OptimizeWarning warnings
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            match method:
                case 'curve_fit_lm':
                    popt, pcov = curve_fit(pred_function, X, y.flatten(), p0, **optimizer_kwargs)
                case 'minimize_bfgs':
                    def objective(p: np.ndarray) -> float:
                        return np.mean((pred_function(X, *p) - y.flatten()) ** 2)

                    res = minimize(objective, p0, method='BFGS', **optimizer_kwargs)
                    popt = res.x
                    pcov = res.hess_inv  # TODO: Check if this is correct

        except (RuntimeError, TypeError):
            raise ConvergenceError("The optimization did not converge")

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
        if self.constants_values is None and len(self.constants_symbols) > 0:
            raise ValueError("The constants have not been fitted yet. Please call the fit method first")

        constants_values = self._all_constants_values[nth_best_constants][0]

        if len(self.constants_symbols) == 0:
            y = self.expression_lambda(*X.T)
        else:
            y = self.expression_lambda(*X.T, *constants_values)  # type: ignore

        if not isinstance(y, (np.ndarray, torch.Tensor)):
            y = np.full((X.shape[0], 1), y.reshape(-1, 1))

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
        if self.constants_values is None and len(self.constants_symbols) > 0:
            raise ValueError("The constants have not been fitted yet. Please call the fit method first")

        constants_values = self._all_constants_values[nth_best_constants][0]

        expression_with_values = substitude_constants(expression, np.round(constants_values, precision))

        if variable_mapping is not None:
            expression_with_values = apply_variable_mapping(expression_with_values, variable_mapping)

        if return_prefix:
            return expression_with_values

        expression_with_values_infix = self.expression_space.prefix_to_infix(expression_with_values, **kwargs)
        return expression_with_values_infix

    def __str__(self) -> str:
        '''
        Return the string representation of the Refiner

        Returns
        -------
        str
            The string representation
        '''
        return f"Refiner(expression={self.input_expression}, best_constants={self.constants_values}, best_loss={self.loss})"

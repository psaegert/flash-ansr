import os
import warnings
import random
import pickle
from copy import deepcopy

from types import CodeType
from typing import Any, Callable, Sequence

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

from simplipy import SimpliPyEngine

from flash_ansr.utils.config_io import load_config, save_config
from flash_ansr.utils.paths import substitute_root_path
from flash_ansr.expressions.compilation import codify
from flash_ansr.expressions.prior_factory import build_prior_callable
from flash_ansr.expressions.holdout import HoldoutManager
from flash_ansr.expressions.skeleton_sampling import SkeletonSampler
from flash_ansr.expressions.support_sampling import SupportSampler, SupportSamplingError
from flash_ansr.expressions.token_ops import identify_constants, flatten_nested_list


class NoValidSampleFoundError(Exception):
    pass


class SkeletonPool:
    '''
    Manage and sample from a set of expression 'skeletons' (prefix expressions with placeholders for constants).

    Parameters
    ----------
    simplipy_engine : SimpliPyEngine
        The expression space to operate in.
    sample_strategy : dict[str, Any]
        The strategy to use for sampling skeletons.
    literal_prior : dict[str, Any] | list[dict[str, Any]]
        The prior distribution for the literals.
    support_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
        Legacy override for the support prior. When provided, it will overwrite the
        corresponding entry in ``support_sampler_config``.
    support_scale_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
        Legacy override for the support-scale prior. Merged into ``support_sampler_config``
        when supplied.
    n_support_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
        Legacy override for the support-count prior. Merged into ``support_sampler_config``
        when supplied.
    support_sampler_config : dict[str, Any] or None, optional
        Unified configuration describing how support points are generated (priors,
        uniqueness constraints, quantized behaviour, etc.).
    holdout_pools : Sequence[SkeletonPool | str] or None, optional
        SkeletonPools or paths to pools to exclude when sampling.
    allow_nan : bool, optional
        Whether to allow NaNs in the support points.
    simplify : bool, optional
        Whether to simplify sampled skeletons.
    '''

    def __init__(
            self,
            simplipy_engine: SimpliPyEngine,
            sample_strategy: dict[str, Any],
            literal_prior: dict[str, Any] | list[dict[str, Any]] | Callable,
            variables: list[str],
            support_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            support_scale_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            n_support_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            support_sampler_config: dict[str, Any] | None = None,
            operator_weights: dict[str, float] | None = None,
            holdout_pools: Sequence["SkeletonPool | str"] | None = None,
            allow_nan: bool = False,
            simplify: bool = True) -> None:
        self.simplipy_engine = simplipy_engine
        self.sample_strategy = sample_strategy
        self.variables = variables
        self.n_variables = len(self.variables)
        self.operator_weights = operator_weights or {op: 1.0 for op in self.simplipy_engine.operator_arity.keys()}

        self.holdout_manager = HoldoutManager(n_variables=self.n_variables, allow_nan=allow_nan)

        self.holdout_pools: list["SkeletonPool | str"] = []
        for holdout_pool in holdout_pools or []:
            self.register_holdout_pool(holdout_pool)

        self.skeletons: set[tuple[str]] = set()
        self.skeleton_codes: dict[tuple[str], tuple[CodeType, list[str]]] = {}

        self.skeleton_sampler = SkeletonSampler(
            simplipy_engine=self.simplipy_engine,
            sample_strategy=self.sample_strategy,
            variables=self.variables,
            operator_weights=self.operator_weights,
        )

        if isinstance(literal_prior, (dict, list)):
            self.literal_prior_config = literal_prior
            self.literal_prior: Callable = build_prior_callable(literal_prior)
        elif callable(literal_prior):
            self.literal_prior = literal_prior
        else:
            raise ValueError("literal_prior must be either a dict, list of dicts, or a callable")

        support_config = deepcopy(support_sampler_config) if support_sampler_config is not None else {}

        if support_prior is not None:
            support_config["support_prior"] = support_prior
        if support_scale_prior is not None:
            support_config["support_scale_prior"] = support_scale_prior
        if n_support_prior is not None:
            support_config["n_support_prior"] = n_support_prior

        self.support_sampler_config = support_config

        self.allow_nan = allow_nan
        self.simplify = simplify

        independent_dims = self.sample_strategy.get('independent_dimensions', False)
        self.support_sampler = SupportSampler(
            n_variables=self.n_variables,
            independent_dimensions=independent_dims,
            config=self.support_sampler_config,
        )

        self.operator_probs: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "SkeletonPool":
        '''
        Create a SkeletonPool from a configuration dictionary or file.

        Parameters
        ----------
        config : dict or str
            The configuration dictionary or path to the configuration file.

        Returns
        -------
        SkeletonPool
            The SkeletonPool object.
        '''
        config_ = load_config(config)

        if "skeleton_pool" in config_.keys():
            config_ = config_["skeleton_pool"]

        # If the config is a string, convert relative paths within the config to absolute paths
        if isinstance(config, str) and isinstance(config_["simplipy_engine"], str):
            if config_["simplipy_engine"].startswith('.'):
                config_["simplipy_engine"] = os.path.join(os.path.dirname(config), config_["simplipy_engine"])

        support_sampler_cfg = deepcopy(config_.get("support_sampler")) if config_.get("support_sampler") else {}
        for key in ("support_prior", "support_scale_prior", "n_support_prior"):
            if key in config_ and key not in support_sampler_cfg:
                support_sampler_cfg[key] = config_[key]

        return cls(
            simplipy_engine=SimpliPyEngine.load(config_["simplipy_engine"], install=True),
            sample_strategy=config_["sample_strategy"],
            literal_prior=config_["literal_prior"],
            variables=config_["variables"],
            support_sampler_config=support_sampler_cfg,
            operator_weights=config_.get("operator_weights"),
            holdout_pools=config_["holdout_pools"],
            allow_nan=config_["allow_nan"],
            simplify=config_.get("simplify", True)
        )

    @classmethod
    def from_dict(
            cls,
            skeletons: set[tuple[str]],
            simplipy_engine: SimpliPyEngine,
            sample_strategy: dict[str, Any],
            literal_prior: dict[str, Any] | list[dict[str, Any]] | Callable,
            variables: list[str],
            support_sampler_config: dict[str, Any] | None = None,
            support_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            support_scale_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            n_support_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            operator_weights: dict[str, float] | None = None,
            skeleton_codes: dict[tuple[str], tuple[CodeType, list[str]]] | None = None,
            holdout_pools: Sequence["SkeletonPool | str"] | None = None,
            allow_nan: bool = False,
            simplify: bool = True) -> "SkeletonPool":
        '''
        Create a SkeletonPool from a set of skeletons.

        Parameters
        ----------
        skeletons : set[tuple[str]]
            The set of skeletons to include in the pool.
        simplipy_engine : SimpliPyEngine
            The expression space to operate in.
        sample_strategy : dict[str, Any]
            The strategy to use for sampling skeletons.
        literal_prior : dict[str, Any] | list[dict[str, Any]]
            The prior distribution for the literals.
        variables : list[str]
            The variables to use in the expressions.
        support_sampler_config : dict[str, Any] or None, optional
            Unified support sampling configuration. If provided alongside the legacy
            priors below, the legacy values overwrite matching keys within this config.
        support_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
            Optional override for the support prior.
        support_scale_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
            Optional override for the support-scale prior.
        n_support_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
            Optional override for the support-count prior.
        operator_weights : dict[str, float] or None, optional
            A dictionary mapping operators to their weights.
        skeleton_codes : dict[tuple[str], tuple[CodeType, list[str]]] or None, optional
            A dictionary mapping skeletons to their compiled codes.
        holdout_pools : Sequence[SkeletonPool | str] or None, optional
            SkeletonPools or paths to pools to exclude when sampling.
        allow_nan : bool, optional
            Whether to allow NaNs in the support points.
        simplify : bool, optional
            Whether to simplify sampled skeletons.

        Returns
        -------
        SkeletonPool
            The SkeletonPool object.
        '''
        skeleton_pool = cls(
            simplipy_engine=simplipy_engine,
            sample_strategy=sample_strategy,
            literal_prior=literal_prior,
            variables=variables,
            support_prior=support_prior,
            support_scale_prior=support_scale_prior,
            n_support_prior=n_support_prior,
            support_sampler_config=support_sampler_config,
            operator_weights=operator_weights,
            holdout_pools=holdout_pools,
            allow_nan=allow_nan,
            simplify=simplify
        )

        skeleton_pool.skeletons = skeletons

        if skeleton_codes is not None:
            skeleton_pool.skeleton_codes = skeleton_codes
        else:
            skeleton_pool.skeleton_codes = skeleton_pool.compile_codes()

        return skeleton_pool

    def compile_codes(self, verbose: bool = False) -> dict[tuple[str], tuple[CodeType, list[str]]]:
        '''
        Compile the skeletons in the pool into executable code.

        Parameters
        ----------
        verbose : bool, optional
            Whether to display a progress bar.

        Returns
        -------
        dict[tuple[str], tuple[CodeType, list[str]]]
            A dictionary mapping skeletons to their compiled codes.
        '''
        codes = {}
        for skeleton in tqdm(self.skeletons, desc="Compiling Skeletons", disable=not verbose, smoothing=0.0):
            # Codify the Expression
            executable_prefix_expression = self.simplipy_engine.operators_to_realizations(skeleton)
            prefix_expression_with_constants, constants = identify_constants(executable_prefix_expression, inplace=True)
            code_string = self.simplipy_engine.prefix_to_infix(prefix_expression_with_constants, realization=True)
            code = codify(code_string, self.variables + constants)

            codes[skeleton] = (code, constants)

        return codes

    def __contains__(self, skeleton: tuple[str] | list[str]) -> bool:
        '''
        Check if a skeleton is in the pool.

        Parameters
        ----------
        skeleton : tuple[str] or list[str]
            The skeleton to check.

        Returns
        -------
        bool
            Whether the skeleton is in the pool.
        '''
        return tuple(skeleton) in self.skeletons

    def is_held_out(self, skeleton: tuple[str] | list[str], constants: list[str], code: CodeType | None = None) -> bool:
        '''
        Check if a skeleton is held out from the pool.

        Parameters
        ----------
        skeleton : tuple[str] or list[str]
            The skeleton to check.
        constants : list[str]
            The constants used in the skeleton.
        code : CodeType or None, optional
            The compiled code for the skeleton. If not provided, it will be compiled.

        Returns
        -------
        bool
            Whether the skeleton is held out.
        '''
        if constants is None:
            raise ValueError("Need constants for test of functional equivalence")

        no_constant_expression = self.remove_num(skeleton)

        if code is None:
            executable_prefix_expression = self.simplipy_engine.operators_to_realizations(no_constant_expression)
            prefix_expression_with_constants, constants = identify_constants(executable_prefix_expression, inplace=True)
            code_string = self.simplipy_engine.prefix_to_infix(prefix_expression_with_constants, realization=True)
            code = codify(code_string, self.variables + constants)

        compiled_fn = self.simplipy_engine.code_to_lambda(code)

        try:
            return self.holdout_manager.is_held_out(
                no_constant_expression,
                compiled_fn,
                num_constants=len(constants),
            )
        except (OverflowError, NameError):
            print("Overflow or Name error during holdout evaluation, assuming held out")
            print(f'Skeleton: {skeleton}')
            print(f'Constants: {constants}')
            print(f'Code: {code}')
            return True

    @property
    def holdout_skeletons(self) -> set[tuple[str, ...]]:
        return self.holdout_manager.skeleton_hashes

    @property
    def holdout_y(self) -> set[tuple[float, ...] | tuple[tuple[float, ...], ...]]:
        return self.holdout_manager.expression_images

    @property
    def holdout_X(self) -> np.ndarray:
        return self.holdout_manager.holdout_X

    @property
    def holdout_C(self) -> np.ndarray:
        return self.holdout_manager.holdout_C

    def remove_num(self, expression: list[str] | tuple[str, ...], verbose: bool = False, debug: bool = False) -> list[str]:
        stack: list = []
        i = len(expression) - 1

        if debug:
            print(f'Input expression: {expression}')

        while i >= 0:
            token = expression[i]

            if debug:
                print(f'Stack: {stack}')
                print(f'Processing token {token}')

            if token in self.simplipy_engine.operator_arity_compat or token in self.simplipy_engine.operator_aliases:
                operator = self.simplipy_engine.operator_aliases.get(token, token)
                arity = self.simplipy_engine.operator_arity_compat[operator]
                operands = list(reversed(stack[-arity:]))

                if any(operand[0] == '<constant>' for operand in operands):
                    if verbose:
                        print('Removing constant')

                    non_num_operands = [operand for operand in operands if operand[0] != '<constant>']

                    if len(non_num_operands) == 0:
                        new_term = '<constant>'
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

        return flatten_nested_list(stack, reverse=True)

    def register_holdout_pool(self, holdout_pool: "SkeletonPool | str") -> None:
        '''
        Register a holdout pool to exclude from sampling: Cache the skeletons and their images to compare against when sampling.

        Parameters
        ----------
        holdout_pool : SkeletonPool or str
            The holdout pool object or path to register.
        '''
        if isinstance(holdout_pool, str):
            if holdout_pool in self.holdout_pools:
                return

            _, holdout_pool_obj = SkeletonPool.load(holdout_pool)
            self.holdout_pools.append(holdout_pool)
        else:
            if any(existing is holdout_pool for existing in self.holdout_pools if not isinstance(existing, str)):
                return

            holdout_pool_obj = holdout_pool
            self.holdout_pools.append(holdout_pool_obj)

        for skeleton in holdout_pool_obj.skeletons:
            no_constant_expression = holdout_pool_obj.remove_num(skeleton)
            executable_prefix_expression = holdout_pool_obj.simplipy_engine.operators_to_realizations(no_constant_expression)
            prefix_expression_with_constants, constants = identify_constants(executable_prefix_expression, inplace=True)
            code_string = holdout_pool_obj.simplipy_engine.prefix_to_infix(prefix_expression_with_constants, realization=True)
            code = codify(code_string, holdout_pool_obj.variables + constants)
            compiled_fn = holdout_pool_obj.simplipy_engine.code_to_lambda(code)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                self.holdout_manager.register_skeleton(
                    no_constant_expression,
                    compiled_fn,
                    num_constants=len(constants),
                    n_variables=holdout_pool_obj.n_variables,
                )

    def clear_holdouts(self) -> None:
        """Remove all registered holdout pools and associated constraints."""
        self.holdout_pools = []
        self.holdout_manager = HoldoutManager(n_variables=self.n_variables, allow_nan=self.allow_nan)

    def save(self, directory: str, config: dict[str, Any] | str | None = None, reference: str = 'relative', recursive: bool = True) -> None:
        '''
        Save the skeleton pool to a directory.

        Parameters
        ----------
        directory : str
            The directory to save the skeleton pool to.
        config : dict or str or None, optional
            The configuration dictionary or path to the configuration file. If None, the model will be saved without a config file.
        reference : str, optional
            The reference path to use for saving the model.
        recursive : bool, optional
            Whether to save the model recursively (i.e. save the config and holdout pools).
        '''
        directory = substitute_root_path(directory)

        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, 'skeletons.pkl'), 'wb') as pool_file:
            pickle.dump(self.skeletons, pool_file)

        # Copy the config to the directory for best portability
        if config is None:
            warnings.warn("No config specified, saving the model without a config file. Loading the model will require manual configuration.")
        else:
            save_config(load_config(config, resolve_paths=True), directory=directory, filename='skeleton_pool.yaml', reference=reference, recursive=recursive, resolve_paths=True)

    @classmethod
    def load(cls, directory: str, verbose: bool = True) -> tuple[dict[str, Any], "SkeletonPool"]:
        '''
        Load a skeleton pool from a directory.

        Parameters
        ----------
        directory : str
            The directory to load the skeleton pool from.
        verbose : bool, optional
            Whether to display a progress bar.

        Returns
        -------
        dict[str, Any], SkeletonPool
            The configuration dictionary and the SkeletonPool object.
        '''
        config_path = os.path.join(directory, 'skeleton_pool.yaml')
        resolved_directory = substitute_root_path(directory)

        pool = cls.from_config(config_path)

        with open(os.path.join(resolved_directory, 'skeletons.pkl'), 'rb') as pool_file:
            pool.skeletons = pickle.load(pool_file)

            pool.skeleton_codes = pool.compile_codes(verbose=verbose)

        return load_config(config_path), pool

    def sample_skeleton(self, new: bool = False, decontaminate: bool = True) -> tuple[tuple[str], CodeType, list[str]]:
        '''
        Sample a skeleton from the pool.

        Parameters
        ----------
        new : bool, optional
            Whether to sample a new skeleton or reuse an existing one.

        Returns
        -------
        tuple[str], CodeType, list[str]
            The skeleton, its compiled code, and the constants used in the skeleton.
        '''
        if len(self.skeletons) == 0 or new:
            for _ in range(self.sample_strategy['max_tries']):
                match self.sample_strategy['n_operator_distribution']:
                    case "equiprobable_lengths":
                        n_operators = np.random.randint(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1)
                    case "length_proportional":
                        if self.operator_probs is None:
                            self.operator_probs = np.arange(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1)**self.sample_strategy['power']
                            self.operator_probs = self.operator_probs / self.operator_probs.sum()
                        n_operators = np.random.choice(
                            range(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1),
                            p=self.operator_probs)
                    case "length_exponential":
                        if self.operator_probs is None:
                            self.operator_probs = np.exp(np.arange(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1)**self.sample_strategy['power'] / self.sample_strategy['lambda'])
                            self.operator_probs = self.operator_probs / self.operator_probs.sum()
                        n_operators = np.random.choice(
                            range(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1),
                            p=self.operator_probs)
                    case _:
                        raise ValueError(f"Invalid n_operator_distribution: {self.sample_strategy['n_operator_distribution']}")

                skeleton = self.skeleton_sampler.sample(n_operators)
                if self.simplify:
                    try:
                        skeleton = self.simplipy_engine.simplify(skeleton, inplace=True, max_pattern_length=4)
                    except Exception as e:
                        print(f"Failed to simplify skeleton: {skeleton}")
                        raise NoValidSampleFoundError(f"Failed to simplify skeleton: {skeleton}") from e

                    if any(forbidden_token in skeleton for forbidden_token in ['float("inf")', 'float("-inf")', 'float("nan")']):
                        raise NoValidSampleFoundError(f"Skeleton contains forbidden tokens: {skeleton}")

                if tuple(skeleton) not in self.skeletons and len(skeleton) <= self.sample_strategy['max_length']:
                    executable_prefix_expression = self.simplipy_engine.operators_to_realizations(skeleton)
                    prefix_expression_with_constants, constants = identify_constants(executable_prefix_expression, inplace=True)
                    code_string = self.simplipy_engine.prefix_to_infix(prefix_expression_with_constants, realization=True)
                    code = codify(code_string, self.variables + constants)

                    if not decontaminate or not self.is_held_out(skeleton, constants):
                        return tuple(skeleton), code, constants   # type: ignore
        else:
            skeleton = random.choice(tuple(self.skeletons))  # type: ignore
            code, constants = self.skeleton_codes[skeleton]  # type: ignore

            return skeleton, code, constants  # type: ignore

        raise NoValidSampleFoundError(f"Failed to sample a non-contaminated skeleton after {self.sample_strategy['max_tries']} retries")

    def sample_data(self, code: CodeType, n_constants: int = 0, n_support: int | None = None, support_prior: Callable | None = None, support_scale_prior: Callable | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Sample support points and literals for an expression.

        Parameters
        ----------
        code : CodeType
            The compiled code for the expression.
        n_constants : int, optional
            The number of constants to sample.
        n_support : int or None, optional
            The number of support points to sample. If None, the number of support points will be sampled from the prior distribution.
        support_prior : Callable or None, optional
            The prior distribution for the support points. If None, the default support prior will be used.
        support_scale_prior : Callable or None, optional
            The prior distribution for the support scale. If None, the default support scale prior will be

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The support points, their images, and the literals.
        '''
        expression_callable = self.simplipy_engine.code_to_lambda(code)
        if n_support is None:
            n_support = self.support_sampler.sample_n_support()

        for _ in range(self.sample_strategy['max_tries']):
            literals = self.literal_prior(size=n_constants).astype(np.float32)

            override_support_prior = SupportSampler.ensure_prior_callable(support_prior) if support_prior is not None else None
            override_support_scale = SupportSampler.ensure_prior_callable(support_scale_prior) if support_scale_prior is not None else None

            try:
                x_support = self.support_sampler.sample(
                    n_support=n_support,
                    support_prior=override_support_prior,
                    support_scale_prior=override_support_scale,
                )
            except SupportSamplingError:
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                y_support = expression_callable(*x_support.T, *literals)

            if not isinstance(y_support, np.ndarray):
                y_support = np.full((n_support, 1), y_support)

            if len(y_support) == 1:
                # Repeat y to match the shape of x
                y_support = np.repeat(y_support, x_support.shape[0])

            # Complex numbers are not supported
            if np.iscomplex(y_support).any() or y_support.dtype != np.float32:
                continue

            if not self.allow_nan:
                # If any of the support points are NaN, skip the expression
                if np.isnan(x_support).any() or np.isinf(x_support).any():
                    continue
                if np.isnan(y_support).any() or np.isinf(y_support).any():
                    continue
            elif np.isnan(y_support).all():
                # Even if NaNs are allowed, if all support points are NaN, skip the expression
                continue

            # All checks passed, break the loop
            break
        else:
            raise NoValidSampleFoundError(f"Failed to generate a valid expression after {self.sample_strategy['max_tries']} retries")

        return x_support, y_support.reshape(-1, 1), literals

    def sample(self, n_support: int | None = None) -> dict:
        '''
        Sample a skeleton, support points, and literals.

        Parameters
        ----------
        n_support : int or None, optional
            The number of support points to sample. If None, the number of support points will be sampled from the prior distribution.

        Returns
        -------
        dict
            A dictionary containing the skeleton hash, code, constants, support points, and literals.
        '''
        skeleton_hash, skeleton_code, skeleton_constants = self.sample_skeleton()
        x_support, y_support, literals = self.sample_data(skeleton_code, len(skeleton_constants), n_support)

        return {
            'skeleton_hash': skeleton_hash,
            'skeleton_code': skeleton_code,
            'skeleton_constants': skeleton_constants,
            'x_support': x_support,
            'y_support': y_support,
            'literals': literals
        }

    def create(self, size: int, verbose: bool = False) -> None:
        '''
        Create a pool of skeletons of a given size.

        Parameters
        ----------
        size : int
            The number of skeletons to create.
        verbose : bool, optional
            Whether to display a progress bar.
        '''
        n_skipped = 0
        n_created = len(self.skeletons)

        pbar = tqdm(total=size, desc="Creating Skeleton Pool", disable=not verbose, smoothing=0.0)

        while n_created < size:
            try:
                skeleton, code, constants = self.sample_skeleton(new=True)
            except NoValidSampleFoundError:
                n_skipped += 1
                pbar.set_postfix_str(f"Skipped: {n_skipped:,}")
                continue

            if not self.simplipy_engine.is_valid(skeleton):
                raise ValueError(f"Invalid skeleton: {skeleton}")

            if not isinstance(skeleton, tuple):
                skeleton = tuple(skeleton)
            self.skeletons.add(skeleton)
            self.skeleton_codes[skeleton] = (code, constants)
            n_created += 1

            pbar.update(1)
            pbar.set_postfix_str(f"Skipped: {n_skipped:,}")

            if n_created >= size:
                break

    def split(self, train_size: float, random_state: int | None = None) -> tuple["SkeletonPool", "SkeletonPool"]:
        """
        Split the skeleton pool into two disjoint pools randomly.

        Parameters
        ----------
        train_size : float
            The size of the first pool as a fraction of the original pool size.
        random_state : int or None, optional
            The random state to use for splitting.

        Returns
        -------
        tuple[SkeletonPool, SkeletonPool]
            The two disjoint skeleton pools.
        """
        train_keys, test_keys = train_test_split(list(self.skeletons), train_size=train_size, random_state=random_state)

        train_pool = SkeletonPool.from_dict(
            set(train_keys),
            simplipy_engine=self.simplipy_engine,
            sample_strategy=self.sample_strategy,
            literal_prior=self.literal_prior,
            skeleton_codes={k: v for k, v in self.skeleton_codes.items() if k in train_keys},
            variables=self.variables,
            support_sampler_config=deepcopy(self.support_sampler_config),
            operator_weights=self.operator_weights,
            holdout_pools=self.holdout_pools,
            allow_nan=self.allow_nan)
        test_pool = SkeletonPool.from_dict(
            set(test_keys),
            simplipy_engine=self.simplipy_engine,
            sample_strategy=self.sample_strategy,
            literal_prior=self.literal_prior,
            skeleton_codes={k: v for k, v in self.skeleton_codes.items() if k in test_keys},
            variables=self.variables,
            support_sampler_config=deepcopy(self.support_sampler_config),
            operator_weights=self.operator_weights,
            holdout_pools=self.holdout_pools,
            allow_nan=self.allow_nan)

        return train_pool, test_pool

    def __len__(self) -> int:
        '''
        Get the number of skeletons in the pool.

        Returns
        -------
        int
            The number of skeletons in the pool.
        '''
        return len(self.skeletons)

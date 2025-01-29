import os
import warnings
import random
import pickle

from types import CodeType
from typing import Any, Callable

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

from flash_ansr.utils import load_config, substitute_root_path, save_config
from flash_ansr.expressions.expression_space import ExpressionSpace
from flash_ansr.expressions.utils import codify, num_to_constants, generate_ubi_dist, get_distribution


class NoValidSampleFoundError(Exception):
    pass


class SkeletonPool:
    '''
    Manage and sample from a set of expression 'skeletons' (prefix expressions with placeholders for constants).

    Parameters
    ----------
    expression_space : ExpressionSpace
        The expression space to operate in.
    sample_strategy : dict[str, Any]
        The strategy to use for sampling skeletons.
    literal_prior : str or Callable[..., np.ndarray]
        The prior distribution for the literals.
    literal_prior_kwargs : dict[str, Any]
        The keyword arguments to pass to the literal prior distribution.
    support_prior : str or Callable[..., np.ndarray]
        The prior distribution for the support points.
    support_prior_kwargs : dict[str, Any]
        The keyword arguments to pass to the support prior distribution.
    n_support_prior : str or Callable[..., np.ndarray]
        The prior distribution for the number of support points.
    n_support_prior_kwargs : dict[str, Any]
        The keyword arguments to pass to the number of support points prior distribution.
    holdout_pools : list[SkeletonPool] or None, optional
        A list of SkeletonPools (i.e. sets of skeletons) to exclude from sampling.
    allow_nan : bool, optional
        Whether to allow NaNs in the support points.
    simplify : bool, optional
        Whether to simplify sampled skeletons.
    '''

    def __init__(
            self,
            expression_space: ExpressionSpace,
            sample_strategy: dict[str, Any],
            literal_prior: str | Callable[..., np.ndarray],
            literal_prior_kwargs: dict[str, Any],
            support_prior: str | Callable[..., np.ndarray],
            support_prior_kwargs: dict[str, Any],
            n_support_prior: str | Callable[..., np.ndarray],
            n_support_prior_kwargs: dict[str, Any],
            holdout_pools: list["SkeletonPool"] | None = None,
            allow_nan: bool = False,
            simplify: bool = True) -> None:
        self.expression_space = expression_space
        self.sample_strategy = sample_strategy

        np.random.default_rng(seed=0)
        self.holdout_X = np.random.uniform(-10, 10, (512, 3))
        self.holdout_C = np.random.uniform(-10, 10, (512, 100))
        self.holdout_y: set[tuple] = set()
        self.holdout_skeletons: set[tuple[str]] = set()

        self.holdout_pools: list["SkeletonPool"] = []
        for holdout_pool in holdout_pools or []:
            self.register_holdout_pool(holdout_pool)

        self.skeletons: set[tuple[str]] = set()
        self.skeleton_codes: dict[tuple[str], tuple[CodeType, list[str]]] = {}

        # Parameters from https://github.com/facebookresearch/SymbolicMathematics/blob/main/src/envs/char_sp.py
        self._n_leaves = 1
        self._n_unary_operators = 1
        self._n_binary_operators = 1

        self.unary_operators = [k for k, v in self.expression_space.operator_arity.items() if v == 1]
        self.binary_operators = [k for k, v in self.expression_space.operator_arity.items() if v == 2]

        self.unary_operator_probs = np.array([self.expression_space.operator_weights[op] for op in self.unary_operators])
        self.unary_operator_probs = self.unary_operator_probs / self.unary_operator_probs.sum()
        self.binary_operator_probs = np.array([self.expression_space.operator_weights[op] for op in self.binary_operators])
        self.binary_operator_probs = self.binary_operator_probs / self.binary_operator_probs.sum()

        self.variable_probability = len(self.expression_space.variables) / (len(self.expression_space.variables) + 1)

        self.unary_binary_distribution = generate_ubi_dist(
            self.sample_strategy.get('max_operators', 10),
            self._n_leaves, self._n_unary_operators, self._n_binary_operators)

        self.literal_prior = get_distribution(literal_prior, literal_prior_kwargs)
        self.literal_prior_kwargs = literal_prior_kwargs
        self.support_prior = get_distribution(support_prior, support_prior_kwargs)
        self.support_prior_kwargs = support_prior_kwargs
        self.n_support_prior = get_distribution(n_support_prior, n_support_prior_kwargs)
        self.n_support_prior_kwargs = n_support_prior_kwargs

        self.allow_nan = allow_nan
        self.simplify = simplify

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
        if isinstance(config, str) and isinstance(config_["expression_space"], str):
            if config_["expression_space"].startswith('.'):
                config_["expression_space"] = os.path.join(os.path.dirname(config), config_["expression_space"])

        return cls(
            expression_space=ExpressionSpace.from_config(config_["expression_space"]),
            sample_strategy=config_["sample_strategy"],
            literal_prior=config_["literal_prior"],
            literal_prior_kwargs=config_["literal_prior_kwargs"],
            support_prior=config_["support_prior"],
            support_prior_kwargs=config_["support_prior_kwargs"],
            n_support_prior=config_["n_support_prior"],
            n_support_prior_kwargs=config_["n_support_prior_kwargs"],
            holdout_pools=config_["holdout_pools"],
            allow_nan=config_["allow_nan"],
            simplify=config_.get("simplify", True)
        )

    @classmethod
    def from_dict(
            cls,
            skeletons: set[tuple[str]],
            expression_space: ExpressionSpace,
            sample_strategy: dict[str, Any],
            literal_prior: str | Callable[..., np.ndarray],
            literal_prior_kwargs: dict[str, Any],
            support_prior: str | Callable[..., np.ndarray],
            support_prior_kwargs: dict[str, Any],
            n_support_prior: str | Callable[..., np.ndarray],
            n_support_prior_kwargs: dict[str, Any],
            skeleton_codes: dict[tuple[str], tuple[CodeType, list[str]]] | None = None,
            holdout_pools: list["SkeletonPool"] | None = None,
            allow_nan: bool = False,
            simplify: bool = True) -> "SkeletonPool":
        '''
        Create a SkeletonPool from a set of skeletons.

        Parameters
        ----------
        skeletons : set[tuple[str]]
            The set of skeletons to include in the pool.
        expression_space : ExpressionSpace
            The expression space to operate in.
        sample_strategy : dict[str, Any]
            The strategy to use for sampling skeletons.
        literal_prior : str or Callable[..., np.ndarray]
            The prior distribution for the literals.
        literal_prior_kwargs : dict[str, Any]
            The keyword arguments to pass to the literal prior distribution.
        support_prior : str or Callable[..., np.ndarray]
            The prior distribution for the support points.
        support_prior_kwargs : dict[str, Any]
            The keyword arguments to pass to the support prior distribution.
        n_support_prior : str or Callable[..., np.ndarray]
            The prior distribution for the number of support points.
        n_support_prior_kwargs : dict[str, Any]
            The keyword arguments to pass to the number of support points prior distribution.
        skeleton_codes : dict[tuple[str], tuple[CodeType, list[str]]] or None, optional
            A dictionary mapping skeletons to their compiled codes.
        holdout_pools : list[SkeletonPool] or None, optional
            A list of SkeletonPools (i.e. sets of skeletons) to exclude from sampling.
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
            expression_space=expression_space,
            sample_strategy=sample_strategy,
            literal_prior=literal_prior,
            literal_prior_kwargs=literal_prior_kwargs,
            support_prior=support_prior,
            support_prior_kwargs=support_prior_kwargs,
            n_support_prior=n_support_prior,
            n_support_prior_kwargs=n_support_prior_kwargs,
            holdout_pools=holdout_pools or [],
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
        for skeleton in tqdm(self.skeletons, desc="Compiling Skeletons", disable=not verbose):
            # Codify the Expression
            executable_prefix_expression = self.expression_space.operators_to_realizations(skeleton)
            prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression)
            code_string = self.expression_space.prefix_to_infix(prefix_expression_with_constants, realization=True)
            code = codify(code_string, self.expression_space.variables + constants)

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

        if tuple(skeleton) in self.holdout_skeletons:  # (symbolic equivalence)
            return True

        if code is None:
            # Remove constants since permutations are not detected as duplicates
            no_constant_expression = self.expression_space.remove_num(skeleton)
            executable_prefix_expression = self.expression_space.operators_to_realizations(no_constant_expression)
            prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression)
            code_string = self.expression_space.prefix_to_infix(prefix_expression_with_constants, realization=True)
            code = codify(code_string, self.expression_space.variables + constants)

        # Evaluate the expression and check if its image is in the holdout images (functional equivalence)
        f = self.expression_space.code_to_lambda(code)

        # FIXME: Different orders of constans may not be detected as duplicates
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X_with_constants = np.concatenate([self.holdout_X, self.holdout_C[:, :len(constants)]], axis=1)
        try:
            expression_image = f(*X_with_constants.T).round(4)
            expression_image[np.isnan(expression_image)] = 0  # Cannot compare NaNs
        except OverflowError:
            return True  # Just to be safe

        if tuple(expression_image) in self.holdout_y:
            return True

        return False

    def register_holdout_pool(self, holdout_pool: "SkeletonPool") -> None:
        '''
        Register a holdout pool to exclude from sampling: Cache the skeletons and their images to compare against when sampling.

        Parameters
        ----------
        holdout_pool : SkeletonPool or str
            The holdout pool to register.
        '''
        if isinstance(holdout_pool, str):
            _, holdout_pool = SkeletonPool.load(holdout_pool)

        for skeleton in holdout_pool.skeletons:
            # Remove constants since permutations are not detected as duplicates
            no_constant_expression = self.expression_space.remove_num(skeleton)
            executable_prefix_expression = self.expression_space.operators_to_realizations(no_constant_expression)
            prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression)
            code_string = self.expression_space.prefix_to_infix(prefix_expression_with_constants, realization=True)
            code = codify(code_string, self.expression_space.variables + constants)

            # Evaluate the Expression and store the result
            f = self.expression_space.code_to_lambda(code)
            X_with_constants = np.concatenate([self.holdout_X, self.holdout_C[:, :len(constants)]], axis=1)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                expression_image = f(*X_with_constants.T).round(4)
                expression_image[np.isnan(expression_image)] = 0  # Cannot compare NaNs
            except OverflowError:
                self.holdout_skeletons.add(skeleton)
                continue

            self.holdout_skeletons.add(skeleton)
            self.holdout_y.add(tuple(expression_image))

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

    def sample_next_pos_ubi(self, n_empty_nodes: int, n_operators: int) -> tuple[int, int]:
        '''
        See https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/blob/main/src/nesymres/dataset/generator.py
        '''
        assert n_empty_nodes > 0
        assert n_operators > 0

        # Check if the unary-binary distribution needs to be expanded
        if n_empty_nodes >= len(self.unary_binary_distribution):
            self.unary_binary_distribution = generate_ubi_dist(n_empty_nodes + 1, self._n_leaves, self._n_unary_operators, self._n_binary_operators)

        probs = []
        for i in range(n_empty_nodes):
            probs.append((self._n_leaves ** i) * self._n_unary_operators * self.unary_binary_distribution[n_empty_nodes - i][n_operators - 1])
        for i in range(n_empty_nodes):
            probs.append((self._n_leaves ** i) * self._n_binary_operators * self.unary_binary_distribution[n_empty_nodes - i + 1][n_operators - 1])

        probs = [p / self.unary_binary_distribution[n_empty_nodes][n_operators] for p in probs]
        probs = np.array(probs, dtype=np.float64)  # type: ignore

        e = np.random.choice(2 * n_empty_nodes, p=probs)

        arity = 1 if e < n_empty_nodes else 2
        e = e % n_empty_nodes

        return e, arity

    def get_leaf(self) -> list[str]:
        '''
        Sample a leaf node (either a variable or a constant).

        Returns
        -------
        list[str]
            The leaf node.
        '''
        if random.random() < self.variable_probability:
            return [random.choice(self.expression_space.variables)]
        else:
            return ['<num>']

    def _sample_skeleton(self, n_operators: int) -> list[str]:
        '''
        Create a tree with exactly `n_operators` operators.

        Parameters
        ----------
        n_operators : int
            The number of operators to include in the tree.

        Returns
        -------
        list[str]
            The tree as a list of tokens.

        Notes
        -----
        See https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/blob/main/src/nesymres/dataset/generator.py
        '''
        stack: list[str | None] = [None]
        n_empty_nodes = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        # create tree
        for n in range(n_operators, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(n_empty_nodes, n)
            if arity == 1:
                op = np.random.choice(self.unary_operators, p=self.unary_operator_probs)
            else:
                op = np.random.choice(self.binary_operators, p=self.binary_operator_probs)

            n_empty_nodes += self.expression_space.operator_arity[op] - 1 - skipped  # created empty nodes - skipped future leaves
            t_leaves += self.expression_space.operator_arity[op] - 1            # update number of total leaves
            l_leaves += skipped                           # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = stack[:pos] + [op] + [None for _ in range(self.expression_space.operator_arity[op])] + stack[pos + 1:]

        # sanity check
        assert len([1 for v in stack if v in self.expression_space.operator_arity.keys()]) == n_operators
        assert len([1 for v in stack if v is None]) == t_leaves

        # create leaves
        leaves = [self.get_leaf() for _ in range(t_leaves)]
        np.random.shuffle(leaves)

        # insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1:]
        assert len(leaves) == 0

        return stack  # type: ignore

    def sample_skeleton(self, new: bool = False) -> tuple[tuple[str], CodeType, list[str]]:
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
                        if not hasattr(self, 'operator_probs'):
                            self.operator_probs = np.arange(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1)**self.sample_strategy['power']
                            self.operator_probs = self.operator_probs / self.operator_probs.sum()
                        n_operators = np.random.choice(
                            range(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1),
                            p=self.operator_probs)
                    case "exponential":
                        if not hasattr(self, 'operator_probs'):
                            self.operator_probs = np.exp(np.arange(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1) / self.sample_strategy['lambda'])
                            self.operator_probs = self.operator_probs / self.operator_probs.sum()
                        n_operators = np.random.choice(
                            range(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1),
                            p=self.operator_probs)
                    case _:
                        raise ValueError(f"Invalid n_operator_distribution: {self.sample_strategy['n_operator_distribution']}")

                skeleton = self._sample_skeleton(n_operators)
                if self.simplify:
                    skeleton = self.expression_space.simplify(skeleton)

                if tuple(skeleton) not in self.skeletons and len(skeleton) <= self.sample_strategy['max_length']:
                    executable_prefix_expression = self.expression_space.operators_to_realizations(skeleton)
                    prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression)
                    code_string = self.expression_space.prefix_to_infix(prefix_expression_with_constants, realization=True)
                    code = codify(code_string, self.expression_space.variables + constants)

                    if not self.is_held_out(skeleton, constants, code):
                        return tuple(skeleton), code, constants   # type: ignore
        else:
            skeleton = random.choice(tuple(self.skeletons))  # type: ignore
            code, constants = self.skeleton_codes[skeleton]  # type: ignore

            return skeleton, code, constants  # type: ignore

        raise NoValidSampleFoundError(f"Failed to sample a non-contaminated skeleton after {self.sample_strategy['max_tries']} retries")

    def sample_data(self, code: CodeType, n_constants: int = 0, n_support: int | None = None, support_prior: Callable | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The support points, their images, and the literals.
        '''
        expression_callable = self.expression_space.code_to_lambda(code)
        if n_support is None:
            n_support = int(np.round(self.n_support_prior(size=1))[0])

        for _ in range(self.sample_strategy['max_tries']):
            literals = self.literal_prior(size=n_constants).astype(np.float32)

            support_prior = support_prior or self.support_prior

            # Use the default support prior as defined by the configuration
            if self.sample_strategy.get('independent_dimensions', False):
                # Sample each dimension independently
                x_support = np.concatenate([support_prior(size=(n_support, 1)) for _ in range(len(self.expression_space.variables))], axis=1).astype(np.float32)
            else:
                #
                x_support = support_prior(size=(n_support, len(self.expression_space.variables))).astype(np.float32)

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
        try:
            n_duplicates = 0
            n_invalid = 0
            n_created = len(self.skeletons)

            pbar = tqdm(total=size, desc="Creating Skeleton Pool", disable=not verbose)

            while n_created < size:
                try:
                    skeleton, code, constants = self.sample_skeleton(new=True)
                except NoValidSampleFoundError:
                    continue

                if self.simplify:
                    simplified_skeleton = self.expression_space.simplify(skeleton)
                else:
                    simplified_skeleton = skeleton

                if not self.expression_space.is_valid(simplified_skeleton):
                    n_invalid += 1
                    pbar.set_postfix_str(f"Duplicates: {n_duplicates:,}, Invalid: {n_invalid:,}")
                    continue
                    # raise ValueError(f"Invalid simplified skeleton: {skeleton} -> {simplified_skeleton}")

                if skeleton in self.skeletons:
                    n_duplicates += 1
                    pbar.set_postfix_str(f"Duplicates: {n_duplicates:,}, Invalid: {n_invalid:,}")
                    continue

                h = tuple(simplified_skeleton)
                self.skeletons.add(h)
                self.skeleton_codes[h] = (code, constants)
                n_created += 1

                pbar.update(1)
                pbar.set_postfix_str(f"Duplicates: {n_duplicates:,}, Invalid: {n_invalid:,}")

                if n_created >= size:
                    break

        except (IndexError, ValueError, KeyError):
            print(skeleton)
            raise

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
            expression_space=self.expression_space,
            sample_strategy=self.sample_strategy,
            literal_prior=self.literal_prior,
            literal_prior_kwargs=self.literal_prior_kwargs,
            support_prior=self.support_prior,
            support_prior_kwargs=self.support_prior_kwargs,
            n_support_prior=self.n_support_prior,
            skeleton_codes={k: v for k, v in self.skeleton_codes.items() if k in train_keys},
            n_support_prior_kwargs=self.n_support_prior_kwargs,
            holdout_pools=self.holdout_pools,
            allow_nan=self.allow_nan)
        test_pool = SkeletonPool.from_dict(
            set(test_keys),
            expression_space=self.expression_space,
            sample_strategy=self.sample_strategy,
            literal_prior=self.literal_prior,
            literal_prior_kwargs=self.literal_prior_kwargs,
            support_prior=self.support_prior,
            support_prior_kwargs=self.support_prior_kwargs,
            n_support_prior=self.n_support_prior,
            n_support_prior_kwargs=self.n_support_prior_kwargs,
            skeleton_codes={k: v for k, v in self.skeleton_codes.items() if k in test_keys},
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

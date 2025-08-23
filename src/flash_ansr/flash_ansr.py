import os
import copy
from typing import Literal, Any, Iterable, TypedDict, Callable
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

from sklearn.base import BaseEstimator

from simplipy import SimpliPyEngine

from flash_ansr.utils import substitute_root_path, pad_input_set, GenerationConfig
from flash_ansr.refine import Refiner, ConvergenceError
from flash_ansr.model import FlashANSRModel, Tokenizer
from flash_ansr.train.train import OptimizerFactory
from flash_ansr.train.scheduler import LRSchedulerFactory


class Result(TypedDict):
    refiner: Refiner
    numeric_prediction: torch.Tensor | None
    beam: list[int]
    log_prob: float
    expression: list[str]
    raw_beam: list[int]
    raw_beam_decoded: str
    complexity: int
    function: Callable
    fits: list[tuple[np.ndarray, np.ndarray, float]]
    score: float
    target_complexity: int | float | None
    fvu: float


class FlashANSR(BaseEstimator):
    '''
    Flash Amortized Neural Symbolic Regressor.

    Parameters
    ----------
    simplipy_engine : SimpliPyEngine
        The expression space used for manipulating expressions.
    flash_ansr_transformer : FlashANSRTransformer
        The core transformer model.
    generation_config : GenerationConfig, optional
        The generation configuration, by default None.
    numeric_head : bool, optional
        Whether to use the numeric head, by default False.
    n_restarts : int, optional
        The number of restarts for the refiner, by default 1.
    refiner_method : str, optional
        The optimization method to use. One of
        - 'curve_fit_lm': Use the curve_fit method with the Levenberg-Marquardt algorithm
        - 'minimize_bfgs': Use the minimize method with the BFGS algorithm
    p0_noise : {'uniform', 'normal'}, optional
        The type of noise to add to the initial guess, by default 'normal'.
    p0_noise_kwargs : dict, optional
        The keyword arguments for the noise function, by default None.
    numpy_errors : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        The behavior for numpy errors, by default 'ignore'.
    parsimony : float, optional
        The parsimony coefficient, by default 0.2.
    verbose : bool, optional
        Whether to print verbose output, by default False.
    '''
    def __init__(
            self,
            simplipy_engine: SimpliPyEngine,
            flash_ansr_transformer: FlashANSRModel,
            tokenizer: Tokenizer,
            generation_config: GenerationConfig | None = None,
            numeric_head: bool = False,
            n_restarts: int = 8,
            refiner_method: Literal['curve_fit_lm', 'minimize_bfgs'] = 'curve_fit_lm',
            refiner_p0_noise: Literal['uniform', 'normal'] | None = 'uniform',
            refiner_p0_noise_kwargs: dict | None = {'low': -5, 'high': 5},
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            parsimony: float = 0.05):
        self.simplipy_engine = simplipy_engine
        self.flash_ansr_transformer = flash_ansr_transformer.eval()
        self.tokenizer = tokenizer

        if generation_config is None:
            generation_config = GenerationConfig()

        self.generation_config = generation_config
        self.numeric_head = numeric_head
        self.n_restarts = n_restarts
        self.refiner_method = refiner_method
        self.refiner_p0_noise = refiner_p0_noise
        self.refiner_p0_noise_kwargs = copy.deepcopy(refiner_p0_noise_kwargs) if refiner_p0_noise_kwargs is not None else None
        self.numpy_errors = numpy_errors
        self.parsimony = parsimony

        self._results: list[Result] = []
        self.results: pd.DataFrame = pd.DataFrame()

        self.variable_mapping: dict[str, str] = {}

    @classmethod
    def load(
            cls,
            directory: str,
            generation_config: GenerationConfig | None = None,
            numeric_head: bool = False,
            n_restarts: int = 1,
            refiner_method: Literal['curve_fit_lm', 'minimize_bfgs'] = 'curve_fit_lm',
            refiner_p0_noise: Literal['uniform', 'normal'] | None = 'uniform',
            refiner_p0_noise_kwargs: dict | None = {'low': -5, 'high': 5},
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            parsimony: float = 0.05,
            device: str = 'cpu') -> "FlashANSR":
        directory = substitute_root_path(directory)

        simplipy_engine_path = os.path.join(directory, 'simplipy_engine.yaml')
        flash_ansr_transformer_path = os.path.join(directory, 'model.yaml')
        tokenizer_path = os.path.join(directory, 'tokenizer.yaml')

        simplipy_engine = SimpliPyEngine.from_config(simplipy_engine_path)

        model = FlashANSRModel.from_config(flash_ansr_transformer_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True, map_location=device))
        model.eval().to(device)

        tokenizer = Tokenizer.from_config(tokenizer_path)

        return cls(
            simplipy_engine=simplipy_engine,
            flash_ansr_transformer=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            numeric_head=numeric_head,
            n_restarts=n_restarts,
            refiner_method=refiner_method,
            refiner_p0_noise=refiner_p0_noise,
            refiner_p0_noise_kwargs=refiner_p0_noise_kwargs,
            numpy_errors=numpy_errors,
            parsimony=parsimony)

    @property
    def n_variables(self) -> int:
        '''
        The number of variables the model was trained on.
        '''
        return self.flash_ansr_transformer.encoder_max_n_variables - 1

    def _truncate_input(self, X: np.ndarray | torch.Tensor | pd.DataFrame) -> np.ndarray | torch.Tensor | pd.DataFrame:
        if X.shape[-1] <= self.n_variables:
            return X

        warnings.warn(f"Input data has more variables than the model was trained on. The model was trained on {self.n_variables = } variables, but the input data has {X.shape[-1] = } variables. X and y will be truncated to {self.n_variables} variables.")
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, :self.n_variables]

        try:
            return X[..., :self.n_variables]
        except IndexError:
            try:
                return X[:, :self.n_variables]
            except IndexError as exc:
                raise ValueError('Cannot truncate the input data') from exc

    def generate(self, data: torch.Tensor, complexity: int | float | None = None, verbose: bool = False) -> tuple[list[list[int]], list[float], list[bool]]:
        match self.generation_config.method:
            case 'beam_search':
                return self.flash_ansr_transformer.beam_search(
                    data=data,
                    complexity=complexity,
                    verbose=verbose,
                    **self.generation_config)
            case 'softmax_sampling':
                return self.flash_ansr_transformer.sample_top_kp(
                    data=data,
                    complexity=complexity,
                    verbose=verbose,
                    **self.generation_config)
            case _:
                raise ValueError(f"Invalid generation method: {self.generation_config.method}")

    def fit(
            self,
            X: np.ndarray | torch.Tensor | pd.DataFrame,
            y: np.ndarray | torch.Tensor | pd.DataFrame | pd.Series,
            complexity: int | float | Iterable | None = None,
            variable_names: list[str] | dict[str, str] | Literal['auto'] | None = 'auto',
            converge_error: Literal['raise', 'ignore', 'print'] = 'ignore',
            verbose: bool = False) -> None:
        '''
        Perform symbolic regression on the input data.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor | pd.DataFrame
            The input data.
        y : np.ndarray | torch.Tensor | pd.DataFrame | pd.Series
            The target data.
        complexity : int | list[int] | None, optional
            The desired complexity (length in tokens) of the expression, by default None.
        variable_names : list[str] | dict[str, str] | {'auto'}, optional
            The variable names, by default 'auto'.
            - If list[str], the i-th column of X will be named variable_names[i].
            - If dict[str, str]:
                - If X is array-like, the i-th column of X will be named after the variable_names keys
                - If X is a DataFrame, variable_names will be used to map the column names to the variable names.
            - If 'auto':
                - If X is a DataFrame, the column names will be used as variable names.
                - If X is an array or tensor, the variables will be named x0, x1, x2, ...
            - If None, the variables will be named x0, x1, x2, ...
        converge_error : {'raise', 'ignore', 'print'}, optional
            The behavior for convergence errors, by default 'ignore'.
        verbose : bool, optional
            Whether to display a progress bar, by default False.
        '''

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif y.shape[-1] != 1:
            raise ValueError("The target data must have a single output dimension")

        X = self._truncate_input(X)

        # Default: No mapping
        self.variable_mapping = {}

        if isinstance(variable_names, list):
            # column i -> variable_names[i]
            self.variable_mapping = {f"x{i + 1}": name for i, name in enumerate(variable_names)}

        elif isinstance(variable_names, dict):
            if isinstance(X, pd.DataFrame):
                # column i -> variable_names[column i]
                self.variable_mapping = {f"x{i + 1}": variable_names[c] for i, c in enumerate(X.columns)}
            else:
                # custom mapping
                self.variable_mapping = variable_names

        elif variable_names == 'auto':
            if isinstance(X, pd.DataFrame):
                # column i -> column name
                self.variable_mapping = {f"x{i + 1}": name for i, name in enumerate(X.columns)}

        # TODO: Improve the handling of different types
        if complexity is None or not hasattr(complexity, '__iter__'):
            complexity_list: list[int | float | None] = [complexity]
        else:
            complexity_list = complexity  # type: ignore

        with torch.no_grad():
            # Convert the input data to a tensor
            if not isinstance(X, torch.Tensor):
                if isinstance(X, pd.DataFrame):
                    X = torch.tensor(X.values, dtype=torch.float32, device=self.flash_ansr_transformer.device)
                else:
                    X = torch.tensor(X, dtype=torch.float32, device=self.flash_ansr_transformer.device)
            else:
                X = X.to(self.flash_ansr_transformer.device)

            if not isinstance(y, torch.Tensor):
                if isinstance(y, (pd.DataFrame, pd.Series)):
                    y = torch.tensor(y.values, dtype=torch.float32, device=self.flash_ansr_transformer.device)
                else:
                    y = torch.tensor(y, dtype=torch.float32, device=self.flash_ansr_transformer.device)
            else:
                y = y.to(self.flash_ansr_transformer.device)

            if y.dim() == 1:
                y = y.unsqueeze(-1)

            y_variance = y.var(dim=0).item()

            X = pad_input_set(X, self.n_variables)

            # Concatenate x and y along the feature dimension
            data_tensor = torch.cat([X, y], dim=-1)

            self._results = []

            # Silence numpy errors
            numpy_errors_before = np.geterr()
            np.seterr(all=self.numpy_errors)

            # --- INFERENCE ---
            for complexity in complexity_list:
                raw_beams, log_probs, _ = self.generate(data_tensor, complexity=complexity, verbose=verbose)

                beams = [self.flash_ansr_transformer.extract_expression_from_beam(raw_beam)[0] for raw_beam in raw_beams]

                raw_beams_decoded = [self.tokenizer.decode(raw_beam, special_tokens='<constant>') for raw_beam in raw_beams]
                beams_decoded = [self.tokenizer.decode(beam, special_tokens='<constant>') for beam in beams]

                # Fit the refiner to each beam
                for raw_beam, raw_beam_decoded, beam, beam_decoded, log_prob in tqdm(zip(raw_beams, raw_beams_decoded, beams, beams_decoded, log_probs), desc="Fitting Constants", disable=not verbose, total=len(beams)):
                    if self.simplipy_engine.is_valid(beam_decoded):
                        numeric_prediction = None

                        if self.numeric_head:
                            raise NotImplementedError("Numeric head is not yet implemented")
                            # with torch.no_grad():
                            #     _, num_output = self.flash_ansr_transformer.forward(beam.unsqueeze(0), data_tensor.unsqueeze(0), numeric_head=True)
                            #     numeric_prediction = num_output[0, :, 0][beam == self.tokenizer["<constant>"]]  # FIXME: Start at 1 or 0?

                        try:
                            refiner = Refiner(simplipy_engine=self.simplipy_engine, n_variables=self.n_variables).fit(
                                expression=beam_decoded,
                                X=X.cpu().numpy(),
                                y=y.cpu().numpy(),
                                n_restarts=self.n_restarts,
                                method=self.refiner_method,
                                p0=numeric_prediction,
                                p0_noise=self.refiner_p0_noise,
                                p0_noise_kwargs=self.refiner_p0_noise_kwargs,
                                converge_error=converge_error)

                            if not refiner.valid_fit:  # Fit failed
                                fvu = np.nan
                                score = np.nan
                            else:
                                if y.shape[0] == 1:
                                    # Cannot compute variance for a single sample. Use loss instead.
                                    fvu = refiner._all_constants_values[0][-1]
                                else:
                                    fvu = refiner._all_constants_values[0][-1] / np.clip(y_variance, np.finfo(np.float32).eps, None)
                                score = np.log10(fvu) + self.parsimony * len(beam_decoded)

                            self._results.append({
                                'log_prob': log_prob,
                                'fvu': fvu,
                                'score': score,
                                'expression': beam_decoded,
                                'complexity': len(beam_decoded),
                                'target_complexity': complexity,
                                'numeric_prediction': numeric_prediction,
                                'raw_beam': raw_beam,
                                'beam': beam,    # type: ignore
                                'raw_beam_decoded': raw_beam_decoded,
                                'function': refiner.expression_lambda,
                                'refiner': refiner,
                                'fits': copy.deepcopy(refiner._all_constants_values),
                            })

                        except ConvergenceError:
                            if converge_error == 'print':
                                print(f"Failed to converge for beam: {beam_decoded}")

            # --- /INFERENCE ---

            self.compile_results(self.parsimony)

            np.seterr(**numpy_errors_before)

    def compile_results(self, parsimony: float) -> None:
        if not self._results:
            raise ConvergenceError("The optimization did not converge for any beam")

        self.initial_parsimony = self.parsimony
        self.parsimony = parsimony

        # Compute the new score for each result
        for result in self._results:
            if 'score' in result:
                # Recompute the score with the new parsimony coefficient
                result['score'] = np.log10(result['fvu']) + self.parsimony * len(result['expression'])

        # Sort the results by the best loss of each beam
        self._results = list(sorted(self._results, key=lambda x: (
            x['score'] if not np.isnan(x['score']) else float('inf'),
            np.isnan(x['score'])
        )))

        # Create a dataframe
        self.results = pd.DataFrame(self._results)

        # Explode the fits for each beam
        self.results = self.results.explode('fits')
        self.results['beam_id'] = self.results.index
        self.results.reset_index(drop=True, inplace=True)

        # Split the fit tuples into columns
        fits_columns = pd.DataFrame(self.results['fits'].tolist(), columns=['fit_constants', 'fit_covariances', 'fit_loss'])
        self.results = pd.concat([self.results, fits_columns], axis=1)
        self.results.drop(columns=['fits'], inplace=True)

    def predict(self, X: np.ndarray | torch.Tensor | pd.DataFrame, nth_best_beam: int = 0, nth_best_constants: int = 0) -> np.ndarray:
        '''
        Predict the target data using the fitted model.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor | pd.DataFrame
            The input data.
        nth_best_beam : int, optional
            The nth best beam to use, by default 0.
        nth_best_constants : int, optional
            The nth best constants to use for the given beam, by default 0.

        Returns
        -------
        np.ndarray
            The predicted target data.
        '''
        X = self._truncate_input(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = pad_input_set(X, self.n_variables)

        if len(self._results) == 0:
            raise ValueError("The model has not been fitted yet. Please call the fit method first.")

        return self._results[nth_best_beam]['refiner'].predict(X, nth_best_constants=nth_best_constants)

    def get_expression(self, nth_best_beam: int = 0, nth_best_constants: int = 0, return_prefix: bool = False, precision: int = 2, map_variables: bool = True, **kwargs: Any) -> list[str] | str:
        '''
        Get the nth best expression.

        Parameters
        ----------
        nth_best_beam : int, optional
            The nth best beam to use, by default 0.
        nth_best_constants : int, optional
            The nth best constants to use for the given beam, by default 0.
        return_prefix : bool, optional
            Whether to return the expression with the prefix, by default False.
        precision : int, optional
            The precision for rounding the constants, by default 2.
        map_variables : bool, optional
            Whether to map the variables to their specified names if possible, by default True.
        **kwargs : Any

        Returns
        -------
        list[str] | str
            The nth best expression.
        '''
        return self._results[nth_best_beam]['refiner'].transform(
            expression=self._results[nth_best_beam]['expression'],
            nth_best_constants=nth_best_constants,
            return_prefix=return_prefix,
            precision=precision,
            variable_mapping=self.variable_mapping if map_variables else None,
            **kwargs)

    def specialize(
            self,
            X: np.ndarray | torch.Tensor | pd.DataFrame,
            y: np.ndarray | torch.Tensor | pd.DataFrame | pd.Series,
            generation_config: GenerationConfig | None = None,
            numeric_head: bool = False,
            n_restarts: int = 8,
            refiner_method: Literal['curve_fit_lm', 'minimize_bfgs'] = 'curve_fit_lm',
            refiner_p0_noise: Literal['uniform', 'normal'] | None = 'uniform',
            refiner_p0_noise_kwargs: dict | None = {'low': -5, 'high': 5},
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            convergence_errors: Literal['raise', 'ignore', 'warn'] = 'ignore',
            parsimony: float = 0.05,
            optimizer: torch.optim.Optimizer | None = None,
            optimizer_kwargs: dict | None = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None | bool = None,
            lr_scheduler_kwargs: dict | None = None,
            n_iter: int = 1000,
            priority_queue_size: int = 16,
            entropy_weight: float = 0.01,
            gradient_norm_clip: float = 1.0,
            verbose: bool = False,
            debug_no_optimizer_step: bool = False) -> None:
        '''
        Specialize the model on the input data with Priority Queue Policy Gradient.
        '''
        # Defaults
        if isinstance(refiner_p0_noise_kwargs, dict):
            refiner_p0_noise_kwargs = copy.deepcopy(refiner_p0_noise_kwargs)

        if generation_config is None:
            generation_config = GenerationConfig(method='softmax_sampling', choices=128, top_p=0.9, top_k=0, temperature=1.0)

        agent = FlashANSR(
            simplipy_engine=self.simplipy_engine,
            flash_ansr_transformer=self.flash_ansr_transformer,
            tokenizer=self.tokenizer,
            generation_config=generation_config,
            numeric_head=numeric_head,
            n_restarts=n_restarts,
            refiner_method=refiner_method,
            refiner_p0_noise=refiner_p0_noise,
            refiner_p0_noise_kwargs=refiner_p0_noise_kwargs,
            numpy_errors=numpy_errors,
            parsimony=parsimony)

        assert id(self.flash_ansr_transformer) == id(agent.flash_ansr_transformer)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if optimizer is None:
            optimizer = OptimizerFactory.get_optimizer(
                'AdamWScheduleFree',
                agent.flash_ansr_transformer.parameters(),
                lr=optimizer_kwargs.get('lr', 1e-6),
                weight_decay=optimizer_kwargs.get('weight_decay', 0.0))

        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}

        if lr_scheduler is None:
            lr_scheduler = LRSchedulerFactory.get_scheduler(
                'Warmup',
                optimizer,
                min_lr=lr_scheduler_kwargs.get('min_lr', 0),
                max_lr=lr_scheduler_kwargs.get('max_lr', 1),
                warmup_steps=lr_scheduler_kwargs.get('warmup_steps', 100),
                total_steps=n_iter)

        # Set the device for training
        device = agent.flash_ansr_transformer.device

        # Data preparation
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if not isinstance(X, torch.Tensor):
            if isinstance(X, pd.DataFrame):
                x_tensor = torch.tensor(X.values, dtype=torch.float32, device=device)
            else:
                x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        else:
            x_tensor = X.to(device)
        if not isinstance(y, torch.Tensor):
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_tensor = torch.tensor(y.values, dtype=torch.float32, device=device)
            else:
                y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        else:
            y_tensor = y.to(device)
        X = pad_input_set(x_tensor, self.n_variables)
        data_tensor = torch.cat([x_tensor, y_tensor], dim=-1)

        self.specialize_history = []
        pbar = tqdm(range(n_iter), desc="Specializing", disable=not verbose)

        priority_queue_beams = []
        priority_queue_objective = []

        total_unique_generated: set[tuple[str]] = set()

        for _ in pbar:
            try:
                # Generate sequences and evaluate
                agent.flash_ansr_transformer.eval()
                agent.fit(X, y)

                statistics_lists = defaultdict(list)

                n_new = 0
                for _, df in agent.results.groupby('beam_id'):
                    # Check if the beam can be used
                    if not np.isfinite(df['fvu']).any():
                        continue

                    # Check if the beam is new
                    new_beam_candidate = tuple(df['raw_beam'].iloc[0])
                    if new_beam_candidate in priority_queue_beams:
                        continue

                    statistics_lists['fvu'].extend(df['fvu'])
                    statistics_lists['complexity'].append(df['complexity'].iloc[0])

                    total_unique_generated.add(new_beam_candidate)
                    priority_queue_beams.append(new_beam_candidate)
                    priority_queue_objective.append(np.nanmedian(df['score']))
                    n_new += 1

                if n_new == 0:
                    # Sample again
                    continue

                # Sort by reward
                sorted_indices = np.argsort(priority_queue_objective)
                priority_queue_beams = [priority_queue_beams[i] for i in sorted_indices][:priority_queue_size]
                priority_queue_objective = [priority_queue_objective[i] for i in sorted_indices][:priority_queue_size]

                statistics = {
                    'min_fvu': np.nanmin(statistics_lists['fvu']) if np.any(np.isfinite(statistics_lists['fvu'])) else np.nan,
                    'max_fvu': np.nanmax(statistics_lists['fvu']) if np.any(np.isfinite(statistics_lists['fvu'])) else np.nan,
                    'mean_fvu': np.nanmean(statistics_lists['fvu']) if np.any(np.isfinite(statistics_lists['fvu'])) else np.nan,
                    'min_complexity': np.nanmin(statistics_lists['complexity']),
                    'max_complexity': np.nanmax(statistics_lists['complexity']),
                    'mean_complexity': np.nanmean(statistics_lists['complexity']),
                    'min_queue_objective': np.nanmin(priority_queue_objective) if np.any(np.isfinite(priority_queue_objective)) else np.nan,
                    'max_queue_objective': np.nanmax(priority_queue_objective) if np.any(np.isfinite(priority_queue_objective)) else np.nan,
                    'mean_queue_objective': np.nanmean(priority_queue_objective) if np.any(np.isfinite(priority_queue_objective)) else np.nan,
                    'n_total': len(total_unique_generated),
                    'n_new': n_new,
                }

                # Padding sequences to same length
                max_length = max(len(beam) for beam in priority_queue_beams)
                padded_beams = [list(beam) + [agent.tokenizer['<pad>']] * (max_length - len(beam)) for beam in priority_queue_beams]

                beam_tensor = torch.tensor(padded_beams, dtype=torch.long).to(device)

                # Increase the log probability of the best expressions (in the priority queue)
                agent.flash_ansr_transformer.train()
                if hasattr(optimizer, 'train'):
                    optimizer.train()
                optimizer.zero_grad()

                logits, _ = agent.flash_ansr_transformer.forward(beam_tensor, data_tensor.unsqueeze(0).repeat(len(priority_queue_beams), 1, 1))
                log_probs = torch.log_softmax(logits, dim=-1)

                # Get log probs for actions taken
                taken_log_probs = log_probs.gather(2, beam_tensor[:, 1:].unsqueeze(-1)).squeeze(-1)

                # Compute masks for padding and sequence endings
                pad_mask = beam_tensor != agent.tokenizer['<pad>']

                # Increase average log likelihood (not weighted by reward)
                policy_loss = -torch.mean(taken_log_probs[pad_mask[:, 1:]])

                # Average loss over batch
                policy_loss /= len(priority_queue_beams)

                # Regularize the entropy of the taken actions (tokens) and ignore padding
                entropy = - torch.sum(torch.exp(log_probs) * log_probs, dim=-1) * pad_mask
                entropy = torch.mean(entropy)

                loss = policy_loss - entropy_weight * entropy

                loss.backward()
                gradient_norms = torch.nn.utils.clip_grad_norm_(agent.flash_ansr_transformer.parameters(), gradient_norm_clip)

                # Backprop and update
                if not debug_no_optimizer_step:
                    optimizer.step()
                    if lr_scheduler is not False:
                        lr_scheduler.step()  # type: ignore

                logs = {
                    'NLL': policy_loss.item(),
                    'H': entropy.item(),
                    'max_gradient_norm': torch.max(gradient_norms).item(),
                    **statistics,
                }

            except ConvergenceError:
                if convergence_errors == 'raise':
                    raise
                elif convergence_errors == 'warn':
                    warnings.warn("Convergence error occurred during training. Skipping iteration.")

                logs = {
                    'NLL': np.nan,
                    'H': np.nan,
                    'max_gradient_norm': np.nan,
                    **{k: np.nan for k in ['min_fvu', 'max_fvu', 'mean_fvu', 'min_complexity', 'max_complexity', 'mean_complexity', 'min_queue_objective', 'max_queue_objective', 'mean_queue_objective', 'n_total', 'n_new']},
                }
                pass

            self.specialize_history.append(logs)
            pbar.set_postfix({
                'NLL': f"{logs['NLL']:.2e}" if np.isfinite(logs['NLL']) else "N/A",
                'H': f"{logs['H']:.2e}" if np.isfinite(logs['H']) else "N/A",
                'Max Queue Objective': f"{np.nanmax(priority_queue_objective):.1f}" if len(priority_queue_objective) > 0 and np.any(np.isfinite(priority_queue_objective)) else "N/A",
                'Min Queue Objective': f"{np.nanmin(priority_queue_objective):.1f}" if len(priority_queue_objective) > 0 and np.any(np.isfinite(priority_queue_objective)) else "N/A",
                'Min FVU': f"{np.nanmin(agent.results['fvu']):.2e}" if len(agent.results) > 0 and np.any(np.isfinite(agent.results['fvu'])) else "N/A",
                'Explored': len(total_unique_generated),
                'Best Expression': agent.simplipy_engine.prefix_to_infix(agent.tokenizer.decode(priority_queue_beams[0], special_tokens='<constant>')) if len(priority_queue_beams) > 0 else "N/A",
            })

        pbar.close()

    def to(self, device: str) -> "FlashANSR":
        '''
        Move the model to a device.

        Parameters
        ----------
        device : str
            The device to move the model to.

        Returns
        -------
        FlashANSR
            The model on the new device.
        '''
        self.flash_ansr_transformer.to(device)
        return self

    def eval(self) -> "FlashANSR":
        '''
        Set the model to evaluation mode.

        Returns
        -------
        FlashANSR
            The model in evaluation mode.
        '''
        self.flash_ansr_transformer.eval()
        return self

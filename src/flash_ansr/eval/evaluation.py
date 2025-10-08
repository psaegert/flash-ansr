import time
from typing import Any
from collections import defaultdict
import warnings

import torch
import numpy as np

from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.data import FlashANSRDataset, FlashASNRPreprocessor
from flash_ansr.refine import ConvergenceError
from flash_ansr.utils import load_config


class Evaluation():
    '''
    Evaluate a Flash-ANSR model on a dataset.

    Parameters
    ----------
    n_support : int, optional
        Number of input points for each equation. Default is None (sampled from the dataset).
    noise_level : float, optional
        Noise level for the constant fitting in units of standard deviations of the target variable. Default is 0.0.
    beam_width : int, optional
        Number of beams for the beam search algorithm. Default is 1.
    n_restarts : int, optional
        Number of restarts for constant fitting. Default is 1.
    max_len : int, optional
        Maximum length of each beam in tokens. Default is 20.
    numeric_head : bool, optional
        Whether to use the numeric head for constant prediction. Default is False.
    equivalence_pruning : bool, optional
        Whether to use equivalence pruning in the beam search algorithm. Default is True.
    pointwise_close_criterion : float, optional
        Criterion for the pointwise close accuracy. Default is 0.95.
    pointwise_close_accuracy_rtol : float, optional
        Relative tolerance for the pointwise close accuracy. Default is 0.05.
    pointwise_close_accuracy_atol : float, optional
        Absolute tolerance for the pointwise close accuracy. Default is 0.001.
    refiner_method : str, optional
        The optimization method to use. One of
        - 'curve_fit_lm': Use the curve_fit method with the Levenberg-Marquardt algorithm
        - 'minimize_bfgs': Use the minimize method with the BFGS algorithm
    refiner_p0_noise : str, optional
        Noise distribution for the initial guess of the refiner. Default is 'normal'.
    refiner_p0_noise_kwargs : dict, optional
        Keyword arguments for the noise distribution of the initial guess of the refiner. Default is None.
    r2_close_criterion : float, optional
        R^2 Criterion for the R^2 close accuracy. Default is 0.95.
    parsimony : float, optional
        Parsimony coefficient applied when ranking the model results. Default is 0.05.
    device : str, optional
        Device to load the model. Default is 'cpu'.

    Notes
    -----
    For more information about the criteria for the pointwise close accuracy and the R^2 close accuracy, see the following paper:
    https://arxiv.org/abs/2106.06427
    '''
    def __init__(
            self,
            n_support: int | None = None,
            noise_level: float = 0.0,
            beam_width: int = 1,
            complexity: str | list[int | float] = 'none',
            preprocess: bool = False,
            device: str = 'cpu') -> None:

        self.n_support = n_support
        self.noise_level = noise_level
        self.beam_width = beam_width
        self.complexity = complexity
        self.preprocess = preprocess

        self.device = device

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "Evaluation":
        '''
        Create an Evaluation object from a configuration dictionary or a configuration file.

        Parameters
        ----------
        config : dict or str
            Configuration dictionary or path to the configuration file.

        Returns
        -------
        Evaluation
            The Evaluation object.
        '''
        config_ = load_config(config)

        if "evaluation" in config_.keys():
            config_ = config_["evaluation"]

        beams = None
        if 'beam_width' in config_.keys():
            beams = config_['beam_width']
        elif 'generation_config' in config_.keys():
            if 'beam_width' in config_['generation_config']['kwargs'].keys():
                beams = config_['generation_config']['kwargs']['beam_width']
            elif 'choices' in config_['generation_config']['kwargs'].keys():
                beams = config_['generation_config']['kwargs']['choices']

        if beams is None:
            raise ValueError('Beam width not found in the configuration.')

        return cls(
            n_support=config_["n_support"],
            noise_level=config_.get("noise_level", 0.0),
            beam_width=beams,
            complexity=config_.get("complexity", 'none'),
            preprocess=config_.get("preprocess", False),
            device=config_["device"]
        )

    def evaluate(
            self,
            model: FlashANSR,
            dataset: FlashANSRDataset,
            size: int | None = None,
            verbose: bool = True) -> dict[str, Any]:
        '''
        Evaluate the model on the dataset.

        Parameters
        ----------
        model : FlashANSR
            The model to evaluate.
        size : int, optional
            Number of samples to evaluate. Default is None.
        verbose : bool, optional
            Whether to print the progress. Default is True.

        Returns
        -------
        dict
            Dictionary with the evaluation results.
        '''
        if verbose:
            print(
                'Evaluating model with configuration: '
                f'model.parsimony={model.parsimony}, noise_level={self.noise_level}, '
                f'n_support={self.n_support}, complexity={self.complexity}'
            )

        model.to(self.device).eval()

        results_dict = defaultdict(list)

        if size is None:
            size = len(dataset.skeleton_pool)

        dataset.skeleton_pool.sample_strategy["max_tries"] = 100

        print('Recompiling skeleton and holdout codes to ensure compatibility...')
        dataset.skeleton_pool.simplipy_engine = model.simplipy_engine
        dataset.skeleton_pool.skeleton_codes = dataset.skeleton_pool.compile_codes(verbose=verbose)
        for holdout_pool in dataset.skeleton_pool.holdout_pools:
            holdout_pool.simplipy_engine = model.simplipy_engine
            holdout_pool.skeleton_codes = holdout_pool.compile_codes(verbose=verbose)

        if self.preprocess:
            dataset.preprocessor = FlashASNRPreprocessor(model.simplipy_engine, format_probs={'complexity': 1.0})

        max_n_support = dataset.skeleton_pool.n_support_prior_config['kwargs']['max_value'] * 2

        warnings.filterwarnings('ignore', category=RuntimeWarning)

        with torch.no_grad():
            collected = 0
            iterator = dataset.iterate(
                size=size * 2,  # In case something goes wrong in a few samples, we have enough buffer to still collect 'size' samples
                max_n_support=max_n_support,
                n_support=self.n_support * 2 if self.n_support is not None else None,
                preprocess=self.preprocess,
                verbose=verbose,
                batch_size=1,
                tqdm_description='Evaluating',
                tqdm_total=size,
            )

            if verbose:
                print(f'Starting evaluation on {size} samples...')

            for batch in iterator:
                batch = dataset.collate(batch, device=self.device)

                n_support = self.n_support
                if n_support is None:
                    n_support = batch['x_tensors'].shape[1] // 2

                if n_support == 0:
                    warnings.warn('n_support evaluated to zero. Skipping batch.')
                    continue

                if self.noise_level > 0.0:
                    batch['y_tensors_noisy'] = batch['y_tensors'] + (
                        self.noise_level * batch['y_tensors'].std() * torch.randn_like(batch['y_tensors'])
                    )
                    if not torch.all(torch.isfinite(batch['y_tensors_noisy'])):
                        warnings.warn('Adding noise to the target variable resulted in non-finite values. Skipping this sample.')
                        continue
                else:
                    batch['y_tensors_noisy'] = batch['y_tensors']

                x_numpy = batch['x_tensors'].cpu().numpy()[0]
                y_numpy = batch['y_tensors'].cpu().numpy()[0]
                y_noisy_numpy = batch['y_tensors_noisy'].cpu().numpy()[0]

                X = x_numpy[:n_support]
                y = y_noisy_numpy[:n_support]

                X_val = x_numpy[n_support:]
                y_val = y_noisy_numpy[n_support:]

                sample_results = {
                    'skeleton': batch['skeleton'][0],
                    'skeleton_hash': batch['skeleton_hash'][0],
                    'expression': batch['expression'][0],
                    'input_ids': batch['input_ids'][0].cpu().numpy(),
                    'labels': batch['labels'][0].cpu().numpy(),
                    'constants': [c.cpu().numpy() for c in batch['constants'][0]],
                    'x': X,
                    'y': y_numpy[:n_support],
                    'y_noisy': y,
                    'x_val': X_val,
                    'y_val': y_numpy[n_support:],
                    'y_noisy_val': y_val,
                    'n_support': n_support,
                    'labels_decoded': dataset.tokenizer.decode(batch['labels'][0].cpu().tolist(), special_tokens='<constant>'),
                    'parsimony': model.parsimony,

                    'fit_time': None,
                    'predicted_expression': None,
                    'predicted_expression_prefix': None,
                    'predicted_expression_simplified': None,
                    'predicted_expression_encoded': None,
                    'predicted_constants': None,
                    'predicted_score': None,
                    'predicted_log_prob': None,
                    'y_pred': None,
                    'y_pred_val': None,
                    'prediction_success': False,
                    'error': None,
                }

                error_occured = False

                fit_time_start = time.time()
                try:
                    if self.complexity == 'none':
                        model.fit(X, y)
                    elif self.complexity == 'ground_truth':
                        model.fit(X, y, complexity=batch['complexities'])
                    elif isinstance(self.complexity, list):
                        model.fit(X, y, complexity=self.complexity)
                    else:
                        raise NotImplementedError(f'Complexity {self.complexity} not implemented yet.')
                    fit_time = time.time() - fit_time_start
                    sample_results['fit_time'] = fit_time
                    sample_results['prediction_success'] = True
                except (ConvergenceError, OverflowError, TypeError, ValueError) as exc:
                    warnings.warn(f'Error while fitting the model: {exc}. Filling nan.')
                    error_occured = True
                    sample_results['error'] = str(exc)

                if not error_occured:
                    if not model._results:
                        warnings.warn('Model produced no results. Filling nan.')
                        error_occured = True
                        sample_results['error'] = 'Model produced no results.'

                    best_result = model._results[0]

                if not error_occured:
                    try:
                        y_pred = model.predict(X, nth_best_beam=0, nth_best_constants=0)
                        if X_val.shape[0] > 0:
                            y_pred_val = model.predict(X_val, nth_best_beam=0, nth_best_constants=0)
                        else:
                            y_pred_val = np.empty_like(y_val)
                        sample_results['y_pred'] = y_pred
                        sample_results['y_pred_val'] = y_pred_val
                    except (ConvergenceError, ValueError) as exc:
                        warnings.warn(f'Error while predicting: {exc}. Filling nan.')
                        error_occured = True
                        sample_results['error'] = str(exc)

                    predicted_expression_readable = model.get_expression(
                        nth_best_beam=0,
                        nth_best_constants=0,
                        map_variables=True
                    )
                    predicted_expression_prefix = model.get_expression(
                        nth_best_beam=0,
                        nth_best_constants=0,
                        return_prefix=True,
                        map_variables=False
                    )

                    predicted_expression_simplified = None
                    if isinstance(predicted_expression_prefix, list) and model.simplipy_engine.is_valid(predicted_expression_prefix):
                        predicted_expression_simplified = model.simplipy_engine.simplify(
                            predicted_expression_prefix,
                            max_pattern_length=4
                        )

                    sample_results['predicted_expression'] = predicted_expression_readable
                    sample_results['predicted_expression_prefix'] = predicted_expression_prefix
                    sample_results['predicted_expression_simplified'] = predicted_expression_simplified

                    predicted_constants = None
                    predicted_score = None
                    if best_result.get('fits'):
                        predicted_constants = best_result['fits'][0][0].tolist()
                    if 'score' in best_result:
                        predicted_score = best_result['score']

                    sample_results['predicted_constants'] = predicted_constants
                    sample_results['predicted_score'] = predicted_score
                    sample_results['predicted_log_prob'] = best_result.get('log_prob', None)

                    predicted_expression_encoded = None
                    if isinstance(predicted_expression_prefix, list):
                        predicted_expression_encoded = model.tokenizer.encode(predicted_expression_prefix, oov='unk')

                    sample_results['predicted_expression_encoded'] = predicted_expression_encoded

                for key, value in sample_results.items():
                    results_dict[key].append(value)

                collected += 1
                if collected >= size:
                    break

        if collected < size:
            warnings.warn(f'Only collected {collected} out of {size} requested samples.')

        # Sort the scores alphabetically by key
        results_dict = dict(sorted(dict(results_dict).items()))  # type: ignore

        return results_dict

import time
import os
import pickle
from typing import Any
from collections import defaultdict
import warnings

import torch
import numpy as np

from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.data import FlashANSRDataset, FlashANSRPreprocessor
from flash_ansr.refine import ConvergenceError
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.paths import substitute_root_path

from simplipy.utils import numbers_to_constant


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
    complexity : str or list of int/float, optional
        Complexity constraint for the generated equations. Can be 'none', 'ground_truth', or a list of complexity values. Default is 'none'.
    preprocess : bool, optional
        Whether to preprocess the data using FlashASNRPreprocessor. Default is False.
    device : str, optional
        Device to run the evaluation on. Default is 'cpu'.
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
            results_dict: dict[str, Any] | None = None,
            size: int | None = None,
            save_every: int | None = None,
            output_file: str | None = None,
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

        if results_dict is None:
            results_store: defaultdict[str, list[Any]] = defaultdict(list)
            existing_results = 0
        else:
            results_store = defaultdict(list)
            for key, value in results_dict.items():
                results_store[key] = list(value)

            lengths = {key: len(value) for key, value in results_store.items()}
            if lengths and len(set(lengths.values())) != 1:
                raise ValueError('Existing results_dict has inconsistent lengths.')

            existing_results = next(iter(lengths.values())) if lengths else 0

        if save_every is not None:
            if output_file is None:
                raise ValueError('output_file must be provided when save_every is set.')
            resolved_output_file = substitute_root_path(output_file)
            output_dir = os.path.dirname(resolved_output_file)
            os.makedirs(output_dir, exist_ok=True)

        if size is None:
            size = len(dataset.skeleton_pool) - existing_results

        if size is not None and size <= 0:
            if size < 0:
                warnings.warn(
                    f'Requested evaluation size is smaller than the number of existing results ({existing_results}). '
                    'Returning existing results without additional evaluation.'
                )
            return dict(sorted(dict(results_store).items()))  # type: ignore

        results_dict = results_store

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
            dataset.preprocessor = FlashANSRPreprocessor(
                simplipy_engine=model.simplipy_engine,
                tokenizer=model.tokenizer,
            )

        base_max_n_support = dataset.skeleton_pool.support_sampler.configured_max_n_support
        if base_max_n_support is None and self.n_support is None:
            raise ValueError(
                "Support sampler configuration must define a maximum support size when evaluation does not "
                "override 'n_support'."
            )

        max_n_support = (
            base_max_n_support * 2 if self.n_support is None else self.n_support * 2
        )

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
                tqdm_kwargs={'desc': 'Evaluating', 'total': size, 'smoothing': 0.0},
                tokenizer_oov='unk'  # Do not raise an error if an unknown token (operator) is encountered
            )

            if verbose:
                print(f'Starting evaluation on {size} problems...')

            for batch_id, batch in enumerate(iterator):
                batch = dataset.collate(batch, device=self.device)

                # Remove padding (not needed for single sample batches, and will interfere with fitting)
                batch['x_tensors'] = batch['x_tensors'][0][:batch['n_support'][0]]
                batch['y_tensors'] = batch['y_tensors'][0][:batch['n_support'][0]]

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

                x_numpy = batch['x_tensors'].cpu().numpy()
                y_numpy = batch['y_tensors'].cpu().numpy()
                y_noisy_numpy = batch['y_tensors_noisy'].cpu().numpy()

                X = x_numpy[:n_support]
                y = y_noisy_numpy[:n_support]

                X_val = x_numpy[n_support:]
                y_val = y_noisy_numpy[n_support:]

                sample_results = {
                    'skeleton': list(batch['skeleton'][0]),
                    'skeleton_hash': tuple(batch['skeleton_hash'][0]) if isinstance(batch['skeleton_hash'][0], (list, tuple)) else batch['skeleton_hash'][0],
                    'expression': list(batch['expression'][0]),
                    'input_ids': batch['input_ids'][0].cpu().numpy().copy(),
                    'labels': batch['labels'][0].cpu().numpy().copy(),
                    'constants': [c.cpu().numpy().copy() for c in batch['constants'][0]],
                    'x': X.copy(),
                    'y': y_numpy[:n_support].copy(),
                    'y_noisy': y.copy(),
                    'x_val': X_val.copy(),
                    'y_val': y_numpy[n_support:].copy(),
                    'y_noisy_val': y_val.copy(),
                    'n_support': int(n_support),
                    'labels_decoded': list(dataset.tokenizer.decode(batch['labels'][0].cpu().tolist(), special_tokens='<constant>')),
                    'parsimony': model.parsimony,
                    'noise_level': self.noise_level,

                    'fit_time': None,
                    'predicted_expression': None,
                    'predicted_expression_prefix': None,
                    'predicted_skeleton_prefix': None,
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
                        model.fit(X, y, complexity=batch['complexity'])
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
                        sample_results['y_pred'] = y_pred.copy()
                        sample_results['y_pred_val'] = y_pred_val.copy()
                    except (ConvergenceError, ValueError) as exc:
                        warnings.warn(f'Error while predicting: {exc}. Filling nan.')
                        error_occured = True
                        sample_results['error'] = str(exc)

                    predicted_expression_readable = model.get_expression(
                        nth_best_beam=0,
                        nth_best_constants=0,
                        map_variables=True
                    )
                    predicted_skeleton_prefix = model.get_expression(
                        nth_best_beam=0,
                        nth_best_constants=0,
                        return_prefix=True,
                        map_variables=False
                    )

                    sample_results['predicted_expression'] = predicted_expression_readable
                    sample_results['predicted_expression_prefix'] = predicted_skeleton_prefix.copy()
                    sample_results['predicted_skeleton_prefix'] = numbers_to_constant(predicted_skeleton_prefix).copy()

                    predicted_constants = None
                    predicted_score = None
                    if best_result.get('fits'):
                        predicted_constants = best_result['fits'][0][0].tolist()
                    if 'score' in best_result:
                        predicted_score = best_result['score']

                    sample_results['predicted_constants'] = predicted_constants
                    sample_results['predicted_score'] = predicted_score
                    sample_results['predicted_log_prob'] = best_result.get('log_prob', None)

                for key, value in sample_results.items():
                    results_dict[key].append(value)

                if save_every is not None and resolved_output_file is not None and (batch_id + 1) % save_every == 0:
                    with open(resolved_output_file, 'wb') as f:
                        pickle.dump(results_dict, f)

                collected += 1
                if collected >= size:
                    break

        if collected < size:
            warnings.warn(f'Only collected {collected} out of {size} requested samples.')

        # Sort the scores alphabetically by key
        results_sorted = dict(sorted(dict(results_dict).items()))  # type: ignore

        if save_every is not None and resolved_output_file is not None:
            if verbose:
                print('Saving final evaluation results...')
            with open(resolved_output_file, 'wb') as f:
                pickle.dump(results_sorted, f)

        return results_sorted

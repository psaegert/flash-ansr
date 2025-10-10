from typing import Any, Callable
from collections import defaultdict
import warnings
import time

import torch
import numpy as np
from sympy import lambdify

from simplipy import SimpliPyEngine
from simplipy.utils import numbers_to_constant

from flash_ansr import FlashANSRDataset
from flash_ansr.utils import load_config

from nesymres.architectures.model import Model  # type: ignore[import]


class NeSymReSEvaluation():
    def __init__(
            self,
            n_support: int | None = None,
            device: str = 'cpu') -> None:

        self.n_support = n_support

        self.device = device

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "NeSymReSEvaluation":
        config_ = load_config(config)

        if "evaluation" in config_.keys():
            config_ = config_["evaluation"]

        return cls(
            n_support=config_["n_support"],
            device=config_["device"]
        )

    def evaluate(
            self,
            model: Model,
            fitfunc: Callable,
            simplipy_engine: SimpliPyEngine,
            dataset: FlashANSRDataset,
            size: int | None = None,
            verbose: bool = True) -> dict[str, Any]:

        model.to(self.device).eval()

        results_dict = defaultdict(list)

        if size is None:
            size = len(dataset.skeleton_pool)

        dataset.skeleton_pool.sample_strategy["max_tries"] = 100
        max_n_support = dataset.skeleton_pool.n_support_prior_config['kwargs']['max_value'] * 2

        with torch.no_grad():
            collected = 0
            iterator = dataset.iterate(
                size=size * 2,
                max_n_support=max_n_support,
                n_support=self.n_support * 2 if self.n_support is not None else None,
                verbose=verbose,
                batch_size=1,
                tqdm_description='Evaluating',
                tqdm_total=size,
            )

            if verbose:
                print(f'Starting evaluation on {size} problems...')

            for batch in iterator:
                batch = dataset.collate(batch, device=self.device)

                n_support = self.n_support
                if n_support is None:
                    n_support = batch['x_tensors'].shape[1] // 2

                if n_support == 0:
                    warnings.warn('n_support evaluated to zero. Skipping batch.')
                    continue

                x_numpy = batch['x_tensors'].cpu().numpy()[0]
                y_numpy = batch['y_tensors'].cpu().numpy()[0]

                X = x_numpy[:n_support]
                y = y_numpy[:n_support]
                y_fit = y.reshape(-1)

                X_val = x_numpy[n_support:]
                y_val = y_numpy[n_support:]

                labels = batch['labels'][0].clone()
                labels_decoded = dataset.tokenizer.decode(labels.tolist(), special_tokens='<constant>')

                sample_results: dict[str, Any] = {
                    'skeleton': batch['skeleton'][0],
                    'skeleton_hash': batch['skeleton_hash'][0],
                    'expression': batch['expression'][0],
                    'input_ids': batch['input_ids'][0].cpu().numpy(),
                    'labels': batch['labels'][0].cpu().numpy(),
                    'constants': [c.cpu().numpy() for c in batch['constants'][0]],
                    'x': X,
                    'y': y,
                    'y_noisy': y,
                    'x_val': X_val,
                    'y_val': y_val,
                    'y_noisy_val': y_val,
                    'n_support': n_support,
                    'labels_decoded': labels_decoded,
                    'parsimony': getattr(model, 'parsimony', None),
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

                try:
                    fit_time_start = time.time()
                    nesymres_output = fitfunc(X, y_fit)
                    sample_results['fit_time'] = time.time() - fit_time_start
                    sample_results['prediction_success'] = True
                except Exception as exc:  # pragma: no cover - defensive safety
                    warnings.warn(f'Error while fitting the model: {exc}. Filling nan.')
                    sample_results['error'] = str(exc)
                    error_occured = True

                if not error_occured:
                    try:
                        predicted_expr = nesymres_output['best_bfgs_preds'][0]
                        predicted_expression = str(predicted_expr)
                        sample_results['predicted_expression'] = predicted_expression
                        predicted_prefix = simplipy_engine.infix_to_prefix(predicted_expression)
                        sample_results['predicted_expression_prefix'] = predicted_prefix
                        sample_results['predicted_skeleton_prefix'] = numbers_to_constant(predicted_prefix)
                        if nesymres_output.get('best_bfgs_consts') is not None:
                            predicted_constants = nesymres_output['best_bfgs_consts'][0]
                            if isinstance(predicted_constants, np.ndarray):
                                sample_results['predicted_constants'] = predicted_constants.tolist()
                            elif isinstance(predicted_constants, (list, tuple)):
                                sample_results['predicted_constants'] = list(predicted_constants)
                            else:
                                sample_results['predicted_constants'] = predicted_constants
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        warnings.warn(f'Error while parsing NeSymReS output: {exc}. Filling nan.')
                        sample_results['error'] = str(exc)
                        sample_results['prediction_success'] = False
                        error_occured = True

                if not error_occured:
                    try:
                        var_symbols = [f'x_{idx + 1}' for idx in range(X.shape[1])]
                        evaluate_expression = lambdify(var_symbols, predicted_expr, 'numpy')

                        y_pred = np.asarray(evaluate_expression(*X.T), dtype=float).reshape(-1, 1)

                        if X_val.size > 0:
                            y_pred_val = np.asarray(evaluate_expression(*X_val.T), dtype=float).reshape(-1, 1)
                        else:
                            y_pred_val = np.empty_like(y_val)

                        if y_pred.shape != y.shape:
                            raise ValueError(f"Shape of y_pred {y_pred.shape} does not match shape of y {y.shape}.")
                        if y_pred_val.shape != y_val.shape:
                            raise ValueError(f"Shape of y_pred_val {y_pred_val.shape} does not match shape of y_val {y_val.shape}.")

                        sample_results['y_pred'] = y_pred
                        sample_results['y_pred_val'] = y_pred_val
                    except (NameError, KeyError, ValueError, TypeError, OverflowError) as exc:
                        warnings.warn(f'Error while computing predictions: {exc}. Filling nan.')
                        sample_results['error'] = str(exc)
                        sample_results['prediction_success'] = False

                for key, value in sample_results.items():
                    results_dict[key].append(value)

                collected += 1
                if collected >= size:
                    break

        if collected < size:
            warnings.warn(f'Only collected {collected} out of {size} requested samples.')

        results_dict = dict(sorted(dict(results_dict).items()))  # type: ignore

        return results_dict

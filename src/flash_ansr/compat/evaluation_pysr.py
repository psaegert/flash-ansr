from pysr import PySRRegressor

from typing import Any
from collections import defaultdict
import warnings
import time
import pickle
import os

import torch

from simplipy import SimpliPyEngine

from flash_ansr import FlashANSRDataset
from flash_ansr.utils import load_config, substitute_root_path

import simplipy
from simplipy.utils import numbers_to_constant


class PySREvaluation():
    def __init__(
            self,
            n_support: int | None = None,
            noise_level: float = 0.0,
            timeout_in_seconds: int = 60,
            parsimony: float = 0.0,
            niterations: int = 100) -> None:

        self.n_support = n_support
        self.noise_level = noise_level
        self.timeout_in_seconds = timeout_in_seconds
        self.niterations = niterations
        self.parsimony = parsimony

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "PySREvaluation":
        config_ = load_config(config)

        if "evaluation" in config_.keys():
            config_ = config_["evaluation"]

        return cls(
            n_support=config_["n_support"],
            noise_level=config_.get("noise_level", 0.0),
            timeout_in_seconds=config_["timeout_in_seconds"],
            niterations=config_["niterations"],
        )

    def evaluate(
            self,
            dataset: FlashANSRDataset,
            simplipy_engine: SimpliPyEngine,
            results_dict: dict[str, Any] | None = None,
            size: int | None = None,
            save_every: int | None = None,
            output_file: str | None = None,
            verbose: bool = True) -> dict[str, Any]:

        if results_dict is None:
            results_dict = defaultdict(list)

        if save_every is not None:
            output_dir = os.path.dirname(substitute_root_path(output_file))
            os.makedirs(output_dir, exist_ok=True)

        if size is None:
            size = len(dataset.skeleton_pool)

        # HACK
        dataset.skeleton_pool.sample_strategy["max_tries"] = 100
        max_n_support = dataset.skeleton_pool.n_support_prior_config['kwargs']['max_value'] * 2

        model = PySRRegressor(
            temp_equation_file=True,
            delete_tempfiles=True,
            timeout_in_seconds=self.timeout_in_seconds,
            niterations=self.niterations,
            unary_operators=[
                'neg',
                'abs',
                'inv',
                'sin',
                'cos',
                'tan',
                'asin',
                'acos',
                'atan',
                'exp',
                'log',
                'pow2(x) = x^2',
                'pow3(x) = x^3',
                'pow4(x) = x^4',
                'pow5(x) = x^5',
                r'pow1_2(x::T) where {T} = x >= 0 ? T(x^(1/2)) : T(NaN)',
                r'pow1_3(x::T) where {T} = x >= 0 ? T(x^(1/3)) : T(-((-x)^(1/3)))',
                r'pow1_4(x::T) where {T} = x >= 0 ? T(x^(1/4)) : T(NaN)',
                r'pow1_5(x::T) where {T} = x >= 0 ? T(x^(1/5)) : T(-((-x)^(1/5)))',
                'mult2(x) = 2*x',
                'mult3(x) = 3*x',
                'mult4(x) = 4*x',
                'mult5(x) = 5*x',
                'div2(x) = x/2',
                'div3(x) = x/3',
                'div4(x) = x/4',
                'div5(x) = x/5',
            ],
            binary_operators=['+', '-', '*', '/', '^'],
            extra_sympy_mappings={
                "pow2": simplipy.operators.pow2,
                "pow3": simplipy.operators.pow3,
                "pow4": simplipy.operators.pow4,
                "pow5": simplipy.operators.pow5,
                "pow1_2": simplipy.operators.pow1_2,
                "pow1_3": lambda x: x**(1 / 3),  # Workaround for https://stackoverflow.com/questions/68577498/sympy-typeerror-cannot-determine-truth-value-of-relational-how-to-make-sure-x
                "pow1_4": simplipy.operators.pow1_4,
                "pow1_5": lambda x: x**(1 / 5),
                "mult2": simplipy.operators.mult2,
                "mult3": simplipy.operators.mult3,
                "mult4": simplipy.operators.mult4,
                "mult5": simplipy.operators.mult5,
                "div2": simplipy.operators.div2,
                "div3": simplipy.operators.div3,
                "div4": simplipy.operators.div4,
                "div5": simplipy.operators.div5,
            },
            constraints={
                '^': (-1, 3)
            }
        )

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        with torch.no_grad():
            collected = 0
            iterator = dataset.iterate(
                size=size * 2,  # In case something goes wrong in a few samples, we have enough buffer to still collect 'size' samples
                max_n_support=max_n_support,
                n_support=self.n_support * 2 if self.n_support is not None else None,
                verbose=verbose,
                batch_size=1,
                tqdm_description='Evaluating',
                tqdm_total=size,
            )

            if verbose:
                print(f'Starting evaluation on {size} problems...')

            for batch_id, batch in enumerate(iterator):
                batch = dataset.collate(batch, device='cpu')

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
                    'parsimony': self.parsimony,

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

                fit_time_before = time.time()
                try:
                    model.fit(X, y)
                    sample_results['fit_time'] = time.time() - fit_time_before
                    sample_results['prediction_success'] = True
                except Exception as e:
                    error_occured = True
                    sample_results['error'] = str(e)

                if not error_occured:
                    y_pred = model.predict(X).reshape(-1, 1)
                    y_pred_val = model.predict(X_val).reshape(-1, 1)

                    if not y_pred.shape == y.shape:
                        raise ValueError(f"Shape of y_pred {y_pred.shape} does not match shape of y {y.shape}.")
                    if not y_pred_val.shape == y_val.shape:
                        raise ValueError(f"Shape of y_pred_val {y_pred_val.shape} does not match shape of y_val {y_val.shape}.")

                    sample_results['y_pred'] = y_pred
                    sample_results['y_pred_val'] = y_pred_val

                    predicted_expression = str(model.get_best()['equation'])
                    sample_results['predicted_expression'] = predicted_expression
                    sample_results['predicted_expression_prefix'] = dataset.simplipy_engine.infix_to_prefix(predicted_expression)
                    sample_results['predicted_skeleton_prefix'] = numbers_to_constant(sample_results['predicted_expression_prefix'])

                for key, value in sample_results.items():
                    results_dict[key].append(value)

                if not len(set(len(v) for v in results_dict.values())) == 1:
                    print({k: len(v) for k, v in results_dict.items()})  # Check that all lists have the same length
                    raise ValueError("Not all lists in results_dict have the same length.")

                time.sleep(0.1)  # For good measure

                if save_every is not None and (batch_id + 1) % save_every == 0:
                    if verbose:
                        print(f"Saving intermediate results after {batch_id + 1} batches ...")

                    with open(substitute_root_path(output_file), 'wb') as f:
                        pickle.dump(results_dict, f)

                collected += 1
                if collected >= size:
                    break

        # Sort the scores alphabetically by key
        results_dict = dict(sorted(dict(results_dict).items()))  # type: ignore

        return results_dict

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
from flash_ansr.model.tokenizer import Tokenizer

import simplipy


class PySREvaluation():
    def __init__(
            self,
            n_support: int | None = None,
            timeout_in_seconds: int = 60,
            niterations: int = 100,
            pointwise_close_criterion: float = 0.95,
            pointwise_close_accuracy_rtol: float = 0.05,
            pointwise_close_accuracy_atol: float = 0.001,
            r2_close_criterion: float = 0.95) -> None:

        self.n_support = n_support
        self.timeout_in_seconds = timeout_in_seconds
        self.niterations = niterations
        self.pointwise_close_criterion = pointwise_close_criterion
        self.pointwise_close_accuracy_rtol = pointwise_close_accuracy_rtol
        self.pointwise_close_accuracy_atol = pointwise_close_accuracy_atol
        self.r2_close_criterion = r2_close_criterion

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "PySREvaluation":
        config_ = load_config(config)

        if "evaluation" in config_.keys():
            config_ = config_["evaluation"]

        return cls(
            n_support=config_["n_support"],
            timeout_in_seconds=config_["timeout_in_seconds"],
            niterations=config_["niterations"],
            pointwise_close_criterion=config_["pointwise_close_criterion"],
            pointwise_close_accuracy_rtol=config_["pointwise_close_accuracy_rtol"],
            pointwise_close_accuracy_atol=config_["pointwise_close_accuracy_atol"],
            r2_close_criterion=config_["r2_close_criterion"]
        )

    def evaluate(
            self,
            dataset: FlashANSRDataset,
            simplipy_engine: SimpliPyEngine,
            tokenizer: Tokenizer,
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
            unary_operators=['neg', 'abs', 'inv', 'sin', 'cos', 'tan', 'atan', 'exp', 'log'],
            binary_operators=['+', '-', '*', '/'],
            extra_sympy_mappings={
                "pow": simplipy.operators.pow,  # type: ignore
                "pow2": simplipy.operators.pow2,  # type: ignore
                "pow3": simplipy.operators.pow3,  # type: ignore
                "pow4": simplipy.operators.pow4,  # type: ignore
                "pow5": simplipy.operators.pow5,  # type: ignore
                "pow1_2": simplipy.operators.pow1_2,  # type: ignore
                "pow1_3": simplipy.operators.pow1_3,  # type: ignore
                "pow1_4": simplipy.operators.pow1_4,  # type: ignore
                "pow1_5": simplipy.operators.pow1_5,  # type: ignore
                "inv": simplipy.operators.inv,  # type: ignore
                "asin": simplipy.operators.asin,  # Prevents Julia out of bounds error
                "acos": simplipy.operators.asin,  # Prevents Julia out of bounds error
                "mult2": simplipy.operators.mult2,  # type: ignore
                "mult3": simplipy.operators.mult3,  # type: ignore
                "mult4": simplipy.operators.mult4,  # type: ignore
                "mult5": simplipy.operators.mult5,  # type: ignore
                "div2": simplipy.operators.div2,  # type: ignore
                "div3": simplipy.operators.div3,  # type: ignore
                "div4": simplipy.operators.div4,  # type: ignore
                "div5": simplipy.operators.div5,  # type: ignore
            },
        )

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        with torch.no_grad():
            for batch_id, batch in enumerate(dataset.iterate(size=size, max_n_support=max_n_support, n_support=max_n_support, verbose=verbose)):
                batch = dataset.collate(batch, device='cpu')

                x_tensor = batch['x_tensors']
                y_tensor = batch['y_tensors']

                X = x_tensor.cpu().numpy()[0, :self.n_support]
                y = y_tensor.cpu().numpy()[0, :self.n_support, 0]

                X_val = x_tensor.cpu().numpy()[0, self.n_support:]
                y_val = y_tensor.cpu().numpy()[0, self.n_support:, 0]

                labels = batch['labels'][0].clone()

                results_dict['skeleton'].append(batch['skeleton'][0])
                results_dict['skeleton_hash'].append(batch['skeleton_hash'][0])
                results_dict['expression'].append(batch['expression'][0])

                results_dict['input_ids'].append(batch['input_ids'][0].cpu().numpy())
                results_dict['labels'].append(labels.cpu().numpy())
                results_dict['constants'].append([c.cpu().numpy() for c in batch['constants']])

                results_dict['x'].append(X)
                results_dict['y'].append(y)

                results_dict['x_val'].append(X_val)
                results_dict['y_val'].append(y_val)

                results_dict['n_support'].append(self.n_support)

                # Create the labels for the next token prediction task (i.e. shift the input_ids by one position to the right)
                labels_decoded = tokenizer.decode(labels.tolist(), special_tokens='<constant>')
                results_dict['labels_decoded'].append(labels_decoded)

                fit_time_before = time.time()
                model.fit(X, y)
                results_dict['fit_time'].append(time.time() - fit_time_before)

                best_skeleton_decoded = []
                for token in simplipy_engine.parse(str(model.get_best()['equation'])):
                    try:
                        float(token)
                        best_skeleton_decoded.append('<constant>')
                    except ValueError:
                        best_skeleton_decoded.append(token)
                results_dict['best_skeleton_decoded'].append(best_skeleton_decoded)

                if dataset.simplipy_engine.is_valid(best_skeleton_decoded):
                    best_skeleton_decoded = dataset.simplipy_engine.simplify(best_skeleton_decoded, max_pattern_length=4)
                results_dict['best_skeleton_simplified_decoded'].append(best_skeleton_decoded)

                best_skeleton = tokenizer.encode(best_skeleton_decoded, oov='unk')
                results_dict['best_skeleton'].append(best_skeleton)

                y_pred = model.predict(X)
                y_pred_val = model.predict(X_val)

                if not y_pred.shape == y.shape:
                    raise ValueError(f"Shape of y_pred {y_pred.shape} does not match shape of y {y.shape}.")
                if not y_pred_val.shape == y_val.shape:
                    raise ValueError(f"Shape of y_pred_val {y_pred_val.shape} does not match shape of y_val {y_val.shape}.")

                results_dict['y_pred'].append(y_pred)
                results_dict['y_pred_val'].append(y_pred_val)

                if not len(set(len(v) for v in results_dict.values())) == 1:
                    print({k: len(v) for k, v in results_dict.items()})  # Check that all lists have the same length
                    raise ValueError("Not all lists in results_dict have the same length.")

                time.sleep(0.1)  # For good measure

                if save_every is not None and (batch_id + 1) % save_every == 0:
                    if verbose:
                        print(f"Saving intermediate results after {batch_id + 1} batches ...")

                    with open(substitute_root_path(output_file), 'wb') as f:
                        pickle.dump(results_dict, f)

        # Sort the scores alphabetically by key
        results_dict = dict(sorted(dict(results_dict).items()))  # type: ignore

        return results_dict

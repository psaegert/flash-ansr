from typing import Any
from collections import defaultdict
import warnings
import time

import torch
import numpy as np
import editdistance

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from pysr import PySRRegressor

from simplipy import SimpliPyEngine

from flash_ansr import FlashANSRDataset
from flash_ansr.utils import load_config
from flash_ansr.eval.token_prediction import (
    accuracy,
    precision,
    recall,
    f1_score,
)
from flash_ansr.model.transformer import Tokenizer
from flash_ansr.eval.utils import NoOpStemmer
from flash_ansr.eval.sequences import zss_tree_edit_distance
import nsrops


nltk.download('wordnet', quiet=True)


class PySREvaluation():
    def __init__(
            self,
            n_support: int | None = None,
            timeout_in_seconds: int = 60,
            pointwise_close_criterion: float = 0.95,
            pointwise_close_accuracy_rtol: float = 0.05,
            pointwise_close_accuracy_atol: float = 0.001,
            r2_close_criterion: float = 0.95) -> None:

        self.n_support = n_support
        self.timeout_in_seconds = timeout_in_seconds
        self.pointwise_close_criterion = pointwise_close_criterion
        self.pointwise_close_accuracy_rtol = pointwise_close_accuracy_rtol
        self.pointwise_close_accuracy_atol = pointwise_close_accuracy_atol
        self.r2_close_criterion = r2_close_criterion

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        self.rouge_scorer._tokenizer.tokenize = lambda x: x

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "PySREvaluation":
        config_ = load_config(config)

        if "evaluation" in config_.keys():
            config_ = config_["evaluation"]

        return cls(
            n_support=config_["n_support"],
            timeout_in_seconds=config_["timeout_in_seconds"],
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
            size: int | None = None,
            verbose: bool = True) -> dict[str, Any]:

        results_dict = defaultdict(list)

        if size is None:
            size = len(dataset.skeleton_pool)

        # HACK
        dataset.skeleton_pool.sample_strategy["max_tries"] = 100

        with torch.no_grad():
            for batch in dataset.iterate(size=size, n_support=self.n_support * 2 if self.n_support is not None else None, verbose=verbose, avoid_fragmentation=False):

                # Initialize here to prevent memory leak?
                model = PySRRegressor(
                    temp_equation_file=True,
                    delete_tempfiles=True,
                    timeout_in_seconds=self.timeout_in_seconds,
                    unary_operators=['neg', 'abs', 'inv', 'sin', 'cos', 'tan', 'atan', 'exp', 'log'],
                    binary_operators=['+', '-', '*', '/'],
                    extra_sympy_mappings={
                        "pow2": nsrops.pow2,  # type: ignore
                        "pow3": nsrops.pow3,  # type: ignore
                        "pow4": nsrops.pow4,  # type: ignore
                        "pow5": nsrops.pow5,  # type: ignore
                        "pow1_2": nsrops.pow1_2,  # type: ignore
                        "pow1_3": nsrops.pow1_3,  # type: ignore
                        "pow1_4": nsrops.pow1_4,  # type: ignore
                        "pow1_5": nsrops.pow1_5,  # type: ignore
                        "inv": nsrops.inv,  # type: ignore
                        "asin": lambda x: np.arcsin(x) if -1 <= x <= 1 else np.nan,  # Prevents Julia out of bounds error
                        "acos": lambda x: np.arccos(x) if -1 <= x <= 1 else np.nan,
                    },
                )

                batch = dataset.collate(batch, device='cpu')

                x_tensor = batch['x_tensors'].unsqueeze(0)
                y_tensor = batch['y_tensors'].unsqueeze(0)

                X = x_tensor.cpu().numpy()[0, :self.n_support]
                y = y_tensor.cpu().numpy()[0, :self.n_support, 0]

                X_val = x_tensor.cpu().numpy()[0, self.n_support:]
                y_val = y_tensor.cpu().numpy()[0, self.n_support:, 0]

                results_dict['input_ids'].append(batch['input_ids'].cpu().numpy())
                results_dict['labels'].append(batch['labels'].cpu().numpy())
                results_dict['constants'].append([c.cpu().numpy() for c in batch['constants']])

                results_dict['x'].append(x_tensor.cpu().numpy()[:, :self.n_support])
                results_dict['y'].append(y_tensor.cpu().numpy()[:, :self.n_support])

                results_dict['x_val'].append(x_tensor.cpu().numpy()[:, self.n_support:])
                results_dict['y_val'].append(y_tensor.cpu().numpy()[:, self.n_support:])

                results_dict['n_support'].append([x_tensor.shape[1] // 2] * x_tensor.shape[0])

                # Create the labels for the next token prediction task (i.e. shift the input_ids by one position to the right)
                labels = batch['labels'][0].clone()
                labels_decoded = tokenizer.decode(batch['labels'].tolist(), special_tokens='<constant>')

                # TODO: For different datasets, sort unused dimensions to the end
                warnings.filterwarnings("ignore", category=RuntimeWarning)

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
                best_skeleton = tokenizer.encode(best_skeleton_decoded, oov='unk')

                # Accuracy, precision, recall, F1 score
                best_skeleton_tensor = torch.tensor(best_skeleton).unsqueeze(0)
                results_dict['recall_beam_1'].extend(recall(best_skeleton_tensor, labels.view(1, -1), ignore_index=0, reduction='none').cpu())
                results_dict['precision_beam_1'].extend(precision(best_skeleton_tensor, labels.view(1, -1), ignore_index=0, reduction='none').cpu())
                results_dict['f1_score_beam_1'].extend(f1_score(best_skeleton_tensor, labels.view(1, -1), ignore_index=0, reduction='none').cpu())
                results_dict['accuracy_beam_1'].extend(accuracy(best_skeleton_tensor, labels.view(1, -1), ignore_index=0, reduction='none').cpu())

                # BLEU
                results_dict['bleu_beam_1'].append(sentence_bleu(references=[labels_decoded], hypothesis=best_skeleton_decoded, smoothing_function=SmoothingFunction().method1))

                # ROUGE
                rouge = self.rouge_scorer.score(best_skeleton_decoded, labels_decoded)

                for metric in ['rouge1', 'rouge2', 'rougeL']:
                    results_dict[f'{metric}_precision_beam_1'].append(rouge[metric].precision)
                    results_dict[f'{metric}_recall_beam_1'].append(rouge[metric].recall)
                    results_dict[f'{metric}_fmeasure_beam_1'].append(rouge[metric].fmeasure)

                # METEOR
                results_dict['meteor_beam_1'].append(meteor_score(references=[labels_decoded], hypothesis=best_skeleton_decoded, preprocess=lambda x: x, stemmer=NoOpStemmer()))

                # Edit distance
                results_dict['edit_distance_beam_1'].append(editdistance.eval(best_skeleton_decoded, labels_decoded))

                # Tree edit distance
                if not simplipy_engine.is_valid(best_skeleton_decoded):
                    tree_edit_distance = float('nan')
                else:
                    tree_edit_distance = zss_tree_edit_distance(best_skeleton_decoded, labels_decoded, simplipy_engine.operator_arity)

                results_dict['tree_edit_distance_beam_1'].append(tree_edit_distance)

                # Structural accuracy using model.simplipy_engine.check_valid(expression)
                results_dict['structural_accuracy_beam_1'].append(int(simplipy_engine.is_valid(best_skeleton_decoded)))

                y_pred = model.predict(X)
                y_pred_val = model.predict(X_val)

                assert y_pred.shape == y.shape, f"{y_pred.shape} != {y.shape}"

                # Fit Data
                mse = np.mean((y_pred - y) ** 2)
                r2 = 1 - np.sum((y_pred - y) ** 2) / max(np.sum((y - np.mean(y)) ** 2), np.finfo(np.float32).eps)

                nsrts_accuracy_close = np.mean(np.isclose(y_pred, y, rtol=self.pointwise_close_accuracy_rtol, atol=self.pointwise_close_accuracy_atol)) > self.pointwise_close_criterion
                nsrts_accuracy_r2 = r2 > self.r2_close_criterion

                residuals = y_pred - y

                # Val Data
                mse_val = np.mean((y_pred_val - y_val) ** 2)
                r2_val = 1 - np.sum((y_pred_val - y_val) ** 2) / max(np.sum((y_val - np.mean(y_val)) ** 2), np.finfo(np.float32).eps)

                nsrts_accuracy_close_val = np.mean(np.isclose(y_pred_val, y_val, rtol=self.pointwise_close_accuracy_rtol, atol=self.pointwise_close_accuracy_atol)) > self.pointwise_close_criterion
                nsrts_accuracy_r2_val = r2_val > self.r2_close_criterion

                residuals_val = y_pred_val - y_val

                results_dict['mse_beam_1'].append(mse)
                results_dict['r2_beam_1'].append(r2)

                results_dict['NSRTS_accuracy_close_beam_1'].append(nsrts_accuracy_close)
                results_dict['NSRTS_accuracy_r2_beam_1'].append(nsrts_accuracy_r2)

                results_dict['residuals_beam_1'].append(residuals)

                results_dict['mse_val_beam_1'].append(mse_val)
                results_dict['r2_val_beam_1'].append(r2_val)

                results_dict['NSRTS_accuracy_close_val_beam_1'].append(nsrts_accuracy_close_val)
                results_dict['NSRTS_accuracy_r2_val_beam_1'].append(nsrts_accuracy_r2_val)

                results_dict['residuals_val_beam_1'].append(residuals_val)

                print(f"mse: {mse:.2f}, r2: {r2:.2f}, nsrts_accuracy_close: {nsrts_accuracy_close}, nsrts_accuracy_r2: {nsrts_accuracy_r2}")

                assert len(set(len(v) for v in results_dict.values())) == 1, print({k: len(v) for k, v in results_dict.items()})  # Check that all lists have the same length

        # Sort the scores alphabetically by key
        results_dict = dict(sorted(dict(results_dict).items()))  # type: ignore

        return results_dict

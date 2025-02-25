from typing import Any, Callable
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

from sympy import lambdify

from flash_ansr import FlashANSRDataset
from flash_ansr.utils import load_config
from flash_ansr import ExpressionSpace
from flash_ansr.eval.token_prediction import (
    accuracy,
    precision,
    recall,
    f1_score,
)
from flash_ansr.eval.utils import NoOpStemmer
from flash_ansr.eval.sequences import zss_tree_edit_distance

from nesymres.architectures.model import Model


nltk.download('wordnet', quiet=True)


class NeSymReSEvaluation():
    def __init__(
            self,
            n_support: int | None = None,
            beam_width: int = 1,
            n_restarts: int = 1,
            pointwise_close_criterion: float = 0.95,
            pointwise_close_accuracy_rtol: float = 0.05,
            pointwise_close_accuracy_atol: float = 0.001,
            r2_close_criterion: float = 0.95,
            device: str = 'cpu') -> None:

        self.n_support = n_support
        self.beam_width = beam_width
        self.n_restarts = n_restarts
        self.pointwise_close_criterion = pointwise_close_criterion
        self.pointwise_close_accuracy_rtol = pointwise_close_accuracy_rtol
        self.pointwise_close_accuracy_atol = pointwise_close_accuracy_atol
        self.r2_close_criterion = r2_close_criterion

        self.device = device

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        self.rouge_scorer._tokenizer.tokenize = lambda x: x

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "NeSymReSEvaluation":
        config_ = load_config(config)

        if "evaluation" in config_.keys():
            config_ = config_["evaluation"]

        return cls(
            n_support=config_["n_support"],
            beam_width=config_["beam_width"],
            n_restarts=config_["n_restarts"],
            pointwise_close_criterion=config_["pointwise_close_criterion"],
            pointwise_close_accuracy_rtol=config_["pointwise_close_accuracy_rtol"],
            pointwise_close_accuracy_atol=config_["pointwise_close_accuracy_atol"],
            r2_close_criterion=config_["r2_close_criterion"],
            device=config_["device"]
        )

    def evaluate(
            self,
            model: Model,
            fitfunc: Callable,
            expression_space: ExpressionSpace,
            dataset: FlashANSRDataset,
            size: int | None = None,
            verbose: bool = True) -> dict[str, Any]:

        model.to(self.device).eval()

        results_dict = defaultdict(list)

        if size is None:
            size = len(dataset.skeleton_pool)

        # HACK
        dataset.skeleton_pool.sample_strategy["max_tries"] = 100

        with torch.no_grad():
            for batch in dataset.iterate(size=size, n_support=self.n_support * 2 if self.n_support is not None else None, avoid_fragmentation=True, verbose=verbose, tqdm_total=size, batch_size=1):
                batch = dataset.collate(batch, device=self.device)

                results_dict['input_ids'].append(batch['input_ids'].cpu().numpy())
                results_dict['labels'].append(batch['labels'].cpu().numpy())
                results_dict['constants'].append([c.cpu().numpy() for c in batch['constants']])

                results_dict['x'].append(batch['x_tensors'].cpu().numpy()[:, :self.n_support])
                results_dict['y'].append(batch['y_tensors'].cpu().numpy()[:, :self.n_support])

                results_dict['x_val'].append(batch['x_tensors'].cpu().numpy()[:, self.n_support:])
                results_dict['y_val'].append(batch['y_tensors'].cpu().numpy()[:, self.n_support:])

                results_dict['n_support'].append([batch['x_tensors'].shape[1] // 2] * batch['x_tensors'].shape[0])

                # Create the labels for the next token prediction task (i.e. shift the batch['input_ids'] by one position to the right)
                labels = batch['labels'][0].clone()
                labels_decoded = expression_space.tokenizer.decode(labels.tolist(), special_tokens='<num>')

                # TODO: For different datasets, sort unused dimensions to the end
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                print(expression_space.tokenizer.decode(batch['input_ids'][0].tolist(), special_tokens='<num>'))

                X = batch['x_tensors'].cpu().numpy()[0, :self.n_support]
                y = batch['y_tensors'].cpu().numpy()[0, :self.n_support, 0]

                X_val = batch['x_tensors'].cpu().numpy()[0, self.n_support:]
                y_val = batch['y_tensors'].cpu().numpy()[0, self.n_support:, 0]

                try:
                    fit_time_before = time.time()
                    nesymres_output = fitfunc(X, y)
                    fit_time = time.time() - fit_time_before

                    best_skeleton_decoded = []
                    for token in expression_space.parse_expression(nesymres_output['best_bfgs_preds'][0]):
                        try:
                            float(token)
                            best_skeleton_decoded.append('<num>')
                        except ValueError:
                            best_skeleton_decoded.append(token)
                    best_skeleton = expression_space.tokenizer.encode(best_skeleton_decoded, oov='unk')

                    # Accuracy, precision, recall, F1 score
                    best_skeleton_tensor = torch.tensor(best_skeleton).unsqueeze(0)
                    recall_beam_1 = recall(best_skeleton_tensor, labels[:-1].view(1, -1).cpu(), ignore_index=0, reduction='none').cpu()
                    precision_beam_1 = precision(best_skeleton_tensor, labels[:-1].view(1, -1).cpu(), ignore_index=0, reduction='none').cpu()
                    f1_score_beam_1 = f1_score(best_skeleton_tensor, labels[:-1].view(1, -1).cpu(), ignore_index=0, reduction='none').cpu()
                    accuracy_beam_1 = accuracy(best_skeleton_tensor, labels[:-1].view(1, -1).cpu(), ignore_index=0, reduction='none').cpu()

                    # BLEU
                    bleu_beam_1 = sentence_bleu(references=[labels_decoded], hypothesis=best_skeleton_decoded, smoothing_function=SmoothingFunction().method1)

                    # ROUGE
                    rouge = self.rouge_scorer.score(best_skeleton_decoded, labels_decoded)

                    # METEOR
                    meteor_beam_1 = meteor_score(references=[labels_decoded], hypothesis=best_skeleton_decoded, preprocess=lambda x: x, stemmer=NoOpStemmer())

                    # Edit distance
                    edit_distance_beam_1 = editdistance.eval(best_skeleton_decoded, labels_decoded)

                    # Tree edit distance
                    if not expression_space.is_valid(best_skeleton_decoded):
                        tree_edit_distance = float('nan')
                    else:
                        tree_edit_distance = zss_tree_edit_distance(best_skeleton_decoded, labels_decoded, expression_space.operator_arity)

                    # Structural accuracy using model.expression_space.check_valid(expression)
                    structural_accuracy_beam_1 = int(expression_space.is_valid(best_skeleton))

                    y_pred = lambdify("x_1,x_2,x_3", nesymres_output['best_bfgs_preds'])(*X.T)[0]
                    y_pred_val = lambdify("x_1,x_2,x_3", nesymres_output['best_bfgs_preds'])(*X_val.T)[0]

                    # assert y_pred.shape == y.shape  # Sometimes causes AttributeError: 'int' object has no attribute 'shape'

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

                    results_dict['fit_time'].append(fit_time)
                    results_dict['recall_beam_1'].extend(recall_beam_1)
                    results_dict['precision_beam_1'].extend(precision_beam_1)
                    results_dict['f1_score_beam_1'].extend(f1_score_beam_1)
                    results_dict['accuracy_beam_1'].extend(accuracy_beam_1)
                    results_dict['bleu_beam_1'].append(bleu_beam_1)

                    for metric in ['rouge1', 'rouge2', 'rougeL']:
                        results_dict[f'{metric}_precision_beam_1'].append(rouge[metric].precision)
                        results_dict[f'{metric}_recall_beam_1'].append(rouge[metric].recall)
                        results_dict[f'{metric}_fmeasure_beam_1'].append(rouge[metric].fmeasure)

                    results_dict['meteor_beam_1'].append(meteor_beam_1)
                    results_dict['edit_distance_beam_1'].append(edit_distance_beam_1)
                    results_dict['tree_edit_distance_beam_1'].append(tree_edit_distance)
                    results_dict['structural_accuracy_beam_1'].append(structural_accuracy_beam_1)

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

                except (NameError, KeyError, ValueError, TypeError, OverflowError):
                    results_dict['fit_time'].append(float('nan'))
                    results_dict['recall_beam_1'].append(float('nan'))
                    results_dict['precision_beam_1'].append(float('nan'))
                    results_dict['f1_score_beam_1'].append(float('nan'))
                    results_dict['accuracy_beam_1'].append(float('nan'))
                    results_dict['bleu_beam_1'].append(float('nan'))

                    for metric in ['rouge1', 'rouge2', 'rougeL']:
                        results_dict[f'{metric}_precision_beam_1'].append(float('nan'))
                        results_dict[f'{metric}_recall_beam_1'].append(float('nan'))
                        results_dict[f'{metric}_fmeasure_beam_1'].append(float('nan'))

                    results_dict['meteor_beam_1'].append(float('nan'))
                    results_dict['edit_distance_beam_1'].append(float('nan'))
                    results_dict['tree_edit_distance_beam_1'].append(float('nan'))
                    results_dict['structural_accuracy_beam_1'].append(float('nan'))

                    results_dict['mse_beam_1'].append(float('nan'))
                    results_dict['r2_beam_1'].append(float('nan'))
                    results_dict['NSRTS_accuracy_close_beam_1'].append(float('nan'))
                    results_dict['NSRTS_accuracy_r2_beam_1'].append(float('nan'))
                    results_dict['residuals_beam_1'].append(None)

                    results_dict['mse_val_beam_1'].append(float('nan'))
                    results_dict['r2_val_beam_1'].append(float('nan'))
                    results_dict['NSRTS_accuracy_close_val_beam_1'].append(float('nan'))
                    results_dict['NSRTS_accuracy_r2_val_beam_1'].append(float('nan'))
                    results_dict['residuals_val_beam_1'].append(None)

                assert len(set(len(v) for v in results_dict.values())) == 1, print({k: len(v) for k, v in results_dict.items()})  # Check that all lists have the same length

        # Sort the scores alphabetically by key
        results_dict = dict(sorted(dict(results_dict).items()))  # type: ignore

        return results_dict

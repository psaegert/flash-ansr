from typing import Any, Literal
from collections import defaultdict
import warnings

import torch
import numpy as np
import editdistance
import time

from torch import nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer, scoring

from flash_ansr.models import FlashANSRTransformer
from flash_ansr.data import FlashANSRDataset
from flash_ansr.refine import Refiner, ConvergenceError
from flash_ansr.eval.token_prediction import (
    correct_token_predictions_at_k,
    reciprocal_rank,
    accuracy,
    precision,
    recall,
    f1_score,
    perplexity
)
from flash_ansr.eval.utils import NoOpStemmer
from flash_ansr.eval.sequences import zss_tree_edit_distance
from flash_ansr.utils import load_config

import nltk


nltk.download('wordnet', quiet=True)


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
            n_restarts: int = 1,
            max_len: int = 20,
            numeric_head: bool = False,
            equivalence_pruning: bool = True,
            pointwise_close_criterion: float = 0.95,
            pointwise_close_accuracy_rtol: float = 0.05,
            pointwise_close_accuracy_atol: float = 0.001,
            refiner_method: Literal['curve_fit_lm', 'minimize_bfgs'] = 'curve_fit_lm',
            refiner_p0_noise: str = 'normal',
            refiner_p0_noise_kwargs: dict[str, Any] | None = None,
            r2_close_criterion: float = 0.95,
            device: str = 'cpu') -> None:

        self.n_support = n_support
        self.noise_level = noise_level
        self.beam_width = beam_width
        self.n_restarts = n_restarts
        self.max_len = max_len
        self.numeric_head = numeric_head
        self.equivalence_pruning = equivalence_pruning
        self.pointwise_close_criterion = pointwise_close_criterion
        self.pointwise_close_accuracy_rtol = pointwise_close_accuracy_rtol
        self.pointwise_close_accuracy_atol = pointwise_close_accuracy_atol
        self.r2_close_criterion = r2_close_criterion

        self.refiner_method = refiner_method
        self.refiner_p0_noise = refiner_p0_noise
        self.refiner_p0_noise_kwargs = refiner_p0_noise_kwargs

        self.device = device

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        self.rouge_scorer._tokenizer.tokenize = lambda x: x

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

        return cls(
            n_support=config_["n_support"],
            noise_level=config_.get("noise_level", 0.0),
            beam_width=config_["beam_width"],
            n_restarts=config_["n_restarts"],
            max_len=config_["max_len"],
            numeric_head=config_["numeric_head"],
            equivalence_pruning=config_["equivalence_pruning"],
            pointwise_close_criterion=config_["pointwise_close_criterion"],
            pointwise_close_accuracy_rtol=config_["pointwise_close_accuracy_rtol"],
            pointwise_close_accuracy_atol=config_["pointwise_close_accuracy_atol"],
            refiner_method=config_.get("refiner_method", 'curve_fit_lm'),
            refiner_p0_noise=config_["refiner_p0_noise"],
            refiner_p0_noise_kwargs=config_.get("refiner_p0_noise_kwargs", None),
            r2_close_criterion=config_["r2_close_criterion"],
            device=config_["device"]
        )

    def evaluate(
            self,
            model: FlashANSRTransformer,
            dataset: FlashANSRDataset,
            size: int | None = None,
            verbose: bool = True) -> dict[str, Any]:
        '''
        Evaluate the model on the dataset.

        Parameters
        ----------
        model : FlashANSRTransformer
            The model to evaluate.
        dataset : FlashANSRDataset
            The dataset to evaluate the model on.
        size : int, optional
            Number of samples to evaluate. Default is None.
        verbose : bool, optional
            Whether to print the progress. Default is True.

        Returns
        -------
        dict
            Dictionary with the evaluation results.
        '''

        model.to(self.device).eval()

        refiner = Refiner(model.expression_space)

        results_dict = defaultdict(list)

        if size is None:
            size = len(dataset.skeleton_pool)

        # HACK
        dataset.skeleton_pool.sample_strategy["max_tries"] = 100

        with torch.no_grad():
            current_size = 0
            for single_element_batch in dataset.iterate(size=None, n_support=self.n_support * 2 if self.n_support is not None else None, avoid_fragmentation=False, verbose=verbose, tqdm_total=size):
                input_ids, x_tensor, y_tensor, labels, constants, skeleton_hashes = FlashANSRDataset.collate_batch(single_element_batch, device=self.device)

                x_tensor = x_tensor.unsqueeze(0)
                y_tensor = y_tensor.unsqueeze(0)

                if self.noise_level > 0.0:
                    y_tensor_noisy = y_tensor + (self.noise_level * y_tensor.std() * torch.randn_like(y_tensor))
                    if not torch.all(torch.isfinite(y_tensor_noisy)):
                        warnings.warn('Adding noise to the target variable resulted in non-finite values. Skipping this sample.')
                        continue
                else:
                    y_tensor_noisy = y_tensor

                results_dict['input_ids'].append(input_ids.cpu().numpy())
                results_dict['labels'].append(labels.cpu().numpy())
                results_dict['constants'].append([c.cpu().numpy() for c in constants])

                results_dict['x'].append(x_tensor.cpu().numpy()[:, :self.n_support])
                results_dict['y'].append(y_tensor.cpu().numpy()[:, :self.n_support])
                results_dict['y_noisy'].append(y_tensor_noisy.cpu().numpy()[:, :self.n_support])

                results_dict['x_val'].append(x_tensor.cpu().numpy()[:, self.n_support:])
                results_dict['y_val'].append(y_tensor.cpu().numpy()[:, self.n_support:])
                results_dict['y_noisy_val'].append(y_tensor_noisy.cpu().numpy()[:, self.n_support:])

                results_dict['n_support'].append([x_tensor.shape[1] // 2] * x_tensor.shape[0])

                # Create the labels for the next token prediction task (i.e. shift the input_ids by one position to the right)
                labels = input_ids.clone()[1:]
                labels_decoded = model.expression_space.tokenizer.decode(labels.tolist(), special_tokens='<num>')

                # Pad the x_tensor with zeros to match the expected maximum input dimension of the set transformer
                pad_length = model.encoder_max_n_variables - x_tensor.shape[-1] - y_tensor_noisy.shape[-1]
                if pad_length > 0:
                    x_tensor = nn.functional.pad(x_tensor, (0, pad_length, 0, 0, 0, 0), value=0)

                data_tensor = torch.cat([x_tensor, y_tensor_noisy], dim=-1)

                # Teacher forced forward pass
                logits, num_out = model.forward(input_ids.unsqueeze(0), data_tensor, numeric_head=self.numeric_head)

                # Beam search
                beam_search_time_start = time.time()
                beams, _ = model.beam_search(data_tensor[0], beam_width=self.beam_width, max_len=self.max_len, equivalence_pruning=self.equivalence_pruning)
                results_dict['beam_search_time'].append(time.time() - beam_search_time_start)
                beams_decoded = [model.expression_space.tokenizer.decode(beam, special_tokens='<num>') for beam in beams]

                for j, beam in enumerate(beams):
                    results_dict[f'free_beam_{j+1}'].append(beam[1:-1])

                results_dict['perplexity'].extend([perplexity(log, lab, ignore_index=0, reduction='mean').item() for log, lab in zip(logits[:, :-1], labels.unsqueeze(0))])
                results_dict['correct_token_predictions_at_1'].extend([correct_token_predictions_at_k(log, lab, k=1, ignore_index=0, reduction='mean').item() for log, lab in zip(logits[:, :-1], labels.unsqueeze(0))])
                results_dict['correct_token_predictions_at_5'].extend([correct_token_predictions_at_k(log, lab, k=5, ignore_index=0, reduction='mean').item() for log, lab in zip(logits[:, :-1], labels.unsqueeze(0))])
                results_dict['correct_token_predictions_at_10'].extend([correct_token_predictions_at_k(log, lab, k=10, ignore_index=0, reduction='mean').item() for log, lab in zip(logits[:, :-1], labels.unsqueeze(0))])
                results_dict['reciprocal_rank'].extend([reciprocal_rank(log, lab, ignore_index=0, reduction='mean').item() for log, lab in zip(logits[:, :-1], labels.unsqueeze(0))])

                # Accuracy, precision, recall, F1 score
                for j, beam in enumerate(beams):
                    beam_tensor = torch.tensor(beam[1:], device=self.device).unsqueeze(0)
                    results_dict[f'recall_beam_{j+1}'].extend(recall(beam_tensor, labels.view(1, -1), ignore_index=0, reduction='none').cpu())
                    results_dict[f'precision_beam_{j+1}'].extend(precision(beam_tensor, labels.view(1, -1), ignore_index=0, reduction='none').cpu())
                    results_dict[f'f1_score_beam_{j+1}'].extend(f1_score(beam_tensor, labels.view(1, -1), ignore_index=0, reduction='none').cpu())
                    results_dict[f'accuracy_beam_{j+1}'].extend(accuracy(beam_tensor, labels.view(1, -1), ignore_index=0, reduction='none').cpu())

                # BLEU
                bleu_scores_array = np.empty(self.beam_width)
                for j, beam in enumerate(beams_decoded):
                    bleu_scores_array[j] = sentence_bleu(references=[labels_decoded], hypothesis=beam, smoothing_function=SmoothingFunction().method1)

                for i in range(self.beam_width):
                    results_dict[f'bleu_beam_{i+1}'].append(bleu_scores_array[i])

                # ROUGE
                scores_list: list[dict[str, scoring.Score]] = []
                for beam in beams_decoded:
                    scores_list.append(self.rouge_scorer.score(beam, labels_decoded))

                for metric in ['rouge1', 'rouge2', 'rougeL']:
                    for i, scores in enumerate(scores_list):
                        results_dict[f'{metric}_precision_beam_{i+1}'].append(scores[metric].precision)
                        results_dict[f'{metric}_recall_beam_{i+1}'].append(scores[metric].recall)
                        results_dict[f'{metric}_fmeasure_beam_{i+1}'].append(scores[metric].fmeasure)

                # METEOR
                meteor_scores_array = np.empty(self.beam_width)
                for j, beam in enumerate(beams_decoded):
                    meteor_scores_array[j] = meteor_score(references=[labels_decoded], hypothesis=beam, preprocess=lambda x: x, stemmer=NoOpStemmer())

                for i in range(self.beam_width):
                    results_dict[f'meteor_beam_{i+1}'].append(meteor_scores_array[i])

                # Edit distance
                edit_distances_array = np.empty(self.beam_width)
                for j, beam in enumerate(beams_decoded):
                    edit_distances_array[j] = editdistance.eval(beam, labels_decoded)

                for i in range(self.beam_width):
                    results_dict[f'edit_distance_beam_{i+1}'].append(edit_distances_array[i])

                # Tree edit distance
                tree_edit_distances_array = np.empty(self.beam_width)
                for j, beam in enumerate(beams_decoded):
                    if not model.expression_space.is_valid(beam):
                        tree_edit_distances_array[j] = float('nan')
                    else:
                        tree_edit_distances_array[j] = zss_tree_edit_distance(beam, labels_decoded, model.expression_space.operator_arity)

                for i in range(self.beam_width):
                    results_dict[f'tree_edit_distance_beam_{i+1}'].append(tree_edit_distances_array[i])

                # Structural accuracy using model.expression_space.check_valid(expression)
                for j, beam in enumerate(beams_decoded):
                    results_dict[f'structural_accuracy_beam_{j+1}'].append(int(model.expression_space.is_valid(beam)))

                # Constant Fitting
                np_errors_before = np.geterr()
                np.seterr(all='ignore')

                X = x_tensor.cpu().numpy()[0, :self.n_support]
                y = y_tensor_noisy.cpu().numpy()[0, :self.n_support]

                X_val = x_tensor.cpu().numpy()[0, self.n_support:]
                y_val = y_tensor_noisy.cpu().numpy()[0, self.n_support:]

                for j, beam in enumerate(beams_decoded):
                    refiner_time = 0.0
                    valid_results = model.expression_space.is_valid(beam)

                    if valid_results:
                        numeric_prediction = None

                        if self.numeric_head:
                            _, num_output = model.forward(beam, data_tensor, numeric_head=True)
                            numeric_prediction = num_output[0, 1:, 0][beam == model.expression_space.tokenizer["<num>"]]  # FIXME: Start at 1 or 0?

                        try:
                            refiner_time_start = time.time()
                            refiner.fit(
                                expression=beam,
                                X=X,
                                y=y,
                                n_restarts=self.n_restarts,
                                method=self.refiner_method,
                                p0=numeric_prediction,
                                p0_noise=self.refiner_p0_noise,
                                p0_noise_kwargs=self.refiner_p0_noise_kwargs,
                                converge_error='raise')
                            refiner_time += (time.time() - refiner_time_start)

                            assert X.dtype == y.dtype

                            y_pred = refiner.predict(X)
                            y_pred_val = refiner.predict(X_val)

                            assert y_pred.shape == y.shape
                            assert y_pred_val.shape == y_val.shape

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

                        except (ConvergenceError, OverflowError, TypeError, ValueError):
                            valid_results = False

                    if not valid_results:
                        mse = float('nan')
                        r2 = float('nan')
                        nsrts_accuracy_close = float('nan')
                        nsrts_accuracy_r2 = float('nan')
                        residuals = None
                        refiner_time = float('nan')
                        mse_val = float('nan')
                        r2_val = float('nan')
                        nsrts_accuracy_close_val = float('nan')
                        nsrts_accuracy_r2_val = float('nan')
                        residuals_val = None

                    results_dict[f'mse_beam_{j+1}'].append(mse)
                    results_dict[f'r2_beam_{j+1}'].append(r2)

                    results_dict[f'NSRTS_accuracy_close_beam_{j+1}'].append(nsrts_accuracy_close)
                    results_dict[f'NSRTS_accuracy_r2_beam_{j+1}'].append(nsrts_accuracy_r2)

                    results_dict[f'residuals_beam_{j+1}'].append(residuals)
                    results_dict[f'refiner_time_beam_{j+1}'].append(refiner_time)

                    results_dict[f'mse_val_beam_{j+1}'].append(mse_val)
                    results_dict[f'r2_val_beam_{j+1}'].append(r2_val)

                    results_dict[f'NSRTS_accuracy_close_val_beam_{j+1}'].append(nsrts_accuracy_close_val)
                    results_dict[f'NSRTS_accuracy_r2_val_beam_{j+1}'].append(nsrts_accuracy_r2_val)

                    results_dict[f'residuals_val_beam_{j+1}'].append(residuals_val)

                np.seterr(**np_errors_before)

                assert len(set(len(v) for v in results_dict.values())) == 1  # Check that all lists have the same length

                current_size += 1

                if current_size >= size:
                    break

        del refiner

        # Sort the scores alphabetically by key
        results_dict = dict(sorted(dict(results_dict).items()))  # type: ignore

        return results_dict

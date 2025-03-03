import time
from typing import Any
from collections import defaultdict
import warnings

import torch
import numpy as np
import editdistance

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer, scoring

from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.data import FlashANSRDataset, FlashASNRPreprocessor
from flash_ansr.refine import ConvergenceError
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
            complexity: str | list[int | float] = 'none',
            preprocess: bool = False,
            pointwise_close_criterion: float = 0.95,
            pointwise_close_accuracy_rtol: float = 0.05,
            pointwise_close_accuracy_atol: float = 0.001,
            r2_close_criterion: float = 0.95,
            device: str = 'cpu') -> None:

        self.n_support = n_support
        self.noise_level = noise_level
        self.beam_width = beam_width
        self.complexity = complexity
        self.preprocess = preprocess
        self.pointwise_close_criterion = pointwise_close_criterion
        self.pointwise_close_accuracy_rtol = pointwise_close_accuracy_rtol
        self.pointwise_close_accuracy_atol = pointwise_close_accuracy_atol
        self.r2_close_criterion = r2_close_criterion

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
            complexity=config_.get("complexity", 'none'),
            preprocess=config_.get("preprocess", False),
            pointwise_close_criterion=config_["pointwise_close_criterion"],
            pointwise_close_accuracy_rtol=config_["pointwise_close_accuracy_rtol"],
            pointwise_close_accuracy_atol=config_["pointwise_close_accuracy_atol"],
            r2_close_criterion=config_["r2_close_criterion"],
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
        # Make sure the model is in evaluation mode
        model.to(self.device).eval()

        results_dict = defaultdict(list)

        if size is None:
            size = len(dataset.skeleton_pool)

        # HACK
        dataset.skeleton_pool.sample_strategy["max_tries"] = 100
        if self.preprocess:
            dataset.preprocessor = FlashASNRPreprocessor(model.expression_space, format_probs={'complexity': 1.0})

        with torch.no_grad():
            current_size = 0
            for batch in dataset.iterate(size=None, n_support=self.n_support * 2 if self.n_support is not None else None, avoid_fragmentation=False, preprocess=self.preprocess, verbose=verbose, tqdm_total=size, batch_size=1):
                batch = dataset.collate(batch, device=self.device)

                if self.noise_level > 0.0:
                    batch['y_tensors_noisy'] = batch['y_tensors'] + (self.noise_level * batch['y_tensors'].std() * torch.randn_like(batch['y_tensors']))
                    if not torch.all(torch.isfinite(batch['y_tensors_noisy'])):
                        warnings.warn('Adding noise to the target variable resulted in non-finite values. Skipping this sample.')
                        continue
                else:
                    batch['y_tensors_noisy'] = batch['y_tensors']

                X = batch['x_tensors'].cpu().numpy()[0, :self.n_support]
                y = batch['y_tensors_noisy'].cpu().numpy()[0, :self.n_support]

                X_val = batch['x_tensors'].cpu().numpy()[0, self.n_support:]
                y_val = batch['y_tensors_noisy'].cpu().numpy()[0, self.n_support:]

                results_dict['input_ids'].append(batch['input_ids'][0].cpu().numpy())
                results_dict['labels'].append(batch['labels'][0].cpu().numpy())
                results_dict['constants'].append([c.cpu().numpy() for c in batch['constants'][0]])

                results_dict['x'].append(batch['x_tensors'].cpu().numpy()[:, :self.n_support])
                results_dict['y'].append(batch['y_tensors'].cpu().numpy()[:, :self.n_support])
                results_dict['y_noisy'].append(batch['y_tensors_noisy'].cpu().numpy()[:, :self.n_support])

                results_dict['x_val'].append(batch['x_tensors'].cpu().numpy()[:, self.n_support:])
                results_dict['y_val'].append(batch['y_tensors'].cpu().numpy()[:, self.n_support:])
                results_dict['y_noisy_val'].append(batch['y_tensors_noisy'].cpu().numpy()[:, self.n_support:])

                results_dict['n_support'].append([batch['x_tensors'].shape[1] // 2] * batch['x_tensors'].shape[0])

                # Create the labels for the next token prediction task (i.e. shift the batch['input_ids'] by one position to the right)
                labels = batch['labels'][0].clone()
                labels_decoded = model.expression_space.tokenizer.decode(labels.tolist(), special_tokens='<num>')

                data_tensor = torch.cat([batch['x_tensors'][:, :self.n_support], batch['y_tensors_noisy'][:, :self.n_support]], dim=-1)

                valid_results = True
                try:
                    # Teacher forced forward pass
                    if self.complexity == 'none':
                        logits, _ = model.flash_ansr_transformer.forward(batch['input_ids'], data_tensor)
                        fit_time_start = time.time()
                        model.fit(X, y)
                        fit_time = time.time() - fit_time_start
                    elif self.complexity == 'ground_truth':
                        logits, _ = model.flash_ansr_transformer.forward(batch['input_ids'], data_tensor, batch['input_num'])
                        fit_time_start = time.time()
                        print('Complexity:', batch['complexities'])
                        model.fit(X, y, complexity=batch['complexities'])
                        fit_time = time.time() - fit_time_start
                    elif isinstance(self.complexity, list):
                        raise NotImplementedError('Complexity list not implemented yet.')
                        # logits, _ = model.flash_ansr_transformer.forward(batch['input_ids'], data_tensor, batch['input_num'])
                        # fit_time_start = time.time()
                        # model.fit(X, y, complexity=self.complexity)
                        # fit_time = time.time() - fit_time_start
                    else:
                        raise NotImplementedError(f'Complexity {self.complexity} not implemented yet.')

                except (ConvergenceError, OverflowError, TypeError, ValueError):
                    print('Error in the forward pass or fitting. Skipping this sample.')
                    valid_results = False
                    fit_time = float('nan')

                results_dict['fit_time'].append(fit_time)

                if valid_results:
                    beams = [r['beam'] for r in model._results]

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

                    for j, result in enumerate(model._results):
                        beam_valid_results = True
                        try:
                            assert X.dtype == y.dtype

                            y_pred = result['refiner'].predict(X)
                            y_pred_val = result['refiner'].predict(X_val)

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
                            beam_valid_results = False

                        if not beam_valid_results:
                            mse = float('nan')
                            r2 = float('nan')
                            nsrts_accuracy_close = float('nan')
                            nsrts_accuracy_r2 = float('nan')
                            residuals = None
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

                        results_dict[f'mse_val_beam_{j+1}'].append(mse_val)
                        results_dict[f'r2_val_beam_{j+1}'].append(r2_val)

                        results_dict[f'NSRTS_accuracy_close_val_beam_{j+1}'].append(nsrts_accuracy_close_val)
                        results_dict[f'NSRTS_accuracy_r2_val_beam_{j+1}'].append(nsrts_accuracy_r2_val)

                        results_dict[f'residuals_val_beam_{j+1}'].append(residuals_val)

                    np.seterr(**np_errors_before)

                else:
                    # Fill with NaNs
                    results_dict['perplexity'].extend([float('nan')] * len(logits))
                    results_dict['correct_token_predictions_at_1'].extend([float('nan')] * len(logits))
                    results_dict['correct_token_predictions_at_5'].extend([float('nan')] * len(logits))
                    results_dict['correct_token_predictions_at_10'].extend([float('nan')] * len(logits))
                    results_dict['reciprocal_rank'].extend([float('nan')] * len(logits))

                    for j in range(self.beam_width):
                        results_dict[f'free_beam_{j+1}'].append([float('nan')])
                        results_dict[f'bleu_beam_{j+1}'].append(float('nan'))
                        results_dict[f'meteor_beam_{j+1}'].append(float('nan'))
                        results_dict[f'edit_distance_beam_{j+1}'].append(float('nan'))
                        results_dict[f'tree_edit_distance_beam_{j+1}'].append(float('nan'))
                        results_dict[f'structural_accuracy_beam_{j+1}'].append(float('nan'))
                        results_dict[f'accuracy_beam_{j+1}'].extend([float('nan')] * len(logits))
                        results_dict[f'f1_score_beam_{j+1}'].extend([float('nan')] * len(logits))
                        results_dict[f'precision_beam_{j+1}'].extend([float('nan')] * len(logits))
                        results_dict[f'recall_beam_{j+1}'].extend([float('nan')] * len(logits))

                        for metric in ['rouge1', 'rouge2', 'rougeL']:
                            results_dict[f'{metric}_precision_beam_{j+1}'].append(float('nan'))
                            results_dict[f'{metric}_recall_beam_{j+1}'].append(float('nan'))
                            results_dict[f'{metric}_fmeasure_beam_{j+1}'].append(float('nan'))

                        results_dict[f'mse_beam_{j+1}'].append(float('nan'))
                        results_dict[f'r2_beam_{j+1}'].append(float('nan'))
                        results_dict[f'NSRTS_accuracy_close_beam_{j+1}'].append(float('nan'))
                        results_dict[f'NSRTS_accuracy_r2_beam_{j+1}'].append(float('nan'))
                        results_dict[f'residuals_beam_{j+1}'].append(None)
                        results_dict[f'mse_val_beam_{j+1}'].append(float('nan'))
                        results_dict[f'r2_val_beam_{j+1}'].append(float('nan'))
                        results_dict[f'NSRTS_accuracy_close_val_beam_{j+1}'].append(float('nan'))
                        results_dict[f'NSRTS_accuracy_r2_val_beam_{j+1}'].append(float('nan'))
                        results_dict[f'residuals_val_beam_{j+1}'].append(None)

                # Check that all lists have the same length
                if not len(set(len(v) for v in results_dict.values())) == 1:
                    for k, v in results_dict.items():
                        print(f'{k}: {len(v)}')
                    raise ValueError(f'Inconsistent lengths of the results_dict lists: {set(len(v) for v in results_dict.values())}')

                current_size += 1

                if current_size >= size:
                    break

        # Sort the scores alphabetically by key
        results_dict = dict(sorted(dict(results_dict).items()))  # type: ignore

        return results_dict

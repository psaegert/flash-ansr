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
from rouge_score import rouge_scorer

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
from flash_ansr.utils import load_config, pad_input_set


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

        if 'beam_width' in config_.keys():
            beams = config_['beam_width']
        elif 'generation_config' in config_.keys():
            if 'beam_width' in config_['generation_config'].keys():
                beams = config_['generation_config']['beam_width']
            elif 'choices' in config_['generation_config'].keys():
                beams = config_['generation_config']['choices']
            else:
                raise ValueError('Beam width not found in the configuration.')
        else:
            raise ValueError('Beam width not found in the configuration.')

        return cls(
            n_support=config_["n_support"],
            noise_level=config_.get("noise_level", 0.0),
            beam_width=beams,
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
        if verbose:
            print(f'Evaluating model with configuration: beam_width={self.beam_width}, noise_level={self.noise_level}, n_support={self.n_support}, complexity={self.complexity}')

        # Make sure the model is in evaluation mode
        model.to(self.device).eval()

        results_dict = defaultdict(list)

        if size is None:
            size = len(dataset.skeleton_pool)

        # HACK
        dataset.skeleton_pool.sample_strategy["max_tries"] = 100

        # HACK: Ensure compatibility of tokenization and input variables
        print('Recompiling skeleton and holdout codes to ensure compatibility...')
        dataset.skeleton_pool.simplipy_engine = model.simplipy_engine
        dataset.skeleton_pool.skeleton_codes = dataset.skeleton_pool.compile_codes(verbose=verbose)
        for holdout_pool in dataset.skeleton_pool.holdout_pools:
            holdout_pool.simplipy_engine = model.simplipy_engine
            holdout_pool.skeleton_codes = holdout_pool.compile_codes(verbose=verbose)

        if self.preprocess:
            dataset.preprocessor = FlashASNRPreprocessor(model.simplipy_engine, format_probs={'complexity': 1.0})

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

                # target_expression = model.flash_ansr_transformer.extract_expression_from_beam(batch['labels'][0].cpu().numpy())[0]
                # target_expression_decoded = model.tokenizer.decode(target_expression, special_tokens='<constant>')

                # target_expression_labels = torch.tensor(target_expression + [model.tokenizer['<eos>']], device=self.device)
                # target_expression_labels_decoded = model.tokenizer.decode(target_expression_labels.cpu().numpy(), special_tokens=['<constant>', '<eos>'])

                batch_size = len(batch['input_ids'])

                x_tensor_padded = pad_input_set(batch['x_tensors'][:, :self.n_support], model.n_variables)

                data_tensor = torch.cat([x_tensor_padded, batch['y_tensors_noisy'][:, :self.n_support]], dim=-1)

                valid_results = True
                try:
                    # Teacher forced forward pass
                    if self.complexity == 'none':
                        next_token_logits, _ = model.flash_ansr_transformer.forward(batch['input_ids'], data_tensor)
                        fit_time_start = time.time()
                        model.fit(X, y)
                        fit_time = time.time() - fit_time_start
                    elif self.complexity == 'ground_truth':
                        next_token_logits, _ = model.flash_ansr_transformer.forward(batch['input_ids'], data_tensor, batch['input_num'])
                        fit_time_start = time.time()
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

                    bos_position = torch.where(batch['input_ids'] == dataset.tokenizer['<bos>'])[1][0].item()

                    expression_next_token_logits_with_eos = next_token_logits[:, bos_position:-1]  # type: ignore
                    expression_next_token_labels_with_eos = batch['labels'][:, bos_position:]  # type: ignore

                    expresssion_labels_decoded = dataset.tokenizer.decode(expression_next_token_labels_with_eos[0][:-1], special_tokens=['<constant>', '<eos>'])

                except (ConvergenceError, OverflowError, TypeError, ValueError):
                    print('Error in the forward pass or fitting.')
                    valid_results = False
                    fit_time = float('nan')

                results_dict['fit_time'].append(fit_time)

                if valid_results:
                    beams = [r['beam'] for r in model._results]
                    log_probs = [r['log_prob'] for r in model._results]

                    beams_decoded = [model.tokenizer.decode(beam, special_tokens='<constant>') for beam in beams]

                    for j in range(self.beam_width):
                        if j >= len(beams):
                            results_dict[f'free_beam_{j+1}'].append(None)
                            results_dict[f'log_prob_beam_{j+1}'].append(float('nan'))
                            continue
                        beam = beams_decoded[j]
                        results_dict[f'free_beam_{j+1}'].append(beam)
                        results_dict[f'log_prob_beam_{j+1}'].append(log_probs[j])

                    results_dict['perplexity'].extend([perplexity(log, lab, ignore_index=0, reduction='mean').item() for log, lab in zip(expression_next_token_logits_with_eos, expression_next_token_labels_with_eos)])
                    results_dict['correct_token_predictions_at_1'].extend([correct_token_predictions_at_k(log, lab, k=1, ignore_index=0, reduction='mean').item() for log, lab in zip(expression_next_token_logits_with_eos, expression_next_token_labels_with_eos)])
                    results_dict['correct_token_predictions_at_5'].extend([correct_token_predictions_at_k(log, lab, k=5, ignore_index=0, reduction='mean').item() for log, lab in zip(expression_next_token_logits_with_eos, expression_next_token_labels_with_eos)])
                    results_dict['correct_token_predictions_at_10'].extend([correct_token_predictions_at_k(log, lab, k=10, ignore_index=0, reduction='mean').item() for log, lab in zip(expression_next_token_logits_with_eos, expression_next_token_labels_with_eos)])
                    results_dict['reciprocal_rank'].extend([reciprocal_rank(log, lab, ignore_index=0, reduction='mean').item() for log, lab in zip(expression_next_token_logits_with_eos, expression_next_token_labels_with_eos)])

                    # Accuracy, precision, recall, F1 score
                    for j in range(self.beam_width):
                        if j >= len(beams):
                            results_dict[f'recall_beam_{j+1}'].append(float('nan'))
                            results_dict[f'precision_beam_{j+1}'].append(float('nan'))
                            results_dict[f'f1_score_beam_{j+1}'].append(float('nan'))
                            results_dict[f'accuracy_beam_{j+1}'].append(float('nan'))
                            continue
                        beam = beams[j]
                        beam_tensor = torch.tensor(beam, device=self.device).unsqueeze(0)
                        results_dict[f'recall_beam_{j+1}'].extend(recall(beam_tensor, expression_next_token_labels_with_eos[:, :-1], ignore_index=0, reduction='none').cpu())
                        results_dict[f'precision_beam_{j+1}'].extend(precision(beam_tensor, expression_next_token_labels_with_eos[:, :-1], ignore_index=0, reduction='none').cpu())
                        results_dict[f'f1_score_beam_{j+1}'].extend(f1_score(beam_tensor, expression_next_token_labels_with_eos[:, :-1], ignore_index=0, reduction='none').cpu())
                        results_dict[f'accuracy_beam_{j+1}'].extend(accuracy(beam_tensor, expression_next_token_labels_with_eos[:, :-1], ignore_index=0, reduction='none').cpu())

                    # BLEU
                    for j in range(self.beam_width):
                        if j >= len(beams_decoded):
                            results_dict[f'bleu_beam_{j+1}'].append(float('nan'))
                            continue
                        beam = beams_decoded[j]
                        results_dict[f'bleu_beam_{j+1}'].append(sentence_bleu(references=[expresssion_labels_decoded], hypothesis=beam, smoothing_function=SmoothingFunction().method1))

                    # ROUGE
                    for j in range(self.beam_width):
                        if j >= len(beams_decoded):
                            for metric in ['rouge1', 'rouge2', 'rougeL']:
                                results_dict[f'{metric}_precision_beam_{j+1}'].append(float('nan'))
                                results_dict[f'{metric}_recall_beam_{j+1}'].append(float('nan'))
                                results_dict[f'{metric}_fmeasure_beam_{j+1}'].append(float('nan'))
                            continue
                        beam = beams_decoded[j]
                        scores = self.rouge_scorer.score(beam, expresssion_labels_decoded)
                        for metric in ['rouge1', 'rouge2', 'rougeL']:
                            results_dict[f'{metric}_precision_beam_{j+1}'].append(scores[metric].precision)
                            results_dict[f'{metric}_recall_beam_{j+1}'].append(scores[metric].recall)
                            results_dict[f'{metric}_fmeasure_beam_{j+1}'].append(scores[metric].fmeasure)

                    # METEOR
                    for j in range(self.beam_width):
                        if j >= len(beams_decoded):
                            results_dict[f'meteor_beam_{j+1}'].append(float('nan'))
                            continue
                        beam = beams_decoded[j]
                        results_dict[f'meteor_beam_{j+1}'].append(meteor_score(references=[expresssion_labels_decoded], hypothesis=beam, preprocess=lambda x: x, stemmer=NoOpStemmer()))

                    # Edit distance
                    for j in range(self.beam_width):
                        if j >= len(beams_decoded):
                            results_dict[f'edit_distance_beam_{j+1}'].append(float('nan'))
                            continue
                        beam = beams_decoded[j]
                        results_dict[f'edit_distance_beam_{j+1}'].append(editdistance.eval(beam, expresssion_labels_decoded))

                    # Tree edit distance
                    for j in range(self.beam_width):
                        if j >= len(beams_decoded):
                            results_dict[f'tree_edit_distance_beam_{j+1}'].append(float('nan'))
                            continue
                        beam = beams_decoded[j]
                        if not model.simplipy_engine.is_valid(beam):
                            results_dict[f'tree_edit_distance_beam_{j+1}'].append(float('nan'))
                        else:
                            results_dict[f'tree_edit_distance_beam_{j+1}'].append(zss_tree_edit_distance(beam, expresssion_labels_decoded, model.simplipy_engine.operator_arity))

                    # Structural accuracy
                    for j in range(self.beam_width):
                        if j >= len(beams_decoded):
                            results_dict[f'structural_accuracy_beam_{j+1}'].append(float('nan'))
                            continue
                        beam = beams_decoded[j]
                        results_dict[f'structural_accuracy_beam_{j+1}'].append(int(model.simplipy_engine.is_valid(beam)))

                    # Constant Fitting
                    np_errors_before = np.geterr()
                    np.seterr(all='ignore')

                    for j in range(self.beam_width):
                        if j < len(beams):
                            result = model._results[j]
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
                                print('Error in the constant fitting.')
                                beam_valid_results = False

                        else:
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
                    results_dict['perplexity'].extend([float('nan')] * batch_size)
                    results_dict['correct_token_predictions_at_1'].extend([float('nan')] * batch_size)
                    results_dict['correct_token_predictions_at_5'].extend([float('nan')] * batch_size)
                    results_dict['correct_token_predictions_at_10'].extend([float('nan')] * batch_size)
                    results_dict['reciprocal_rank'].extend([float('nan')] * batch_size)

                    for j in range(self.beam_width):
                        results_dict[f'free_beam_{j+1}'].append([float('nan')])
                        results_dict[f'log_prob_beam_{j+1}'].append(float('nan'))
                        results_dict[f'bleu_beam_{j+1}'].append(float('nan'))
                        results_dict[f'meteor_beam_{j+1}'].append(float('nan'))
                        results_dict[f'edit_distance_beam_{j+1}'].append(float('nan'))
                        results_dict[f'tree_edit_distance_beam_{j+1}'].append(float('nan'))
                        results_dict[f'structural_accuracy_beam_{j+1}'].append(float('nan'))
                        results_dict[f'accuracy_beam_{j+1}'].extend([float('nan')] * batch_size)
                        results_dict[f'f1_score_beam_{j+1}'].extend([float('nan')] * batch_size)
                        results_dict[f'precision_beam_{j+1}'].extend([float('nan')] * batch_size)
                        results_dict[f'recall_beam_{j+1}'].extend([float('nan')] * batch_size)

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

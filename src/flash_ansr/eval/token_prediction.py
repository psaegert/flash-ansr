from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F


def correct_token_predictions_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int, reduction: Literal['mean', 'sum', 'none'] = 'mean', ignore_index: int | list[int] | None = None) -> torch.Tensor:
    '''
    Compute the number of correct next-token predictions at k.

    Parameters
    ----------
    logits : torch.Tensor
        The model's output logits.
    labels : torch.Tensor
        The ground truth labels.
    k : int
        The number of top-k predictions to consider.
    reduction : {'mean', 'sum', 'none'}, optional
        The reduction method to apply to the output tensor. Default is 'mean'.
    ignore_index : int, list[int], or None, optional
        The index or indices to ignore in the evaluation (e.g. padding). Default is None.

    Returns
    -------
    torch.Tensor
        The number of correct next-token predictions at k.
    '''
    if logits.ndim != 2:
        raise ValueError(f"Expected logits to have 2 dimensions, got {logits.ndim}")

    if labels.ndim != 1:
        raise ValueError(f"Expected labels to have 1 dimension, got {labels.ndim}")

    if isinstance(ignore_index, int):
        ignore_index = [ignore_index]

    if ignore_index is not None:
        ignore_mask = (labels.unsqueeze(-1) == torch.tensor(ignore_index, device=labels.device, dtype=labels.dtype).unsqueeze(0)).any(dim=-1)

    _, topk_pred = logits.topk(k, dim=-1)
    labels = labels.unsqueeze(-1).expand_as(topk_pred)

    if ignore_index is not None:
        correct = torch.any(torch.eq(topk_pred[~ignore_mask], labels[~ignore_mask]), dim=-1).float()
    else:
        correct = torch.any(torch.eq(topk_pred, labels), dim=-1).float()

    match reduction:
        case 'mean':
            return correct.mean()
        case 'sum':
            return correct.sum()
        case 'none':
            return correct
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def reciprocal_rank(logits: torch.Tensor, labels: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean', ignore_index: int | list[int] | None = None) -> torch.Tensor:
    '''
    Compute the reciprocal ranks of the correct next-token prediction.

    Parameters
    ----------
    logits : torch.Tensor
        The model's output logits.
    labels : torch.Tensor
        The ground truth labels.
    reduction : {'mean', 'sum', 'none'}, optional
        The reduction method to apply to the output tensor. Default is 'mean'.
    ignore_index : int, list[int], or None, optional
        The index or indices to ignore in the evaluation (e.g. padding). Default is None.

    Returns
    -------
    torch.Tensor
        The reciprocal ranks of the correct next-token prediction.
    '''
    if logits.ndim != 2:
        raise ValueError(f"Expected logits to have 2 dimensions, got {logits.ndim}")

    if labels.ndim != 1:
        raise ValueError(f"Expected labels to have 1 dimension, got {labels.ndim}")

    if isinstance(ignore_index, int):
        ignore_index = [ignore_index]

    ranks = torch.argsort(logits, descending=True, dim=-1).argsort(-1)

    if ignore_index is not None:
        ignore_mask = (labels.unsqueeze(-1) == torch.tensor(ignore_index, device=labels.device, dtype=labels.dtype).unsqueeze(0)).any(dim=-1)
        reciprocal_ranks = torch.reciprocal(ranks[~ignore_mask].gather(1, labels[~ignore_mask].unsqueeze(-1)).float() + 1).squeeze(-1)
    else:
        reciprocal_ranks = torch.reciprocal(ranks.gather(1, labels.unsqueeze(-1)).float() + 1).squeeze(-1)

    match reduction:
        case 'mean':
            return reciprocal_ranks.mean()
        case 'sum':
            return reciprocal_ranks.sum()
        case 'none':
            return reciprocal_ranks
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        raise ValueError("Expected scalar tensor when converting to python value")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        raise ValueError("Expected scalar ndarray when converting to python value")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ensure_python_scalars(sequence: list[Any]) -> list[Any]:
    return [_to_python_scalar(item) for item in sequence]


def _is_sequence_like(item: Any) -> bool:
    if isinstance(item, (str, bytes)):
        return False
    if isinstance(item, torch.Tensor):
        return item.ndim <= 1
    if isinstance(item, np.ndarray):
        return item.ndim <= 1
    return isinstance(item, Sequence)


def _flatten_token_sequence(sequence_like: Any, name: str) -> list[Any]:
    if isinstance(sequence_like, torch.Tensor):
        if sequence_like.ndim == 0:
            return [_to_python_scalar(sequence_like)]
        if sequence_like.ndim == 1:
            return _ensure_python_scalars(sequence_like.tolist())
        raise ValueError(f"Expected 1D tensor for {name}, got {sequence_like.ndim}D tensor")
    if isinstance(sequence_like, np.ndarray):
        if sequence_like.ndim == 0:
            return [_to_python_scalar(sequence_like)]
        if sequence_like.ndim == 1:
            return _ensure_python_scalars(sequence_like.tolist())
        raise ValueError(f"Expected 1D ndarray for {name}, got {sequence_like.ndim}D array")
    if isinstance(sequence_like, (str, bytes)):
        return [sequence_like]
    if isinstance(sequence_like, Sequence):
        return _ensure_python_scalars(list(sequence_like))
    return [_to_python_scalar(sequence_like)]


def _normalize_sequences(data: Any, name: str) -> list[list[Any]]:
    if isinstance(data, torch.Tensor):
        if data.ndim == 0:
            return [[_to_python_scalar(data)]]
        if data.ndim == 1:
            return [_ensure_python_scalars(data.tolist())]
        if data.ndim == 2:
            return [_ensure_python_scalars(row.tolist()) for row in data]
        raise ValueError(f"Expected {name} to be 1D or 2D tensor, got {data.ndim}D")

    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            return [[_to_python_scalar(data)]]
        if data.ndim == 1:
            return [_ensure_python_scalars(data.tolist())]
        if data.ndim == 2:
            return [_ensure_python_scalars(row.tolist()) for row in data]
        raise ValueError(f"Expected {name} to be 1D or 2D ndarray, got {data.ndim}D")

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if not data:
            return [[]]
        if all(_is_sequence_like(item) for item in data):
            return [_flatten_token_sequence(item, name) for item in data]
        return [_ensure_python_scalars(list(data))]

    return [[_to_python_scalar(data)]]


def _normalize_ignore_index(ignore_index: Any) -> set[Any]:
    if ignore_index is None:
        return set()

    if isinstance(ignore_index, (str, bytes)):
        return {ignore_index}

    if isinstance(ignore_index, torch.Tensor):
        if ignore_index.ndim == 0:
            return {ignore_index.item()}
        return set(ignore_index.detach().cpu().flatten().tolist())

    if isinstance(ignore_index, np.ndarray):
        return set(np.asarray(ignore_index).flatten().tolist())

    if isinstance(ignore_index, Sequence) and not isinstance(ignore_index, (str, bytes)):
        return {_to_python_scalar(item) for item in ignore_index}

    return {_to_python_scalar(ignore_index)}


def _prepare_inputs(pred_labels: Any, labels: Any, ignore_index: Any, pred_name: str) -> tuple[list[list[Any]], list[list[Any]], set[Any]]:
    pred_sequences = _normalize_sequences(pred_labels, pred_name)
    label_sequences = _normalize_sequences(labels, 'labels')

    if len(pred_sequences) != len(label_sequences):
        raise ValueError(
            f"Mismatched number of sequences between {pred_name} ({len(pred_sequences)}) and labels ({len(label_sequences)})."
        )

    ignore_set = _normalize_ignore_index(ignore_index)
    return pred_sequences, label_sequences, ignore_set


def _apply_reduction(values: torch.Tensor, reduction: Literal['mean', 'sum', 'none']) -> torch.Tensor:
    if reduction == 'none':
        return values
    if reduction == 'mean':
        if values.numel() == 0:
            return torch.tensor(float('nan'), dtype=values.dtype)
        return torch.nanmean(values)
    if reduction == 'sum':
        if values.numel() == 0:
            return torch.tensor(0.0, dtype=values.dtype)
        return torch.nansum(values)
    raise ValueError(f"Invalid reduction: {reduction}")


def _compute_precision_values(pred_sequences: list[list[Any]], label_sequences: list[list[Any]], ignore_set: set[Any]) -> torch.Tensor:
    batch_values: list[float] = []
    for preds, labels in zip(pred_sequences, label_sequences):
        filtered_preds = [token for token in preds if token not in ignore_set]
        filtered_labels = [token for token in labels if token not in ignore_set]

        pred_set = set(filtered_preds)
        label_set = set(filtered_labels)

        if not pred_set:
            batch_values.append(float('nan'))
            continue

        true_positives = len(pred_set & label_set)
        batch_values.append(true_positives / len(pred_set))

    return torch.tensor(batch_values, dtype=torch.float32)


def _compute_recall_values(pred_sequences: list[list[Any]], label_sequences: list[list[Any]], ignore_set: set[Any]) -> torch.Tensor:
    batch_values: list[float] = []
    for preds, labels in zip(pred_sequences, label_sequences):
        filtered_preds = [token for token in preds if token not in ignore_set]
        filtered_labels = [token for token in labels if token not in ignore_set]

        pred_set = set(filtered_preds)
        label_set = set(filtered_labels)

        if not label_set:
            batch_values.append(float('nan'))
            continue

        true_positives = len(pred_set & label_set)
        batch_values.append(true_positives / len(label_set))

    return torch.tensor(batch_values, dtype=torch.float32)


def recall(pred_labels: Any, labels: Any, reduction: Literal['mean', 'sum', 'none'] = 'mean', ignore_index: Any | None = None) -> torch.Tensor:
    '''
    Compute the recall of the model's predictions.

    Parameters
    ----------
    pred_labels : torch.Tensor
        The model's predicted labels.
    labels : torch.Tensor
        The ground truth labels.
    reduction : {'mean', 'sum', 'none'}, optional
        The reduction method to apply to the output tensor. Default is 'mean'.
    ignore_index : int, list[int], or None, optional
        The index or indices to ignore in the evaluation (e.g. padding). Default is None.

    Returns
    -------
    torch.Tensor
        The recall scores of the model's predictions.
    '''
    pred_sequences, label_sequences, ignore_set = _prepare_inputs(pred_labels, labels, ignore_index, 'pred_labels')
    recalls = _compute_recall_values(pred_sequences, label_sequences, ignore_set)
    return _apply_reduction(recalls, reduction)


def precision(pred_labels: Any, labels: Any, reduction: Literal['mean', 'sum', 'none'] = 'mean', ignore_index: Any | None = None) -> torch.Tensor:
    '''
    Compute the precision of the model's predictions.

    Parameters
    ----------
    pred_labels : torch.Tensor
        The model's predicted labels.
    labels : torch.Tensor
        The ground truth labels.
    reduction : {'mean', 'sum', 'none'}, optional
        The reduction method to apply to the output tensor. Default is 'mean'.
    ignore_index : int, list[int], or None, optional
        The index or indices to ignore in the evaluation (e.g. padding). Default is None.

    Returns
    -------
    torch.Tensor
        The precision scores of the model's predictions.
    '''
    pred_sequences, label_sequences, ignore_set = _prepare_inputs(pred_labels, labels, ignore_index, 'pred_labels')
    precisions = _compute_precision_values(pred_sequences, label_sequences, ignore_set)
    return _apply_reduction(precisions, reduction)


def f1_score(pred_labels: Any, labels: Any, reduction: Literal['mean', 'sum', 'none'] = 'mean', ignore_index: Any | None = None) -> torch.Tensor:
    '''
    Compute the F1 score of the model's predictions.

    Parameters
    ----------
    pred_labels : torch.Tensor
        The model's predicted labels.
    labels : torch.Tensor
        The ground truth labels.
    reduction : {'mean', 'sum', 'none'}, optional
        The reduction method to apply to the output tensor. Default is 'mean'.
    ignore_index : int, list[int], or None, optional
        The index or indices to ignore in the evaluation (e.g. padding). Default is None.

    Returns
    -------
    torch.Tensor
        The F1 scores of the model's predictions.
    '''
    pred_sequences, label_sequences, ignore_set = _prepare_inputs(pred_labels, labels, ignore_index, 'pred_labels')
    precision_values = _compute_precision_values(pred_sequences, label_sequences, ignore_set)
    recall_values = _compute_recall_values(pred_sequences, label_sequences, ignore_set)

    f1_scores = 2 * (precision_values * recall_values) / (precision_values + recall_values)
    f1_scores[torch.isnan(f1_scores)] = 0

    return _apply_reduction(f1_scores, reduction)


def accuracy(pred_labels: torch.Tensor, labels: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean', ignore_index: int | list[int] | None = None) -> torch.Tensor:
    '''
    Compute the accuracy of the model's predictions.

    Parameters
    ----------
    pred_labels : torch.Tensor
        The model's predicted labels.
    labels : torch.Tensor
        The ground truth labels.
    reduction : {'mean', 'sum', 'none'}, optional
        The reduction method to apply to the output tensor. Default is 'mean'.
    ignore_index : int, list[int], or None, optional
        The index or indices to ignore in the evaluation (e.g. padding). Default is None.

    Returns
    -------
    torch.Tensor
        The accuracy of the model's predictions.
    '''
    if pred_labels.ndim != 2:
        raise ValueError(f"Expected pred_labels to have 2 dimensions (batch_size, sequence_length), got {pred_labels.shape}")

    if labels.ndim != 2:
        raise ValueError(f"Expected labels to have 2 dimensions (batch_size, sequence_length), got {labels.shape}")

    accuracies_list = []

    for pred, lbl in zip(pred_labels, labels):
        if ignore_index is None:
            if len(pred) != len(lbl):
                accuracies_list.append(0.0)
            else:
                accuracies_list.append(torch.all(pred == lbl).float().item())
        else:
            if isinstance(ignore_index, int):
                ignore_index = [ignore_index]
            ignore_indices = torch.tensor(ignore_index, dtype=torch.long, device=labels.device)

            # Pad the shorter sequence with the first ignored index
            if len(pred) < len(lbl):
                padding = torch.full((len(lbl) - len(pred),), ignore_indices[0], dtype=pred.dtype, device=labels.device)  # type: ignore
                pred = torch.cat((pred, padding))
            elif len(lbl) < len(pred):
                padding = torch.full((len(pred) - len(lbl),), ignore_indices[0], dtype=lbl.dtype, device=labels.device)  # type: ignore
                lbl = torch.cat((lbl, padding))

            # Create a combined mask for positions where both pred and lbl should not be ignored
            valid_indices_mask = ~torch.isin(pred, ignore_indices) | ~torch.isin(lbl, ignore_indices)

            # Apply the combined mask to predictions and labels to keep alignment
            pred = pred[valid_indices_mask]
            lbl = lbl[valid_indices_mask]

            accuracies_list.append(torch.all(pred == lbl).float().item())

    accuracies = torch.tensor(accuracies_list)

    # Handle reduction
    match reduction:
        case 'mean':
            return accuracies.mean()
        case 'sum':
            return accuracies.sum()
        case 'none':
            return accuracies
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def perplexity(logits: torch.Tensor, labels: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean', ignore_index: int | None = None) -> torch.Tensor:
    '''
    Compute the perplexity of the model's predictions.

    Parameters
    ----------
    logits : torch.Tensor
        The model's output logits.
    labels : torch.Tensor
        The ground truth labels.
    reduction : {'mean', 'sum', 'none'}, optional
        The reduction method to apply to the output tensor. Default is 'mean'.
    ignore_index : int or None, optional
        The index to ignore in the evaluation (e.g. padding). Default is None.

    Returns
    -------
    torch.Tensor
        The perplexity of the model's predictions.
    '''
    # Flatten logits and labels for computing cross-entropy loss
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)

    # Compute cross-entropy loss, ignoring padding index
    if ignore_index is not None:
        cross_entropy_loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction='none')
    else:
        cross_entropy_loss = F.cross_entropy(logits, labels, reduction='none')

    # Compute perplexity
    perplexity_values = torch.exp(cross_entropy_loss)

    match reduction:
        case 'mean':
            return perplexity_values.mean()
        case 'sum':
            return perplexity_values.sum()
        case 'none':
            return perplexity_values
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")

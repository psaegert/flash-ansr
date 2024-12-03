from typing import Literal

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


def recall(pred_labels: torch.Tensor, labels: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean', ignore_index: int | list[int] | None = None) -> torch.Tensor:
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
    if pred_labels.ndim != 2:
        raise ValueError(f"Expected pred_labels to have 2 dimensions (batch_size, sequence_length), got {pred_labels.shape}")

    if labels.ndim != 2:
        raise ValueError(f"Expected labels to have 2 dimension (batch_size, sequence_length), got {labels.shape}")

    batch_recalls = []

    # Handle ignore_index as a tensor for efficient operations
    if ignore_index is not None:
        if isinstance(ignore_index, int):
            ignore_index = [ignore_index]
        ignore_index_tensor = torch.tensor(ignore_index, dtype=torch.long, device=labels.device)
    else:
        ignore_index_tensor = torch.tensor([], dtype=torch.long, device=labels.device)

    for pred, lbl in zip(pred_labels, labels):
        # Filter out ignored labels
        valid_labels_mask = ~torch.isin(lbl, ignore_index_tensor)
        valid_labels = lbl[valid_labels_mask].unique()
        pred_unique = pred.unique()

        # Calculate true positives: Valid labels that are predicted
        true_positives = valid_labels[torch.isin(valid_labels, pred_unique)].numel()

        # Calculate recall
        if valid_labels.numel() > 0:
            batch_recalls.append(true_positives / valid_labels.numel())
        else:
            batch_recalls.append(float('nan'))  # Handle case where no valid ground truth labels are provided

    recalls = torch.tensor(batch_recalls)

    # Handle reduction
    match reduction:
        case 'mean':
            return torch.nanmean(recalls)
        case 'sum':
            return torch.nansum(recalls)
        case 'none':
            return recalls
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def precision(pred_labels: torch.Tensor, labels: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean', ignore_index: int | list[int] | None = None) -> torch.Tensor:
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
    if pred_labels.ndim != 2:
        raise ValueError(f"Expected pred_labels to have 2 dimensions (batch_size, sequence_length), got {pred_labels.shape}")

    if labels.ndim != 2:
        raise ValueError(f"Expected labels to have 2 dimension (batch_size, sequence_length), got {labels.shape}")

    batch_precisions = []

    # Convert ignore_index into a tensor for efficient operations
    if ignore_index is not None:
        if isinstance(ignore_index, int):
            ignore_index = [ignore_index]
        ignore_index_tensor = torch.tensor(ignore_index, dtype=torch.long, device=labels.device)
    else:
        ignore_index_tensor = torch.tensor([], dtype=torch.long, device=labels.device)

    for pred, lbl in zip(pred_labels, labels):
        # Filter out ignored predictions and labels
        valid_pred_mask = ~torch.isin(pred, ignore_index_tensor)
        valid_preds = pred[valid_pred_mask]
        valid_labels_mask = ~torch.isin(lbl, ignore_index_tensor)
        valid_labels = lbl[valid_labels_mask].unique()

        # Calculate true positives: Predictions that are correct
        true_positives = valid_preds[torch.isin(valid_preds, valid_labels)].numel()

        # Calculate precision
        if valid_preds.numel() > 0:
            batch_precisions.append(true_positives / valid_preds.numel())
        else:
            batch_precisions.append(float('nan'))  # Handle case where no valid predictions are provided

    precisions = torch.tensor(batch_precisions)

    # Handle reduction
    match reduction:
        case 'mean':
            return torch.nanmean(precisions)
        case 'sum':
            return torch.nansum(precisions)
        case 'none':
            return precisions
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def f1_score(pred_labels: torch.Tensor, labels: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean', ignore_index: int | list[int] | None = None) -> torch.Tensor:
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
    if pred_labels.ndim != 2:
        raise ValueError(f"Expected pred_labels to have 2 dimensions (batch_size, sequence_length), got {pred_labels.shape}")

    if labels.ndim != 2:
        raise ValueError(f"Expected labels to have 2 dimension (batch_size, sequence_length), got {labels.shape}")

    # Compute precision and recall from the same logits and labels
    prec = precision(pred_labels, labels, ignore_index=ignore_index, reduction='none')
    rec = recall(pred_labels, labels, ignore_index=ignore_index, reduction='none')

    # Calculate the F1 score using the harmonic mean of precision and recall
    f1_scores = 2 * (prec * rec) / (prec + rec)

    # Handle cases where the denominator might be zero
    f1_scores[torch.isnan(f1_scores)] = 0  # Set NaNs resulting from zero division to 0

    # Handle reduction
    match reduction:
        case 'mean':
            return f1_scores.mean()
        case 'sum':
            return f1_scores.sum()
        case 'none':
            return f1_scores
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


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

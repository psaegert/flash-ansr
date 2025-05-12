import warnings

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from scipy.optimize import curve_fit, OptimizeWarning

from flash_ansr import ExpressionSpace
from flash_ansr.expressions.utils import codify, num_to_constants, safe_f


class AILWQ_ContrastiveLoss(nn.Module):
    """
    https://github.com/AILWQ/Joint_Supervised_Learning_for_SR/blob/main/loss_function/ContrastiveLoss.py
    """
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, enc_features: torch.Tensor, labels: list | torch.Tensor) -> torch.Tensor:
        n = len(labels)
        device = enc_features.device

        # Create labels tensor
        if isinstance(labels, list):
            unique_labels_map = {label: i for i, label in enumerate(set(labels))}
            labels_tensor = torch.tensor([unique_labels_map[label] for label in labels], device=enc_features.device)
        else:
            labels_tensor = labels

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(enc_features.unsqueeze(1), enc_features.unsqueeze(0), dim=2)

        # Create masks for the same class and different class (diagonal and where labels are equal)
        mask = torch.ones_like(similarity_matrix, device=device) * (labels_tensor.expand(n, n).eq(labels_tensor.expand(n, n).t()))
        mask_no_sim = torch.ones_like(mask, device=device) - mask
        mask_diagonal = torch.ones(n, n, device=device) - torch.eye(n, n, device=device)

        # Positive term
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        similarity_matrix = similarity_matrix * mask_diagonal
        sim = mask * similarity_matrix

        # Negative term
        no_sim = similarity_matrix - sim  # [batch_size, batch_size]
        no_sim_sum = torch.sum(no_sim, dim=1)  # [batch_size]
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T  # [batch_size, batch_size]
        sim_sum = no_sim_sum_expend

        if torch.sum(sim_sum) == 0:
            return torch.tensor(0, device=device)

        loss = torch.div(sim, sim_sum)
        loss = mask_no_sim + loss + torch.eye(n, n, device=device)

        # compute loss of a batch
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)) + 1e-5)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5, temperature: float = 0.5) -> None:
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        """
        Compute the contrastive loss based on cosine similarities.

        Args:
            features (torch.Tensor): A tensor of shape (N, D), where N is the number of samples and D is the feature dimension.
            labels (torch.Tensor): A tensor of shape (N,) with integer class labels for each sample.
            margin (float): Margin value for dissimilar pairs. Default is 0.5.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features)

        # Create labels tensor
        if isinstance(labels, list) or isinstance(labels, np.ndarray):
            unique_labels_map = {label: i for i, label in enumerate(set(labels))}
            labels_tensor = torch.tensor([unique_labels_map[label] for label in labels], device=features.device)
        else:
            labels_tensor = labels

        # Normalize features to have unit norm (for cosine similarity)
        features = F.normalize(features, p=2, dim=1)

        # Compute the cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Apply exponentiation with temperature scaling
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)

        # Create a mask to identify positive (same label) and negative (different label) pairs
        labels_tensor = labels_tensor.view(-1, 1)  # Reshape to (N, 1) for broadcasting
        positive_mask = labels_tensor == labels_tensor.T  # Mask for positive pairs
        negative_mask = ~positive_mask  # Mask for negative pairs

        # Row-wise normalization of similarity matrix
        row_sum = similarity_matrix.sum(dim=1, keepdim=True)  # Shape (N, 1)

        # Compute positive loss
        positive_loss = -torch.log((similarity_matrix * positive_mask).sum(dim=1) / row_sum.squeeze()).mean()

        # Compute negative loss
        negative_loss = F.relu(torch.log((similarity_matrix * negative_mask).sum(dim=1) / row_sum.squeeze()) - self.margin).mean()

        # Combine positive and negative losses
        loss = positive_loss + negative_loss

        return loss


def log_FVU(expression_space: ExpressionSpace, X_target: torch.Tensor, y_target: torch.Tensor, skeleton_candidate: list[str]) -> torch.Tensor:
    executable_prefix_expression = expression_space.operators_to_realizations(skeleton_candidate)
    prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression, inplace=True)
    code_string = expression_space.prefix_to_infix(prefix_expression_with_constants, realization=True)
    code = codify(code_string, expression_space.variables + constants)

    f = expression_space.code_to_lambda(code)

    if len(constants) == 0:
        y_pred = f(*X_target.T)
    else:
        def pred_function(X: np.ndarray, *constants: np.ndarray | None) -> np.ndarray:
            if len(constants) == 0:
                y = safe_f(f, X)
            y = safe_f(f, X, constants)

            # If the numbers are complex, return nan
            if np.iscomplexobj(y):
                return np.full((X.shape[0],), np.nan)

            return y

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                best_popt, best_error = None, np.inf
                for k in range(1):
                    p0 = np.random.normal(loc=0, scale=5, size=len(constants))
                    popt, _ = curve_fit(pred_function, X_target, y_target.flatten(), p0=p0)

                    y_pred = f(*X_target.T, *popt)

                    tentative_new_error = (y_pred - y_target).abs().mean()
                    if tentative_new_error < best_error:
                        best_error = tentative_new_error
                        best_popt = popt

                if best_popt is None:
                    raise RuntimeError("Optimization failed to converge")

        except RuntimeError:
            return torch.tensor(torch.nan)

        y_pred = f(*X_target.T, *best_popt)
        if not isinstance(y_pred, np.ndarray) and not isinstance(y_pred, torch.Tensor):
            y_pred = torch.full(X_target.shape[0], y_pred)  # type: ignore

    if y_pred.shape != y_target.shape:
        if y_target.shape[1] == 1:
            y_target = y_target.squeeze(1)
        else:
            raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_target.shape}")

    return torch.log10(torch.mean((y_target - y_pred) ** 2) / torch.mean((y_target - torch.mean(y_target)) ** 2) + torch.finfo(torch.float32).eps)


def log_fvu_to_target_cosine_similarity(log_fvu_matrix: torch.Tensor, min_fvu: float, max_fvu: float) -> torch.Tensor:
    """
    Convert log FVU values to cosine similarity scores.
    """
    # Normalize the log FVU values to a range of [0, 1]
    normalized_fvu = (log_fvu_matrix - min_fvu) / (max_fvu - min_fvu)

    # Convert to cosine similarity
    cosine_similarity = 1 - normalized_fvu

    return cosine_similarity


class CosineSimilarityLogFVUMatchLoss(nn.Module):
    def __init__(self, expression_space: ExpressionSpace, n_samples: int | None = 1):
        super().__init__()
        self.expression_space = expression_space
        self.n_samples = n_samples

    def forward(self, x_target: torch.Tensor, y_target: torch.Tensor, candidate_skeletons: list[list[str]], embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarity between the log FVU values of the target and the candidate skeletons.
        """
        # Get the number of samples
        if self.n_samples is None:
            n_samples = x_target.shape[0]
        else:
            n_samples = self.n_samples

        # Compute the log FVU values
        log_fvu_matrix = torch.zeros((n_samples, len(candidate_skeletons)), dtype=torch.float32, device=x_target.device)
        for i in range(n_samples):
            for j in range(len(candidate_skeletons)):
                if i != j:
                    log_fvu_matrix[i, j] = log_FVU(self.expression_space, x_target[i].cpu(), y_target[i].cpu(), candidate_skeletons[j])

        # Convert to cosine similarity
        target_cosine_similarity_matrix = log_fvu_to_target_cosine_similarity(log_fvu_matrix, -7, 7)

        # Nan means no constants could be found to fit the data, so the skeletons must be very different
        target_cosine_similarity_matrix[~torch.isfinite(target_cosine_similarity_matrix)] = -1
        target_cosine_similarity_matrix = target_cosine_similarity_matrix.fill_diagonal_(1)

        # Compute the cosine similarity between the actual embeddings
        cosine_similarity_matrix = torch.nn.functional.cosine_similarity(embeddings[:self.n_samples, ...].unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

        # Compute the loss
        loss = torch.nn.functional.mse_loss(cosine_similarity_matrix, target_cosine_similarity_matrix)
        return loss

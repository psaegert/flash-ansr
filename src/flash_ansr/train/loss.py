import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


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

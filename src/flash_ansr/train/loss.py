import torch
from torch import nn
import torch.nn.functional as F


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
    def __init__(self, temperature: float = 0.07) -> None:
        """
        Standard implementation of Supervised Contrastive Loss.
        https://arxiv.org/abs/2004.11362
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): A tensor of shape (N, D). Assumed to be L2-normalized.
            labels (torch.Tensor): A tensor of shape (N,) with integer class labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        device = features.device
        batch_size = features.shape[0]

        # Create a mask to identify positive pairs
        labels = labels.contiguous().view(-1, 1)
        # positive_mask[i, j] is True if features[i] and features[j] have the same label
        positive_mask = torch.eq(labels, labels.T).float().to(device)

        # Create a mask to exclude self-similarity (the diagonal)
        # off_diagonal_mask[i, j] is True if i != j
        off_diagonal_mask = ~torch.eye(batch_size, device=device, dtype=torch.bool)

        # Compute cosine similarity
        similarity_matrix = torch.matmul(features, features.T)

        # --- Isolate positive pairs (excluding self) ---
        # We want positive pairs that are not the sample itself
        positive_mask = positive_mask * off_diagonal_mask

        # --- Numerator of SupCon Loss ---
        # For each anchor, we want the log-probabilities of its positive samples.
        # The logits are the scaled similarities.
        logits = similarity_matrix / self.temperature

        # To avoid numerical instability, we use the log-sum-exp trick.
        # For each row i, subtract the max logit value for stability.
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # --- Denominator of SupCon Loss ---
        # The denominator is the sum of exponentiated similarities over all other samples.
        exp_logits = torch.exp(logits) * off_diagonal_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # --- Compute the loss ---
        # The loss is the mean of the negative log-probabilities of the positive pairs.
        # For each anchor, we average the log-probabilities over its positive samples.
        # mean_log_prob_pos[i] = sum_{p in P(i)} log_prob[i, p] / |P(i)|
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / torch.clamp(positive_mask.sum(1), min=1.0)

        # The final loss is the negative of this value, averaged over the batch.
        loss = -mean_log_prob_pos.mean()

        return loss

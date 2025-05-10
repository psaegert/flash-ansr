from typing import Any, Union

import torch
from torch import nn

from flash_ansr.models.encoders.set_transformer import SAB, MAB
from flash_ansr.models.encoders.set_encoder import SetEncoder
from flash_ansr.models.encoders.pre_encoder import PreEncoder


def split_mask_batched(X: torch.Tensor) -> torch.Tensor:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {X.ndim}D tensor of shape {X.shape}")

    batch_size, num_points, _ = X.shape

    # Find dimension with max range for each batch
    mins = X.min(dim=1).values
    maxs = X.max(dim=1).values
    ranges = maxs - mins
    max_range_dim = ranges.argmax(dim=1)

    # Initialize result mask
    result_mask = torch.zeros((batch_size, num_points), dtype=torch.bool, device=X.device)

    # Process each batch independently
    for i in range(batch_size):
        # Extract values along the max range dimension for this batch
        values = X[i, :, max_range_dim[i]]

        # Calculate target size for the first half
        target_size = num_points // 2

        # Find median value (equivalent to finding kth element)
        median_value = torch.kthvalue(values, target_size + 1).values

        # Create masks for each category
        less_mask = values < median_value
        equal_mask = values == median_value

        # Count elements in each category
        less_count = torch.sum(less_mask).item()
        equal_count = torch.sum(equal_mask).item()

        # Calculate how many median elements go to first half to maintain balance
        median_in_first = max(0, target_size - less_count)

        # If we need to split the median elements
        if 0 < median_in_first < equal_count:
            # Get indices of the median elements
            equal_indices = torch.nonzero(equal_mask, as_tuple=True)[0]

            # Add the required number of median elements to the less mask
            result_mask[i, less_mask] = True
            result_mask[i, equal_indices[:median_in_first]] = True  # type: ignore
        else:
            # Simple case: all less than median, and possibly some equal to median
            result_mask[i] = values <= median_value if less_count < target_size else less_mask

    return result_mask


class AABBNode:
    def __init__(self, X: torch.Tensor, left: Union["AABBNode", None] = None, right: Union["AABBNode", None] = None):
        self.X = X
        self.left = left
        self.right = right

    @property
    def shape(self) -> tuple[int, ...]:
        return self.X.shape


def create_tree_batched(X: torch.Tensor, max_depth: int = 5, depth: int = 0) -> AABBNode:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {X.ndim}D tensor of shape {X.shape}")

    if depth + 1 >= max_depth:
        return AABBNode(X)

    # print(f"Creating tree at depth {depth + 1} with shape {X.shape}")

    if X.shape[1] > 1:
        mask_batched = split_mask_batched(X)
        # print(f"Mask shape: {mask_batched.shape}")
        X_left = torch.stack([X[i][mask_batched[i]] for i in range(X.shape[0])], dim=0)
        X_right = torch.stack([X[i][~mask_batched[i]] for i in range(X.shape[0])], dim=0)
        return AABBNode(X, create_tree_batched(X_left, max_depth, depth + 1), create_tree_batched(X_right, max_depth, depth + 1))

    return AABBNode(X, create_tree_batched(X, max_depth, depth + 1), create_tree_batched(X, max_depth, depth + 1))


class AABBTreeSetTransformer(SetEncoder):
    def __init__(
            self,
            input_embedding_size: int,
            input_dimension_size: int,
            output_embedding_size: int,
            depth: int,
            hidden_size: int = 512,
            n_heads: int = 8,
            clean_path: bool = True,
            pre_encoder_kwargs: dict[str, Any] | None = None):
        super().__init__()

        self.depth = depth
        self.hidden_size = hidden_size

        if pre_encoder_kwargs is None:
            pre_encoder_kwargs = {}

        self.pre_encoder = PreEncoder(input_size=input_dimension_size, **pre_encoder_kwargs)

        self.linear_in = nn.Linear(self.pre_encoder.output_size, hidden_size)

        self.sabs = nn.ModuleList([SAB(hidden_size, hidden_size, n_heads=n_heads, clean_path=clean_path) for _ in range(depth)])
        self.mabs_lr = nn.ModuleList([MAB(hidden_size, hidden_size, hidden_size, n_heads=n_heads, clean_path=clean_path) for _ in range(depth)])
        self.mabs_rl = nn.ModuleList([MAB(hidden_size, hidden_size, hidden_size, n_heads=n_heads, clean_path=clean_path) for _ in range(depth)])

        self.linear_out = nn.Linear(hidden_size, output_embedding_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim > 3:
            raise ValueError(f"Expected 2D or 3D tensor, got {X.ndim}D tensor of shape {X.shape}")
        elif X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got {X.ndim}D tensor of shape {X.shape}")

        tree_batched = create_tree_batched(X.cpu(), max_depth=self.depth)
        tree_encoded = self.encode_tree(tree_batched)

        return self.linear_out(tree_encoded)

    def encode_tree(self, tree: AABBNode, depth: int = 0) -> torch.Tensor:
        # Leaf node
        if tree.left is None or tree.right is None:
            X = tree.X.to(self.linear_in.weight.device)

            if self.pre_encoder is not None:
                X = self.pre_encoder(X)
                B, M, D, E = X.shape
                X = X.view(B, M, D * E)

            return self.linear_in(X)

        X_left = self.encode_tree(tree.left, depth + 1)
        X_right = self.encode_tree(tree.right, depth + 1)

        X_left = self.sabs[depth](X_left)
        X_right = self.sabs[depth](X_right)

        X_lr = self.mabs_lr[depth](X_left, X_right)

        X_lr_rl = self.mabs_rl[depth](X_right, X_lr)

        return X_lr_rl

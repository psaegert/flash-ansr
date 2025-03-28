from flash_ansr.models.encoders.set_encoder import SetEncoder
from flash_ansr.models.transformer_utils import PositionalEncoding
import torch
from torch import nn
import torch.nn.functional as F
import math
import string

# FIXME. Highly experimental. Not tested.


class AxisAttention(nn.Module):
    def __init__(
        self,
        dim_Q: int,
        dim_KV: int,
        dim_attn: int,
        dropout: float = 0.0,
        axis: int = 1,
    ) -> None:
        """
        Initialize an AxisAttention module.

        Args:
            dim_Q: Input dimension of query tensor
            dim_KV: Input dimension of key-value tensor
            dim_attn: Internal attention dimension
            dropout: Dropout probability after softmax
            axis: Axis to perform attention over (default: 1)
            use_residual: Whether to use residual connection
        """
        super().__init__()

        self.dim_attn = dim_attn
        self.axis = axis

        # Projection layers
        self.proj_q = nn.Linear(dim_Q, dim_attn)
        self.proj_k = nn.Linear(dim_KV, dim_attn)
        self.proj_v = nn.Linear(dim_KV, dim_attn)

        # Output projection
        self.proj_out = nn.Linear(dim_attn, dim_Q)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        self.cached_expressions: dict = {}

        self.attention_normalizer = math.sqrt(dim_attn)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Perform attention along the specified axis.

        Args:
            query: Query tensor of shape (..., dim_Q)
            key_value: Key-value tensor of shape (..., dim_KV)
            verbose: Whether to print intermediate shapes

        Returns:
            Output tensor with the same shape as query
        """
        # Project inputs to attention space
        q = self.proj_q(query)      # (..., dim_attn)
        k = self.proj_k(key_value)  # (..., dim_attn)
        v = self.proj_v(key_value)  # (..., dim_attn)

        if len(self.cached_expressions) != 5:
            # Ensure axis is positive
            axis = self.axis if self.axis >= 0 else query.ndim + self.axis
            if axis == query.ndim or axis == key_value.ndim:
                raise ValueError(f"Axis cannot be the last dimension of the tensors. Got {axis=} alghough the tensors have {query.ndim=} and {key_value.ndim=} dimensions.")

            # Generate einsum notation
            ndim = query.ndim

            # Create dimension labels using lowercase letters for all dimensions
            available_letters = string.ascii_lowercase[3:]
            dims = available_letters[:ndim - 1]

            # Build einsum expressions
            batch_dims = ''.join([dims[i] for i in range(ndim - 1) if i != axis])
            self.cached_expressions["q_expr"] = batch_dims[:axis] + ('a' if axis < ndim - 1 else '') + batch_dims[axis:] + 'c'
            self.cached_expressions["k_expr"] = batch_dims[:axis] + ('b' if axis < ndim - 1 else '') + batch_dims[axis:] + 'c'
            self.cached_expressions["v_expr"] = batch_dims[:axis] + ('b' if axis < ndim - 1 else '') + batch_dims[axis:] + 'c'

            # Output should match query's shape
            self.cached_expressions["out_expr"] = batch_dims[:axis] + ('a' if axis < ndim - 1 else '') + batch_dims[axis:] + 'c'

            # Attention map expression
            self.cached_expressions["attn_expr"] = batch_dims[:axis] + ('ab' if axis < ndim - 1 else '') + batch_dims[axis:]

        # Compute attention scores
        if verbose:
            print(f'{self.cached_expressions["q_expr"]},{self.cached_expressions["k_expr"]}->{self.cached_expressions["attn_expr"]}')
        attn = torch.einsum(f'{self.cached_expressions["q_expr"]},{self.cached_expressions["k_expr"]}->{self.cached_expressions["attn_expr"]}', q, k) * self.attention_normalizer

        # Apply softmax along the key dimension (last dimension of attention tensor)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        if verbose:
            print(f'{self.cached_expressions["attn_expr"]},{self.cached_expressions["v_expr"]}->{self.cached_expressions["out_expr"]}')
        output = torch.einsum(f'{self.cached_expressions["attn_expr"]},{self.cached_expressions["v_expr"]}->{self.cached_expressions["out_expr"]}', attn, v)

        # Add residual connection if requested
        output = query + self.proj_out(output)

        return output


class AxisSAB(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, input_size: int, output_size: int, n_heads: int, axis: int = 1) -> None:
        super().__init__()
        self.mab = AxisAttention(input_size, input_size, output_size, axis=axis)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class AxisISAB(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, input_size: int, output_size: int, n_heads: int, n_induce: int, axis: int = 1) -> None:
        super().__init__()

        self.inducing_points = nn.Parameter(torch.Tensor(n_induce, output_size))
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab0 = AxisAttention(output_size, input_size, output_size, axis=axis)
        self.mab1 = AxisAttention(input_size, output_size, output_size, axis=axis)

        self.axis = axis
        self.n_induce = n_induce

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Format inducing points to align with the attention axis
        formatted_induction_points = self._format_inducing_points(X)

        # Apply attention mechanisms
        H = self.mab0(formatted_induction_points, X)
        return self.mab1(X, H)

    def _format_inducing_points(self, X: torch.Tensor) -> torch.Tensor:
        """
        Format inducing points to align with the attention axis of X.

        Args:
            X: Input tensor of shape (..., input_size)

        Returns:
            Formatted inducing points tensor with shape matching X except:
            - The attention axis dimension is replaced with n_induce
            - The last dimension is output_size
        """
        # Ensure axis is positive
        axis = self.axis if self.axis >= 0 else X.ndim + self.axis

        # Start with the basic inducing points: [n_induce, output_size]
        points = self.inducing_points

        # Create the target shape:
        # - Same as X for all batch dimensions before the attention axis
        # - n_induce at the attention axis
        # - Same as X for all batch dimensions after the attention axis
        # - output_size as the last dimension
        target_shape = list(X.shape)
        target_shape[axis] = self.n_induce

        # Create a list of dimensions to expand
        expand_shape = list(target_shape)

        # Build reshape pattern by adding singleton dimensions
        view_shape = [1] * (X.ndim - 1)  # -1 because points already has the last dimension
        view_shape[axis] = self.n_induce

        # Reshape inducing points to have singleton dimensions in all batch dims
        points = points.view(*view_shape, -1)  # type: ignore

        # Expand to match the target shape (efficiently reuses memory)
        points = points.expand(*expand_shape)  # type: ignore

        return points


class AxisPMA(nn.Module):
    """
    Pooling by Multihead Attention with axis support.
    Adapted from https://github.com/juho-lee/set_transformer
    """
    def __init__(self, size: int, n_heads: int, n_seeds: int, axis: int = 1) -> None:
        """
        Initialize a Pooling by Multihead Attention module with axis support.

        Args:
            size: Dimension of input and output features
            n_heads: Number of attention heads
            n_seeds: Number of seed vectors (output set size)
            axis: Axis to perform attention over (default: 1)
        """
        super().__init__()

        self.S = nn.Parameter(torch.Tensor(n_seeds, size))
        nn.init.xavier_uniform_(self.S)

        self.mab = AxisAttention(size, size, size, axis=axis)
        self.axis = axis
        self.n_seeds = n_seeds

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling attention to input tensor.

        Args:
            X: Input tensor of shape (..., size)

        Returns:
            Output tensor with n_seeds elements along the attention axis
        """
        # Format seed vectors to align with the attention axis
        formatted_seeds = self._format_seeds(X)

        # Apply attention mechanism
        return self.mab(formatted_seeds, X)

    def _format_seeds(self, X: torch.Tensor) -> torch.Tensor:
        """
        Format seed vectors to align with the attention axis of X.

        Args:
            X: Input tensor of shape (..., size)

        Returns:
            Formatted seed vectors with shape matching X's batch dimensions,
            n_seeds at the attention axis, and size as the feature dimension
        """
        # Ensure axis is positive
        axis = self.axis if self.axis >= 0 else X.ndim + self.axis

        # Get the batch shape (all dimensions except the last one)
        batch_shape = list(X.shape[:-1])

        # Create the target shape for the seeds:
        # - Same as X for all batch dimensions except the attention axis
        # - n_seeds at the attention axis
        # - size as the last dimension
        target_shape = batch_shape.copy()
        target_shape[axis] = self.n_seeds

        # Create a view shape with singleton dimensions for all batch dims
        view_shape = [1] * len(batch_shape)
        view_shape[axis] = self.n_seeds

        # Reshape seed vectors to have singleton dimensions in all batch dims
        seeds = self.S.view(*view_shape, -1)

        # Expand to match the target shape (efficiently reuses memory)
        seeds = seeds.expand(*target_shape, X.shape[-1])

        return seeds


class AlternatingSetTransformer(SetEncoder):
    # https://github.com/juho-lee/set_transformer
    def __init__(
            self,
            input_embedding_size: int,
            input_dimension_size: int,
            output_embedding_size: int,
            n_seeds: int,
            hidden_size: int = 512,
            n_enc_isab: int = 2,
            n_dec_sab: int = 2,
            n_induce: int | list[int] = 64,
            n_heads: int = 4,
            add_positional_encoding: bool = True) -> None:
        super().__init__()
        if n_enc_isab < 1:
            raise ValueError(f"Number of ISABs in encoder `n_enc_isab` ({n_enc_isab}) must be greater than 0")

        if n_dec_sab < 0:
            raise ValueError(f"Number of SABs in decoder `n_dec_sab` ({n_dec_sab}) cannot be negative")

        if isinstance(n_induce, int):
            n_induce = [n_induce] * n_enc_isab
        elif len(n_induce) != n_enc_isab:
            raise ValueError(
                f"Number of inducing points `n_induce` ({n_induce}) must be an integer or a list of length {n_enc_isab}")

        self.linear_in = nn.Linear(input_embedding_size, hidden_size)
        self.enc = nn.Sequential(*[AxisISAB(hidden_size, hidden_size, n_heads, n_induce[i], axis=1 + (i % 2)) for i in range(n_enc_isab)])
        self.pma = AxisPMA(hidden_size, n_heads, n_seeds, axis=1)
        self.dec = nn.Sequential(*[AxisSAB(hidden_size, hidden_size, n_heads, axis=1 + (i % 2)) for i in range(n_dec_sab)])
        self.linear_out = nn.Linear(hidden_size, output_embedding_size)
        self.positional_encoding_out = PositionalEncoding()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.linear_out(self.dec(self.pma(self.enc(self.linear_in(X)))))

        B, M, D, E = out.shape

        out = out + self.positional_encoding_out(shape=(D, E), device=out.device)

        return out

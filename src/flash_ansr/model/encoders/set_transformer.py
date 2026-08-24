"""Set Transformer encoder building blocks."""
import warnings
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flash_ansr.model.common import FeedForward, SetNormBase, get_norm_layer
from flash_ansr.model.encoders.base import SetEncoder


class MultiheadAttentionBlock(nn.Module):
    """Multi-head attention building block.

    Parameters
    ----------
    dim_q : int
        Dimensionality of the query input.
    dim_kv : int
        Dimensionality of the key/value input.
    dim_out : int
        Output dimensionality of the attention projection.
    n_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability applied inside scaled dot-product attention. Defaults to ``0.0``.
    bias : bool, optional
        Whether to include bias terms in the projections. Defaults to ``True``.
    is_self_attention : bool, optional
        When ``True``, query/key/value share projections and ``dim_q`` must equal ``dim_kv``. Defaults to ``False``.
    query_is_projected : bool, optional
        Expect the provided query to already match ``dim_out`` (used by modules with learned seed vectors). Defaults to ``False``.
    """
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_out: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        is_self_attention: bool = False,
        query_is_projected: bool = False
    ):
        super().__init__()
        if dim_out % n_heads != 0:
            raise ValueError(f"dim_out ({dim_out}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.head_dim = dim_out // n_heads
        self.dropout = dropout
        self.is_self_attention = is_self_attention
        self.query_is_projected = query_is_projected

        if self.is_self_attention:
            if dim_q != dim_kv:
                raise ValueError("For self-attention, dim_q must be equal to dim_kv.")
            self.w_qkv = nn.Linear(dim_q, 3 * dim_out, bias=bias)
        else:
            if self.query_is_projected:
                # If query is pre-projected, its dimension must match dim_out
                if dim_q != dim_out:
                    raise ValueError(f"If query_is_projected, dim_q ({dim_q}) must equal dim_out ({dim_out})")
                self.w_q = nn.Identity()
            else:
                self.w_q = nn.Linear(dim_q, dim_out, bias=bias)  # type: ignore
            self.w_kv = nn.Linear(dim_kv, 2 * dim_out, bias=bias)

        self.w_o = nn.Linear(dim_out, dim_out, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention between ``query`` and ``key_value``.

        Parameters
        ----------
        query : torch.Tensor
            Tensor of shape ``(batch, len_q, dim_q)`` containing the query set.
        key_value : torch.Tensor
            Tensor of shape ``(batch, len_kv, dim_kv)`` providing keys and values.
        attn_mask : torch.Tensor, optional
            Optional mask broadcastable to ``(batch, n_heads, len_q, len_kv)``. ``0`` entries
            are treated as padded positions and excluded from attention.

        Returns
        -------
        torch.Tensor
            Attention output of shape ``(batch, len_q, dim_out)``.
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]

        if self.is_self_attention:
            q, k, v = self.w_qkv(query).chunk(3, dim=-1)
        else:
            q = self.w_q(query)  # This will be an identity operation if query_is_projected
            k, v = self.w_kv(key_value).chunk(2, dim=-1)

        q = q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # Reshape mask for PyTorch's SDPA (scaled dot-product attention).
        # Input mask is (B, L_kv). We need it to be broadcastable to (B, H, L_q, L_kv).
        # Reshaping to (B, 1, 1, L_kv) achieves this broadcasting while keeping head sharing cheap.
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(batch_size, 1, 1, seq_len_kv)
            # If the mask is for self-attention (e.g., 4D), we can use it directly.

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        return self.w_o(attn_output)


class MAB(nn.Module):
    """Multi-head attention block with residual feed-forward refinement.

    Parameters
    ----------
    dim_q : int
        Dimensionality of the query input.
    dim_kv : int
        Dimensionality of the key/value input.
    dim : int
        Output dimensionality of the block.
    n_heads : int
        Number of attention heads.
    ffn_hidden_dim : int, optional
        Hidden dimensionality of the feed-forward network. Defaults to ``None`` which
        selects an implementation-specific width.
    dropout : float, optional
        Dropout probability applied to attention and feed-forward layers. Defaults to ``0.0``.
    use_checkpointing : bool, optional
        Enable gradient checkpointing to reduce memory footprint during training. Defaults to ``False``.
    is_self_attention : bool, optional
        Treat the block as self-attention when ``True``. Defaults to ``False``.
    query_is_projected : bool, optional
        Whether the query already matches the output dimensionality. Defaults to ``False``.
    attn_norm : str, optional
        Normalization strategy for attention inputs. Defaults to ``"none"``.
    ffn_norm : str, optional
        Normalization strategy for feed-forward inputs. Defaults to ``"none"``.
    norm_position : str, optional
        Either ``"pre"`` (Pre-LN: ``x + sub(norm(x))``) or ``"post"`` (Post-LN:
        ``norm(x + sub(x))``). Defaults to ``"pre"``. In post-norm mode the
        residual is normalized once after the addition; ``norm_kv`` is left
        unused on the key/value stream to mirror standard Post-LN Transformers.
    mask_query_norms : bool, optional
        When ``True`` and a ``query_mask`` is passed to :meth:`forward`, the query/residual-stream
        norms (``norm_q``/``norm_ffn``) receive that mask so SetNorm statistics are computed over
        valid query rows only, and the attention and feed-forward outputs are zeroed on padded
        query rows so bias terms cannot write garbage into the residual stream. ``query_mask``
        must be 2D ``(batch, len_q)``. Defaults to ``False`` (legacy
        behavior: when the query is a zero-padded set, its SetNorm statistics include the padding,
        which understates the RMS by ``sqrt(n_valid / set_len)`` and inflates valid rows by the
        inverse factor; existing checkpoints were trained with this behavior).
    """
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim: int,
        n_heads: int,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
        is_self_attention: bool = False,
        query_is_projected: bool = False,  # New flag
        attn_norm: str = "none",
        ffn_norm: str = "none",
        norm_position: str = "pre",
        mask_query_norms: bool = False,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.mask_query_norms = mask_query_norms

        norm_position_l = norm_position.lower()
        if norm_position_l not in ("pre", "post"):
            raise ValueError(f"norm_position must be 'pre' or 'post', got {norm_position!r}")
        self.norm_position = norm_position_l

        self.norm_q = get_norm_layer(attn_norm, dim_q)
        self.norm_kv = get_norm_layer(attn_norm, dim_kv)

        self.attention = MultiheadAttentionBlock(
            dim_q=dim_q,
            dim_kv=dim_kv,
            dim_out=dim,
            n_heads=n_heads,
            dropout=dropout,
            is_self_attention=is_self_attention,
            query_is_projected=query_is_projected
        )

        # In post-norm, the residual norm on the attention sub-layer must match
        # the *output* dimensionality (== dim), not dim_q. For self-attention or
        # query_is_projected blocks we have dim_q == dim, but for cross-attention
        # MABs (e.g. ISAB.mab_cross, PMA.mab) we need a separate norm.
        if self.norm_position == "post" and dim_q != dim:
            self.norm_residual_attn = get_norm_layer(attn_norm, dim)
        else:
            self.norm_residual_attn = None

        self.norm_ffn = get_norm_layer(ffn_norm, dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def _norm_query_stream(self, norm: nn.Module, x: torch.Tensor, query_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply a query/residual-stream norm, mask-aware when ``mask_query_norms`` is enabled."""
        if self.mask_query_norms and query_mask is not None and isinstance(norm, SetNormBase):
            return norm(x, attn_mask=query_mask)
        return norm(x)

    def _forward(self, query: torch.Tensor, key_value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Internal forward pass supporting gradient checkpointing."""
        row_mask: Optional[torch.Tensor] = None
        if self.mask_query_norms and query_mask is not None:
            if query_mask.dim() != 2:
                raise ValueError(
                    f"query_mask must be 2D (batch, len_q); got shape {tuple(query_mask.shape)}. "
                    f"Higher-rank attention masks cannot be reused as query-row masks."
                )
            if query_mask.dtype != torch.bool:
                query_mask = query_mask != 0
            row_mask = query_mask.unsqueeze(-1)
        if self.norm_position == "pre":
            # Pre-LN: x + sub(norm(x))
            q_norm = self._norm_query_stream(self.norm_q, query, query_mask)

            # Mask padded elements in the key/value before normalization if using SetNorms
            if isinstance(self.norm_kv, SetNormBase):
                kv_norm = self.norm_kv(key_value, attn_mask=attn_mask)
            else:
                kv_norm = self.norm_kv(key_value)

            attn_output = self.attention(q_norm, kv_norm, attn_mask=attn_mask)
            if row_mask is not None:
                # Padded query rows produce nonzero attention outputs through the projection
                # biases; zero them so the residual stream stays exactly zero on padding and
                # downstream set statistics see only valid rows.
                attn_output = attn_output * row_mask
            query = query + attn_output

            q_norm = self._norm_query_stream(self.norm_ffn, query, query_mask)
            ffn_output = self.ffn(q_norm)
            if row_mask is not None:
                # No-op for RMSSetNorm (norm(0)=0, bias-free FFN keeps zeros), but load-bearing
                # for norms with a shift (e.g. OriginalSetNorm's beta maps zero rows to beta).
                ffn_output = ffn_output * row_mask
            query = query + ffn_output
        else:
            # Post-LN: norm(x + sub(x)). norm_kv is intentionally unused on the
            # key/value stream (mirrors standard Post-LN Transformers). With
            # mask_query_norms enabled, the residual-stream norms receive the
            # query mask (the query is the padded set in self-refinement blocks);
            # otherwise legacy behavior is preserved.
            attn_output = self.attention(query, key_value, attn_mask=attn_mask)
            if row_mask is not None:
                attn_output = attn_output * row_mask
            residual = query + attn_output
            norm_attn = self.norm_residual_attn if self.norm_residual_attn is not None else self.norm_q
            query = self._norm_query_stream(norm_attn, residual, query_mask)

            ffn_output = self.ffn(query)
            if row_mask is not None:
                ffn_output = ffn_output * row_mask
            query = self._norm_query_stream(self.norm_ffn, query + ffn_output, query_mask)
        return query

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run attention and feed-forward refinement with optional checkpointing.

        Parameters
        ----------
        query : torch.Tensor
            Tensor of shape ``(batch, len_q, dim_q)`` containing the queries.
        key_value : torch.Tensor
            Tensor of shape ``(batch, len_kv, dim_kv)`` containing keys and values.
        attn_mask : torch.Tensor, optional
            Optional padding mask broadcastable to ``(batch, n_heads, len_q, len_kv)``.
            Describes the KEY/VALUE stream (masked keys are excluded from attention and
            from SetNorm statistics on the kv side).
        query_mask : torch.Tensor, optional
            Optional padding mask of shape ``(batch, len_q)`` describing the QUERY stream.
            Only used when the block was built with ``mask_query_norms=True``: SetNorm
            statistics on the query/residual stream are then computed over valid rows only,
            and sub-layer outputs are zeroed on padded query rows.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, len_q, dim)``.
        """
        if self.use_checkpointing and self.training:
            # Pass masks through checkpointing
            return checkpoint(self._forward, query, key_value, attn_mask, query_mask, use_reentrant=False)
        else:
            return self._forward(query, key_value, attn_mask, query_mask)


class SAB(nn.Module):
    """Self-attention block operating on set elements."""

    def __init__(self, dim: int, n_heads: int, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0, attn_norm: str = "none", ffn_norm: str = "none", use_checkpointing: bool = False, norm_position: str = "pre"):
        super().__init__()
        self.mab = MAB(
            dim_q=dim, dim_kv=dim, dim=dim, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=use_checkpointing,
            is_self_attention=True,
            attn_norm=attn_norm,
            ffn_norm=ffn_norm,
            norm_position=norm_position,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to a dense set representation.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, set_len, dim)`` containing set elements with no padding.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, set_len, dim)`` after self-attention.
        """
        # SAB operates on the dense, fixed-size set emitted by PMA, so no mask is required here.
        return self.mab(x, x)


class ISAB(nn.Module):
    """Induced Set Attention Block with shared inducing points across the batch.

    ``mask_query_norms`` only affects ``mab_self``, where the (possibly zero-padded) input set
    is the QUERY stream: with the flag enabled, the set's padding mask is forwarded to the
    query/residual-stream SetNorms so their shared statistics are computed over valid rows only
    (legacy behavior counts the padding in the denominator, inflating valid rows by
    ``sqrt(set_len / n_valid)``). ``mab_cross`` needs no query mask — its queries are the dense
    inducing points — and already receives the set mask on its key/value stream.
    """

    def __init__(self, dim_in: int, dim_out: int, n_heads: int, n_inducing_points: int, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0, attn_norm: str = "none", ffn_norm: str = "none", use_checkpointing: bool = False, norm_position: str = "pre", mask_query_norms: bool = False):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing_points, dim_out))
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab_cross = MAB(
            dim_q=dim_out, dim_kv=dim_in, dim=dim_out, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=use_checkpointing, is_self_attention=False,
            attn_norm=attn_norm, ffn_norm=ffn_norm,
            query_is_projected=True,
            norm_position=norm_position,
        )
        self.mab_self = MAB(
            dim_q=dim_in, dim_kv=dim_out, dim=dim_out, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=use_checkpointing, is_self_attention=False,
            attn_norm=attn_norm, ffn_norm=ffn_norm,
            norm_position=norm_position,
            mask_query_norms=mask_query_norms,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transform an input set using inducing points.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, set_len, dim_in)`` representing the input set.
        attn_mask : torch.Tensor, optional
            Padding mask with ``1`` for valid elements and ``0`` for padded ones. Can be
            of shape ``(batch, set_len)`` or broadcastable to ``(batch, 1, 1, set_len)``;
            with ``mask_query_norms=True`` the mask is also reused as the query-row mask
            and must then be 2D ``(batch, set_len)`` (enforced).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, set_len, dim_out)`` containing the refined set representation.
        """
        batch_size = x.shape[0]
        inducing = self.inducing_points.expand(batch_size, -1, -1)

        # Inducing points attend to the input set x. Mask is applied to x (key/value).
        h = self.mab_cross(inducing, x, attn_mask=attn_mask)

        # Input set x attends to the dense inducing point representation h.
        # No mask is needed for the attention calculation itself, as h (key/value) is not padded.
        # The set's padding mask is passed as query_mask so that (when mask_query_norms is
        # enabled) the query/residual-stream SetNorms exclude padded rows from their statistics.
        out = self.mab_self(x, h, query_mask=attn_mask)

        # Zero out the outputs corresponding to padded inputs to prevent information leakage
        # in residual connections and subsequent layers.
        if attn_mask is not None:
            out = out * attn_mask.unsqueeze(-1)

        return out


class PMA(nn.Module):
    """Pooling by multi-head attention module turning sets into fixed-size summaries."""

    def __init__(self, dim: int, n_heads: int, n_seeds: int, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0, attn_norm: str = "none", ffn_norm: str = "none", use_checkpointing: bool = False, norm_position: str = "pre"):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, n_seeds, dim))
        nn.init.xavier_uniform_(self.seed_vectors)

        self.mab = MAB(
            dim_q=dim, dim_kv=dim, dim=dim, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=use_checkpointing, is_self_attention=False,
            attn_norm=attn_norm, ffn_norm=ffn_norm,
            query_is_projected=True,
            norm_position=norm_position,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate a variable-sized set into ``n_seeds`` summary vectors.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, set_len, dim)`` describing the input set.
        attn_mask : torch.Tensor, optional
            Padding mask broadcastable to ``(batch, 1, 1, set_len)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, n_seeds, dim)`` representing pooled set summaries.
        """
        batch_size = x.shape[0]
        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        # Seeds attend to the input set x. Mask is applied to x (key/value).
        return self.mab(seeds, x, attn_mask=attn_mask)


class SetTransformer(SetEncoder):
    """Implementation of the Set Transformer architecture.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each element in the input set.
    output_dim : int or None
        Desired dimensionality for the final output. When ``None`` the model returns ``model_dim``.
    model_dim : int, optional
        Internal dimensionality used throughout the network. Defaults to ``256``.
    n_heads : int, optional
        Number of attention heads used in each block. Defaults to ``8``.
    n_isab : int, optional
        Number of induced set attention blocks. Defaults to ``2``.
    n_sab : int, optional
        Number of decoder self-attention blocks. Defaults to ``1``.
    n_inducing_points : int or list of int, optional
        Number of inducing points per ISAB. Passing an integer shares the value across blocks.
        Defaults to ``32``.
    n_seeds : int, optional
        Number of seed vectors in the pooling module. Defaults to ``1``.
    ffn_hidden_dim : int, optional
        Hidden dimensionality of feed-forward layers. Defaults to ``None`` for automatic choice.
    dropout : float, optional
        Dropout probability for attention and feed-forward layers. Defaults to ``0.0``.
    attn_norm : str, optional
        Normalization applied before attention layers. Defaults to ``"none"``.
    ffn_norm : str, optional
        Normalization applied before feed-forward layers. Defaults to ``"none"``.
    output_norm : str, optional
        Final normalization layer type. Defaults to ``"none"``.
    use_checkpointing : bool, optional
        Whether to apply gradient checkpointing in attention blocks. Defaults to ``False``.
    mask_query_norms : bool, optional
        Thread the set padding mask into the ISAB self-refinement blocks' query/residual-stream
        SetNorms so their statistics exclude zero-padding (see :class:`MAB`). Defaults to
        ``False`` — the legacy behavior existing checkpoints were trained with.
    """

    def __init__(
        self, input_dim: int, output_dim: int | None, model_dim: int = 256, n_heads: int = 8,
        n_isab: int = 2, n_sab: int = 1, n_inducing_points: Union[int, List[int]] = 32,
        n_seeds: int = 1, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0,
        attn_norm: str = "none", ffn_norm: str = "none", output_norm: str = "none", use_checkpointing: bool = False,
        norm_position: str = "pre",
        mask_query_norms: bool = False,
    ):
        super().__init__()
        if isinstance(n_inducing_points, int):
            n_inducing_points = [n_inducing_points] * n_isab

        self.embedding = nn.Linear(input_dim, model_dim)

        self.isabs = nn.ModuleList([
            ISAB(model_dim, model_dim, n_heads, n_ip, ffn_hidden_dim=ffn_hidden_dim, dropout=dropout, attn_norm=attn_norm, ffn_norm=ffn_norm, use_checkpointing=use_checkpointing, norm_position=norm_position, mask_query_norms=mask_query_norms)
            for n_ip in n_inducing_points
        ])
        self.pma = PMA(model_dim, n_heads, n_seeds, ffn_hidden_dim=ffn_hidden_dim, dropout=dropout, attn_norm=attn_norm, ffn_norm=ffn_norm, use_checkpointing=use_checkpointing, norm_position=norm_position)
        self.sabs = nn.ModuleList([
            SAB(model_dim, n_heads, ffn_hidden_dim=ffn_hidden_dim, dropout=dropout, attn_norm=attn_norm, ffn_norm=ffn_norm, use_checkpointing=use_checkpointing, norm_position=norm_position)
            for _ in range(n_sab)
        ])
        self.output_norm = get_norm_layer(output_norm, model_dim)

        if output_dim is not None:
            self.output = nn.Linear(model_dim, output_dim)
        else:
            self.output = nn.Linear(model_dim, model_dim)

        self.output_dim = output_dim if output_dim is not None else model_dim

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                return_point_representations: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a set into a fixed-size representation.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, set_len, input_dim)`` representing the input set.
        attn_mask : torch.Tensor, optional
            Padding mask with ones for valid elements and zeros for padding. Accepts
            shape ``(batch, set_len)`` or any broadcastable variant.

        return_point_representations : bool, optional
            Also return the per-point representations after the ISAB stack (immediately
            before pooling), shape ``(batch, set_len, model_dim)`` with padded rows zeroed --
            the attach point for per-point heads. Defaults to ``False``.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Tensor of shape ``(batch, n_seeds, output_dim)`` containing set encodings; with
            ``return_point_representations`` a ``(encodings, point_representations)`` pair.
        """
        # Normalize the padding mask to bool. SDPA interprets FLOAT masks as ADDITIVE logit
        # biases (a 0/1 float mask therefore masks nothing); only bool masks exclude keys.
        # The standard pipeline (FlashANSRDataset.collate) already passes bool; coerce and warn
        # for any other caller so silent additive semantics cannot occur.
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            warnings.warn(
                "SetTransformer received a non-bool attn_mask; coercing to bool (mask != 0). "
                "Float masks are interpreted by scaled_dot_product_attention as additive logit "
                "biases and would silently not mask padding. If you intended an ADDITIVE mask "
                "(0 = keep, -inf = drop), this coercion INVERTS it - pass a bool mask instead.",
                stacklevel=2,
            )
            attn_mask = attn_mask != 0

        x = self.embedding(x)

        # Apply mask to input features before encoder to zero out padding.
        if attn_mask is not None:
            x = x * attn_mask.unsqueeze(-1)

        for isab in self.isabs:
            x = isab(x, attn_mask=attn_mask)

        point_representations = x

        x = self.pma(x, attn_mask=attn_mask)

        # The decoder operates on the dense output of PMA, so no mask is needed.
        for sab in self.sabs:
            x = sab(x)

        x = self.output_norm(x)
        out = self.output(x)
        if return_point_representations:
            return out, point_representations
        return out

from typing import Optional, Iterator, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flash_ansr.utils import load_config


USE_CHECKPOINTING = True


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The output is cast to the input's dtype, preventing issues with mixed precision.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    This is a memory-efficient and performant alternative to the standard FFN.
    Reference: "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        # Use a heuristic for the intermediate dim, as in Llama models
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = (hidden_dim + 7) & -8  # Multiple of 8

        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.w2(x)
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """Implements Rotary Positional Embeddings (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary embeddings to query and key tensors."""
    # cos, sin have shape (1, 1, seq_len, head_dim)
    # xq, xk have shape (batch, n_heads, seq_len, head_dim)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Decoder attention with RoPE and causal masking support."""
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, use_rope: bool = False):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rope = use_rope

        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        rope_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]

        q = self.w_q(query)
        k = self.w_k(key_value)
        v = self.w_v(key_value)

        q = q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if rope_emb is None:
                raise ValueError("rope_emb must be provided when use_rope is True")
            cos, sin = rope_emb
            q, k = apply_rotary_emb(q, k, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        return self.w_o(attn_output)


class TransformerDecoderBlock(nn.Module):
    """A single block of the Transformer Decoder."""
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.self_attn_norm = RMSNorm(dim)
        self.self_attention = Attention(dim=dim, n_heads=n_heads, dropout=dropout, use_rope=True)

        self.cross_attn_norm = RMSNorm(dim)
        self.encoder_mem_norm = RMSNorm(dim)
        self.cross_attention = Attention(dim=dim, n_heads=n_heads, dropout=dropout, use_rope=False)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim=dim, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def _forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
        rope_emb: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        x = x + self.self_attention(
            self.self_attn_norm(x), self.self_attn_norm(x),
            rope_emb=rope_emb, is_causal=True
        )
        x = x + self.cross_attention(
            self.cross_attn_norm(x), self.encoder_mem_norm(encoder_memory)
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
        rope_emb: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if self.training and self.use_checkpointing:
            # Re-define ckpt_fn to be compatible with checkpoint's argument handling
            def ckpt_fn(x: torch.Tensor, encoder_memory: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
                return self._forward(x, encoder_memory, (cos, sin))
            cos, sin = rope_emb
            return checkpoint(ckpt_fn, x, encoder_memory, cos, sin, use_reentrant=False)
        else:
            return self._forward(x, encoder_memory, rope_emb)


class TransformerDecoder(nn.Module):
    """State-of-the-art Transformer Decoder."""
    def __init__(
        self,
        vocab_size: int,
        input_dim: int,
        model_dim: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int = 4096,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)

        self.tok_embeddings = nn.Embedding(vocab_size, model_dim)
        self.rope = RotaryEmbedding(dim=model_dim // n_heads, max_seq_len=max_seq_len)

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                dim=model_dim, n_heads=n_heads, ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout, use_checkpointing=USE_CHECKPOINTING,
            ) for _ in range(n_layers)
        ])

        self.output_norm = RMSNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, vocab_size, bias=False)
        self.output_projection.weight = self.tok_embeddings.weight  # Weight tying

    def forward(self, tokens: torch.Tensor, encoder_memory: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        rope_emb = self.rope(h, seq_len=seq_len)

        for layer in self.layers:
            h = layer(h, encoder_memory, rope_emb)

        h = self.output_norm(h)
        logits = self.output_projection(h)
        return logits


class Tokenizer:
    '''
    Tokenizer class for converting tokens to indices and vice versa.

    Parameters
    ----------
    vocab : list[str]
        The vocabulary of the tokenizer.
    special_tokens : list[str], optional
        The special tokens to add to the vocabulary, by default None
    '''
    def __init__(self, vocab: list[str], special_tokens: list[str] | None = None) -> None:
        self.special_tokens = special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<mask>", "<constant>"]
        self.vocab = self.special_tokens + vocab

        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = dict(enumerate(self.vocab))

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "Tokenizer":
        '''
        Create a Tokenizer from a configuration dictionary or file.

        Parameters
        ----------
        config : dict[str, Any] | str
            The configuration dictionary or file path.

        Returns
        -------
        Tokenizer
            The Tokenizer instance.
        '''
        config_ = load_config(config)

        if "tokenizer" in config_.keys():
            config_ = config_["tokenizer"]

        return cls(vocab=config_["operators"] + config_["variables"], special_tokens=config_["special_tokens"])

    def encode(self, tokens: list[str], return_tensors: bool = False, add_bos: bool = False, add_eos: bool = False, oov: Literal['raise', 'unk'] = 'raise') -> list[int] | torch.Tensor:
        '''
        Encode a list of tokens to indices.

        Parameters
        ----------
        tokens : list[str]
            The list of tokens to encode.
        return_tensors : bool, optional
            Whether to return a tensor or a list, by default False
        add_bos : bool, optional
            Whether to add a beginning of sentence token, by default False
        add_eos : bool, optional
            Whether to add an end of sentence token, by default False
        oov : Literal['raise', 'unk'], optional
            How to handle out of vocabulary tokens, by default 'raise'

        Returns
        -------
        list[int] | torch.Tensor
            The list of indices or tensor.
        '''
        # TODO: Add support for input strings
        try:
            indices = [self.token2idx[token] for token in tokens]
        except KeyError as e:
            if oov == 'unk':
                indices = [self.token2idx.get(token, self.token2idx["<unk>"]) for token in tokens]
            else:
                print(f'Could not encode tokens {tokens}')
                raise e

        if add_bos:
            indices = [self.token2idx["<bos>"]] + indices

        if add_eos:
            indices = indices + [self.token2idx["<eos>"]]

        if return_tensors:
            return torch.tensor(indices, dtype=torch.long)

        return indices

    def decode(self, indices: list[int] | torch.Tensor, special_tokens: bool | str | list[str] = True) -> list[str]:
        '''
        Decode a list of indices to tokens.

        Parameters
        ----------
        indices : list[int] | torch.Tensor
            The list of indices to decode.
        special_tokens : bool | str | list[str], optional
            Whether to include special tokens, by default True

        Returns
        -------
        list[str]
            The list of tokens.
        '''
        if special_tokens is True:
            special_tokens = self.special_tokens
        elif special_tokens is False:
            special_tokens = []

        elif isinstance(special_tokens, str):
            special_tokens = [special_tokens]

        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        tokens = [self.idx2token[idx] for idx in indices]

        tokens = [token for token in tokens if token not in self.special_tokens or token in special_tokens]

        return tokens

    def __len__(self) -> int:
        '''
        Get the size of the vocabulary.

        Returns
        -------
        int
            The size of the vocabulary.
        '''
        return len(self.vocab)

    def __getitem__(self, key: str | int) -> int | str:
        '''
        Get the index of a token or the token of an index.

        Parameters
        ----------
        key : str | int
            The token or index to get.

        Returns
        -------
        int | str
            The index or token.
        '''
        if isinstance(key, str):
            return self.token2idx[key]

        if isinstance(key, int):
            return self.idx2token[key]

        raise TypeError(f"Unsupported key type {type(key)}")

    def __contains__(self, key: str | int) -> bool:
        '''
        Check if a token or index is in the vocabulary.

        Parameters
        ----------
        key : str | int
            The token or index to check.

        Returns
        -------
        bool
            Whether the token or index is in the vocabulary.
        '''
        if isinstance(key, str):
            return key in self.token2idx

        if isinstance(key, int):
            return key in self.idx2token

        raise TypeError(f"Unsupported key type {type(key)}")

    def __iter__(self) -> Iterator[str]:
        '''
        Iterate over the vocabulary.

        Returns
        -------
        Iterator[str]
            The iterator over the vocabulary.
        '''
        return iter(self.vocab)

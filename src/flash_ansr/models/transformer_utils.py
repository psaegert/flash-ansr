from typing import Iterator, Any, Literal

import torch
from torch import nn

from flash_ansr.utils import load_config


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
        self.special_tokens = special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<mask>", "<num>"]
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

        return cls(vocab=config_["vocab"], special_tokens=config_["special_tokens"])

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


class PositionalEncoding(nn.Module):
    '''
    Positional encoding module for transformer models.

    Notes
    -----
    See https://alexrichter.xyz/posts/implementing-sinusoidal-positional-embedding-transformer-pytorch/
    '''
    def __init__(self) -> None:
        super().__init__()

        # Store the encoding in attrbiutes to avoid re-computation
        self.seq_len: int | None = None
        self.input_size: int | None = None
        self.encoding: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Returns positional encoding for given input tensor X (batch_size, seq_len, size)

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, seq_len, size)

        Returns
        -------
        torch.Tensor
            Positional encoding of shape (seq_len, size)
        '''
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        if self.seq_len is None or self.input_size is None or self.encoding is None or \
                (x.shape[1], x.shape[2], x.device) != (self.seq_len, self.input_size, self.encoding.device):

            self.seq_len = x.shape[1]
            self.input_size = x.shape[2]
            self.encoding = torch.zeros((self.seq_len, self.input_size), device=x.device)

            t = 1 / 10000**(torch.arange(0, self.input_size, 2) / self.input_size)
            k = torch.arange(self.seq_len)
            v = torch.outer(k, t)

            self.encoding[:, 0::2] = v.sin()
            self.encoding[:, 1::2] = v.cos()

        return self.encoding

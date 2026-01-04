import re
import warnings
from typing import Iterator, Any, Literal

import torch

from flash_ansr.utils.config_io import load_config


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
        if add_bos or add_eos:
            warnings.warn(
                "The 'add_bos' and 'add_eos' parameters will be removed in a future release. "
                "Construct sequences with explicit prefix/suffix tokens before calling encode().",
                DeprecationWarning,
                stacklevel=2,
            )

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

    def extract_expression_from_beam(self, beam: list[int]) -> tuple[list[int], list[int], list[int]]:
        start_token = self.token2idx.get('<expression>')
        end_token = self.token2idx.get('</expression>')

        if start_token is None or end_token is None:
            expression = list(beam)
            before: list[int] = []
            after: list[int] = []

            bos_id = self.token2idx.get('<bos>')
            if bos_id is not None and expression and expression[0] == bos_id:
                before = expression[:1]
                expression = expression[1:]

            eos_id = self.token2idx.get('<eos>')
            if eos_id is not None and eos_id in expression:
                eos_index = expression.index(eos_id)
                after = expression[eos_index:]
                expression = expression[:eos_index]

            return expression, before, after

        try:
            expr_start = beam.index(start_token)
        except ValueError as exc:
            raise ValueError(f"Beam must contain <expression> token. Got {beam}.") from exc

        try:
            expr_end = beam.index(end_token, expr_start + 1)
        except ValueError as exc:
            raise ValueError(f"Beam must contain </expression> token after <expression>. Got {beam}.") from exc

        if expr_end <= expr_start + 1:
            return [], beam[:expr_start + 1], beam[expr_end:]

        before = beam[:expr_start + 1]
        after = beam[expr_end:]

        return beam[expr_start + 1:expr_end], before, after

    def constantify_expression(self, expression: list[int] | list[str], exact: bool = False) -> list[int] | list[str]:
        # Replace mult4, div3 etc by multiplication with <constant>

        # Find out if the expression is encoded or not
        if isinstance(expression, (list, tuple)) and all(isinstance(token, int) for token in expression):
            # If it's encoded, we need to convert it to the tokenizer's string representation
            constantified_expression = []
            for token in expression:
                if re.match(r"^mult\d+$", self.idx2token[token]) or re.match(r"^div\d+$", self.idx2token[token]):  # type: ignore
                    # Replace with '*', '<constant>
                    constantified_expression.append(self['*'])
                    if exact:
                        raise NotImplementedError("Exact constantification not implemented for encoded expressions.")
                    else:
                        constantified_expression.append(self['<constant>'])
                else:
                    constantified_expression.append(token)

        elif isinstance(expression, (list, tuple)) and all(isinstance(token, str) for token in expression):
            # If it's already a string representation, we can directly replace the patterns
            constantified_expression = []
            for token in expression:
                if re.match(r"^mult\d+$", token) or re.match(r"^div\d+$", token):  # type: ignore
                    # Replace with '*', '<constant>'
                    constantified_expression.append('*')
                    if exact:
                        # Find the factor or divisor from the token
                        match = re.match(r"^(mult|div)(\d+)$", token)  # type: ignore
                        if match:
                            factor = match.group(2)
                            constantified_expression.append(factor)
                        else:
                            raise ValueError(f"Could not parse token {token} for exact constantification.")
                    else:
                        constantified_expression.append('<constant>')
                else:
                    constantified_expression.append(token)
        else:
            raise ValueError("Expression must be a list of integers or strings.")
        return constantified_expression  # type: ignore

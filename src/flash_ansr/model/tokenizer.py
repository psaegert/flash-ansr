"""The :class:`Tokenizer` mapping expression tokens to model vocabulary indices and back."""
import re
import warnings
from typing import Iterator, Any, Literal

import torch

from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import BYTE_TOKENS, CONSTANTS_FORMAT, IEEE754_START_TOKEN


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
    #: Vocabulary entries that sit in the SPECIAL-token section but are legitimate expression
    #: CONTENT: the model is trained to emit them and simplipy accepts them as nullary operators.
    #: `decode(..., special_tokens='<constant>')` silently deleted `np.pi`/`np.e` from every
    #: candidate, leaving an arity-short token list that the validity gate then rejected without a
    #: counter -- so the pipeline was structurally incapable of ever returning a pi- or e-bearing
    #: expression, on any checkpoint or catalog.
    EXPRESSION_SPECIAL_TOKENS: tuple[str, ...] = ('<constant>', 'np.pi', 'np.e')

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

        # A vocabulary built for a different constants format shares no content token with
        # this one, so the mismatch is named here rather than surfacing much later as an
        # out-of-vocabulary error for a single token. A config that declares nothing is
        # checked by its alphabet instead, below.
        declared = config_.get("constants_format")
        if declared is not None and declared != CONSTANTS_FORMAT:
            raise ValueError(
                f"This tokenizer declares constants_format {declared!r}, but this build serves "
                f"{CONSTANTS_FORMAT!r}, which spells constants over a different alphabet. "
                f"Regenerate the configuration, or use a release that serves {declared!r}.")

        tokenizer = cls(vocab=config_["operators"] + config_["variables"],
                        special_tokens=config_["special_tokens"])

        # The same check for a vocabulary that declares no format, such as one stored
        # alongside a checkpoint. A vocabulary that serializes constants at all must carry
        # this codec's alphabet; without it every content token would come back
        # out-of-vocabulary one layer at a time.
        if declared is None and IEEE754_START_TOKEN in tokenizer and BYTE_TOKENS[0] not in tokenizer:
            raise ValueError(
                f"This tokenizer opens {IEEE754_START_TOKEN} spans but has no {BYTE_TOKENS[0]!r}, "
                f"so it does not serve {CONSTANTS_FORMAT!r}. Regenerate the configuration, or "
                f"use a release that serves the format this vocabulary was built for.")

        return tokenizer

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

    def decode_expression(self, indices: list[int] | torch.Tensor) -> list[str]:
        """Decode to an EXPRESSION token list, keeping every special that is expression content.

        The right decode for anything that will be handed to simplipy (validity, simplification,
        refinement): it keeps `<constant>` and, unlike ``decode(special_tokens='<constant>')``,
        also `np.pi` / `np.e`, which the tokenizer files list among the special tokens.
        """
        keep = [token for token in self.EXPRESSION_SPECIAL_TOKENS if token in self.token2idx]
        return self.decode(indices, special_tokens=keep)

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
        '''
        Split a decoded beam into the expression body and the surrounding tokens.

        If the vocabulary defines ``<expression>`` / ``</expression>`` markers, the tokens between
        them are returned as the body. Otherwise the body is everything except a leading ``<bos>``
        and a trailing ``<eos>`` (and whatever follows it).

        Parameters
        ----------
        beam : list[int]
            The sequence of token indices produced by the decoder.

        Returns
        -------
        tuple[list[int], list[int], list[int]]
            A ``(expression, before, after)`` triple of token-index lists: the expression body,
            the tokens preceding it and the tokens following it.

        Raises
        ------
        ValueError
            If the vocabulary defines ``<expression>`` markers but ``beam`` does not contain a
            matching opening/closing pair.
        '''
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
        '''
        Rewrite integer-factor operators (``mult4``, ``div3`` ...) as explicit constant multiplications.

        Each ``mult<n>`` / ``div<n>`` token is replaced by ``*`` followed by a constant: a
        ``<constant>`` placeholder by default, or the literal integer factor when ``exact`` is True.
        The input may be either a list of token indices or a list of token strings; the output uses
        the same representation.

        Parameters
        ----------
        expression : list[int] | list[str]
            The expression as token indices or token strings.
        exact : bool, optional
            If True, emit the literal integer factor instead of a ``<constant>`` placeholder.
            Only supported for string expressions. Defaults to False.

        Returns
        -------
        list[int] | list[str]
            The constantified expression in the same representation as the input.

        Raises
        ------
        NotImplementedError
            If ``exact`` is requested for an encoded (integer) expression.
        ValueError
            If ``expression`` is neither a list of integers nor a list of strings, or an ``exact``
            factor token cannot be parsed.
        '''
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

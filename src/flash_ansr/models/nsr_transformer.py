import os
import warnings
from typing import Any, Literal

import torch
from torch import nn
from tqdm import tqdm

from flash_ansr.utils import load_config, save_config, substitute_root_path
from flash_ansr.expressions.expression_space import ExpressionSpace
from flash_ansr.models.transformer_utils import PositionalEncoding
from flash_ansr.models.encoders.pre_encoder import PreEncoder
from flash_ansr.models.factory import ModelFactory
from flash_ansr.preprocess import FlashASNRPreprocessor


class FlashANSRTransformer(nn.Module):
    '''
    Flash Amortized Neural Symbolic Regression Transformer model.

    Parameters
    ----------
    expression_space : ExpressionSpace
        The expression space to use for the model.
    encoder_max_n_variables : int
        The maximum number of variables that the encoder can handle.
    size : int, optional
        The internal size of the model's hidden layers and attention, by default 512.
    norm_first : bool, optional
        Whether to use layer normalization before attention, by default False.
    pre_encoder_input_type : str, optional
        The type of input to the pre-encoder, by default "ieee-754".
    pre_encoder_support_nan : bool, optional
        Whether the pre-encoder should support NaN values, by default False.
    pre_encoder_exponent_scale : float, optional
        The scale factor for the exponent in the pre-encoder, by default None.
    encoder : str or nn.Module, optional
        The encoder model to use, by default "SetTransformer".
    encoder_kwargs : dict[str, Any], optional
        Keyword arguments to pass to the encoder model constructor, by default None.
    decoder_n_heads : int, optional
        The number of attention heads in the decoder, by default 8.
    decoder_ff_size : int, optional
        The size of the feed-forward layer in the decoder, by default 2048.
    decoder_dropout : float, optional
        The dropout rate in the decoder, by default 0.1.
    decoder_n_layers : int, optional
        The number of layers in the decoder, by default 6.
    learnable_positional_embeddings : bool, optional
        Whether to use learnable positional embeddings, by default False.
    max_input_length : int, optional
        The maximum input length for positional embeddings, by default 32.
    '''
    memory: torch.Tensor

    def __init__(
            self,
            expression_space: ExpressionSpace,
            encoder_max_n_variables: int,
            size: int = 512,
            norm_first: bool = False,
            activation: str = "ReLU",
            pre_encoder_input_type: str = "ieee-754",
            pre_encoder_support_nan: bool = False,
            pre_encoder_exponent_scale: float | None = None,
            encoder: str | nn.Module = "SetTransformer",
            encoder_kwargs: dict[str, Any] | None = None,
            decoder_n_heads: int = 8,
            decoder_ff_size: int = 2048,
            decoder_dropout: float = 0.1,
            decoder_n_layers: int = 6,
            learnable_positional_embeddings: bool = False,
            max_input_length: int = 32,
            support_numeric_tokens: bool = False) -> None:
        super().__init__()

        self.expression_space = expression_space
        self.encoder_max_n_variables = encoder_max_n_variables

        self.pre_encoder = PreEncoder(
            input_size=encoder_max_n_variables,
            mode=pre_encoder_input_type,
            support_nan=pre_encoder_support_nan,
            exponent_scale=pre_encoder_exponent_scale)

        if isinstance(encoder, str):
            self.encoder = ModelFactory.get_model(
                encoder,
                input_embedding_size=self.pre_encoder.encoding_size,
                input_dimension_size=encoder_max_n_variables,
                output_embedding_size=size,
                **encoder_kwargs or {})
        else:
            self.encoder = encoder

        self.embedding = nn.Embedding(len(expression_space.tokenizer), size)
        if learnable_positional_embeddings:
            self.pos_encoding = nn.Embedding(max_input_length, size)
            self.pos_tokens = nn.Parameter(torch.arange(max_input_length, device=self.device), requires_grad=False)
        else:
            self.pos_encoding = PositionalEncoding()

        self.support_numeric_tokens = support_numeric_tokens
        if support_numeric_tokens:

            self.pre_encoder_numeric_tokens = PreEncoder(
                input_size=1,
                mode=pre_encoder_input_type,
                support_nan=True,
                exponent_scale=pre_encoder_exponent_scale)

            self.numeric_embedding = nn.Linear(self.pre_encoder_numeric_tokens.output_size, size)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=size,
                nhead=decoder_n_heads,
                dim_feedforward=decoder_ff_size,
                dropout=decoder_dropout,
                batch_first=True,
                activation=ModelFactory.get_model(activation),
                norm_first=norm_first
            ),
            num_layers=decoder_n_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(size, size),
            nn.GELU(),
            nn.Dropout(p=decoder_dropout),
            nn.Linear(size, len(expression_space.tokenizer)))

        self.num_out = nn.Sequential(
            nn.Linear(size, size),
            nn.GELU(),
            nn.Dropout(p=decoder_dropout),
            nn.Linear(size, 1))

        self.preprocessor = FlashASNRPreprocessor(expression_space)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "FlashANSRTransformer":
        '''
        Create an FlashANSRTransformer from a configuration dictionary or file.

        Parameters
        ----------
        config : dict[str, Any] or str
            The configuration dictionary or file.

        Returns
        -------
        FlashANSRTransformer
            The FlashANSRTransformer instance.
        '''
        config_ = load_config(config)

        if "nsr" in config_.keys():
            config_ = config_["nsr"]

        # If the config is a string, convert relative paths within the config to absolute paths
        if isinstance(config, str) and isinstance(config_["expression_space"], str):
            if config_["expression_space"].startswith('.'):
                config_["expression_space"] = os.path.join(os.path.dirname(config), config_["expression_space"])

        expression_space = ExpressionSpace.from_config(config_["expression_space"])

        return cls(
            expression_space=expression_space,
            encoder_max_n_variables=config_["encoder_max_n_variables"],
            size=config_["size"],
            norm_first=config_.get("norm_first", False),
            pre_encoder_input_type=config_["pre_encoder_input_type"],
            pre_encoder_support_nan=config_["pre_encoder_support_nan"],
            pre_encoder_exponent_scale=config_.get("pre_encoder_exponent_scale"),
            encoder=config_["encoder"],
            encoder_kwargs=config_.get("encoder_kwargs"),
            decoder_n_heads=config_["decoder_n_heads"],
            decoder_ff_size=config_["decoder_ff_size"],
            decoder_dropout=config_["decoder_dropout"],
            decoder_n_layers=config_["decoder_n_layers"],
            learnable_positional_embeddings=config_["learnable_positional_embeddings"],
            max_input_length=config_["max_input_length"],
            support_numeric_tokens=config_.get("support_numeric_tokens", False))

    def forward(self, input_tokens: torch.Tensor, data: torch.Tensor, input_num: torch.Tensor | None = None, numeric_head: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        '''
        Forward pass through the model.

        Parameters
        ----------
        input_tokens : torch.Tensor
            The input tokens.
        data : torch.Tensor
            The data tensor.
        input_num : torch.Tensor, optional
            The input numeric tensor, by default None.
        numeric_head : bool, optional
            Whether to include the numeric head, by default False.

        Returns
        -------
        logits : torch.Tensor
            The logits (next token probabilities from the next token head).
        num_out : torch.Tensor
            The numeric output from the numeric head.
        '''
        embeddings = self.embedding(input_tokens)

        if isinstance(self.pos_encoding, nn.Embedding):
            embeddings = embeddings + self.pos_encoding(self.pos_tokens[:input_tokens.shape[1]])
        elif isinstance(self.pos_encoding, PositionalEncoding):
            embeddings = embeddings + self.pos_encoding(embeddings)

        if input_num is not None:
            if not self.support_numeric_tokens:
                raise ValueError("Model does not support numeric tokens.")

            input_num_pre_encodings = self.pre_encoder_numeric_tokens(input_num)
            input_num_pre_encodings[torch.isnan(input_num_pre_encodings)] = 0
            embeddings = embeddings + self.numeric_embedding(input_num_pre_encodings)

        data_pre_encodings: torch.Tensor = self.pre_encoder(data)
        B, M, D, E = data_pre_encodings.size()
        self.memory = self.encoder(data_pre_encodings.view(B, M, D * E))

        if self.memory.ndim > 3:
            self.memory = self.memory.view(B, -1, self.memory.size(-1))

        attn_mask = nn.Transformer.generate_square_subsequent_mask(input_tokens.shape[1], device=input_tokens.device)
        padding_mask = (input_tokens == self.expression_space.tokenizer["<pad>"]).float().masked_fill(input_tokens == self.expression_space.tokenizer["<pad>"], 1e-9)

        output = self.decoder.forward(
            tgt=embeddings,
            memory=self.memory,
            tgt_mask=attn_mask,
            tgt_key_padding_mask=padding_mask)

        logits = self.fc_out(output)

        # FIXME: Does this decorage TRF from predicting <num> tokens?
        if numeric_head:
            output_full = self.decoder.forward(
                tgt=embeddings,
                memory=self.memory,
                tgt_key_padding_mask=padding_mask)
            num_out = self.num_out(output_full)
        else:
            num_out = None

        return logits, num_out

    def beam_search(self, data: torch.Tensor, beam_width: int = 4, max_len: int = 100, mini_batch_size: int = 128, equivalence_pruning: bool = True, complexity: int | float | None = None, verbose: bool = False) -> tuple[list[list[int]], list[float], list[bool]]:
        '''
        Beam search algorithm to generate sequences.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor.
        beam_size : int, optional
            The number of beams, by default 4.
        max_len : int, optional
            The maximum length of the sequences, by default 100.
        mini_batch_size : int, optional
            The mini-batch size, by default 128.
        equivalence_pruning : bool, optional
            Whether to prune equivalent sequences, by default True.
        complexity : int or float, optional
            The desired complexity (length in tokens) of the generated expression. If None, the complexity is not enforced, by default None.
        verbose : bool, optional
            Whether to print debug information, by default False.

        Returns
        -------
        filtered_sequences : list[list[int]]
            The list of generated sequences.
        filtered_scores : list[float]
            The list of sequence probabilities.
        filtered_is_valid : list[bool]
            A list of booleans indicating whether the sequence is a valid expression.
        '''
        if complexity is None:
            initial_beam, input_num = [self.expression_space.tokenizer['<bos>']], None
        else:
            initial_beam, input_num = self.preprocessor.format_complexity([self.expression_space.tokenizer['<bos>']], complexity=complexity)

        # Step 1: Initialize the beam with the initial input sequence
        beams = [(initial_beam, 0.0)]  # each beam is a tuple: (sequence, score)
        completed_sequences = []  # store completed sequences here
        n_pruned = 0

        pbar = tqdm(total=max_len, disable=not verbose, desc=f"Generating beams (max length: {max_len})")

        for _ in range(max_len):
            all_sequences = []
            all_scores = []

            # Prepare batched sequences and their scores
            for seq, score in beams:
                if seq[-1] == self.expression_space.tokenizer['<eos>']:
                    completed_sequences.append((seq, score))
                    continue  # Don't expand this sequence further
                all_sequences.append(seq)
                all_scores.append(score)

            if len(all_sequences) == 0:
                break  # All sequences completed

            # Step 2: Pad the sequences to ensure they all have the same length
            max_seq_len = max(len(seq) for seq in all_sequences)
            input_ids_padded = [seq + [self.expression_space.tokenizer['<pad>']] * (max_seq_len - len(seq)) for seq in all_sequences]
            input_num_padded = input_num + [torch.nan] * (max_seq_len - len(input_num)) if input_num is not None else None

            # Convert sequences and scores to tensors
            input_ids_tensor = torch.tensor(input_ids_padded, device=data.device)
            input_num_tensor = torch.tensor(input_num_padded, device=data.device).unsqueeze(-1) if input_num is not None else None

            # Step 3: Run forward pass in mini-batches
            all_next_token_logits = []  # Collect logits from all mini-batches
            for start_idx in range(0, len(all_sequences), mini_batch_size):
                # Extract the mini-batch
                end_idx = min(start_idx + mini_batch_size, len(all_sequences))
                mini_batch = input_ids_tensor[start_idx:end_idx]
                mini_batch_data = data.unsqueeze(0).repeat(end_idx - start_idx, 1, 1)

                # Forward pass for the mini-batch
                logits, _ = self.forward(mini_batch, mini_batch_data, input_num=input_num_tensor, numeric_head=False)

                # Collect the logits for the next token
                all_next_token_logits.append(logits[:, -1, :])  # Shape: (mini_batch_size, vocab_size)

            # Concatenate logits from all mini-batches into one tensor
            next_token_logits = torch.cat(all_next_token_logits, dim=0)  # Shape: (total_beams, vocab_size)
            next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)  # Shape: (total_beams, vocab_size)

            # Step 4: Collect all possible next beams
            all_candidates = []

            for i, (seq, score) in enumerate(zip(all_sequences, all_scores)):
                # For each token, create a new beam
                for token in range(next_token_log_probs.shape[1]):
                    log_prob = next_token_log_probs[i, token].item()

                    new_seq = seq + [token]
                    new_score = score + log_prob
                    all_candidates.append((new_seq, new_score))

            # Step 5: Sort all candidates by score (highest scores first)
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)

            # Step 6: Simplify completed sequences and filter out duplicates
            if equivalence_pruning:
                all_candidates_unique: list[tuple] = []
                for candidate in all_candidates:
                    if candidate[0][-1] == self.expression_space.tokenizer['<eos>']:
                        candidate_expression, before, after = self.expression_space.extract_expression_from_beam(candidate[0])
                        candidate_expression_decoded = self.expression_space.tokenizer.decode(candidate_expression, special_tokens='<num>')

                        if self.expression_space.is_valid(candidate_expression_decoded) and len(candidate_expression_decoded) > 1:
                            candidate_simplified = self.expression_space.tokenizer.encode(self.expression_space.simplify(candidate_expression_decoded))

                            # Add back everything before <bos> and after <eos>
                            candidate_simplified = before + candidate_simplified + after

                            if candidate_simplified not in [seq for seq, _ in all_candidates_unique] and candidate_simplified not in [seq for seq, _ in completed_sequences]:
                                all_candidates_unique.append((candidate_simplified, candidate[1]))
                            else:
                                n_pruned += 1
                    else:
                        all_candidates_unique.append(candidate)

                # Step 7: Select the top 'beam_size' candidates to be the new beams
                beams = all_candidates_unique[:beam_width]

            else:
                beams = all_candidates[:beam_width]

            # If no beams are left to expand, break early
            if not beams:
                break

            pbar.set_postfix({'completed': len(completed_sequences), 'pruned': n_pruned})
            pbar.update(1)

        # Step 7: Combine completed sequences with current beams (in case they are incomplete)
        completed_sequences.extend(beams)

        # Step 8: Sort all sequences by score (highest scores first)
        completed_sequences = sorted(completed_sequences, key=lambda x: x[1], reverse=True)

        # Step 9: Return the top 'beam_size' sequences
        return [seq for seq, _ in completed_sequences[:beam_width]], [score for _, score in completed_sequences[:beam_width]], [True] * len(completed_sequences[:beam_width])

    def sample_top_kp(
            self, data: torch.Tensor, choices: int = 10, top_k: int = 0, top_p: float = 1, max_len: int = 100,
            mini_batch_size: int = 128, complexity: int | float | None = None,
            temperature: float = 1.0, valid_only: bool = True, simplify: bool = True, unique: bool = True, verbose: bool = False) -> tuple[list[int], list[float], list[bool]]:
        '''
        Top-k sampling algorithm to generate sequences.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor.
        choices : int, optional
            The number of sequences to generate, by default 10.
        top_k : int, optional
            The number of highest probability tokens to sample from, by default 0 (consider entire vocabulary).
        top_p : float, optional
            The probability mass to sample from (nuclues sampling), by default 1.0 (consider entire vocabulary).
        max_len : int, optional
            The maximum length of the sequences, by default 100.
        mini_batch_size : int, optional
            The mini-batch size for batch processing, by default 128.
        complexity : int or float, optional
            The desired complexity (length in tokens) of the generated expression.
            If None, the complexity is not enforced, by default None.
        temperature : float, optional
            The temperature parameter for softmax. Higher values increase randomness, by default 1.0.
        valid_only : bool, optional
            Whether to only return valid expressions, by default True.
        simplify : bool, optional
            Whether to simplify the generated expressions, by default True.
        unique : bool, optional
            Whether to return only unique expressions, by default True.
        verbose : bool, optional
            Whether to print debug information, by default False.

        Returns
        -------
        filtered_sequences : list[list[int]]
            The list of generated sequences.
        filtered_scores : list[float]
            The list of sequence probabilities.
        filtered_is_valid : list[bool]
            A list of booleans indicating whether the sequence is a valid expression.

        Notes
        -----
        https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
        '''
        # Prepare initial token based on complexity
        if complexity is None:
            initial_token, input_num = [self.expression_space.tokenizer['<bos>']], None
        else:
            initial_token, input_num = self.preprocessor.format_complexity([self.expression_space.tokenizer['<bos>']], complexity=complexity)

        # Initialize sequences for generation
        sequences = [initial_token.copy() for _ in range(choices)]
        scores = [0.0] * choices

        completed_sequences: list = []
        completed_scores = []

        pbar = tqdm(total=max_len, disable=not verbose, desc=f"Generating choices (-/{choices:,})")

        # Continue until all sequences are completed or max length is reached
        for current_length in range(max_len):
            pbar.set_description(f"Generating choices ({len(completed_sequences):,}/{choices:,})")
            new_sequences = []
            new_scores = []

            # Prepare batches for efficient processing
            for start_idx in range(0, len(sequences), mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, len(sequences))
                batch_sequences = sequences[start_idx:end_idx]
                batch_scores = scores[start_idx:end_idx]

                # Pad the input numbers
                current_len = len(batch_sequences[0])
                input_num_padded = input_num + [torch.nan] * (current_len - len(input_num)) if input_num is not None else None

                # Convert to tensors
                input_ids_tensor = torch.tensor(batch_sequences, device=data.device)
                input_num_tensor = torch.tensor(input_num_padded, device=data.device).unsqueeze(-1) if input_num is not None else None

                # Forward pass
                batch_data = data.unsqueeze(0).repeat(end_idx - start_idx, 1, 1)
                logits, _ = self.forward(input_ids_tensor, batch_data, input_num=input_num_tensor, numeric_head=False)

                # Get logits for the next token
                logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

                original_scores = torch.log_softmax(logits, dim=-1)

                # Filter top k
                top_k = min(top_k, logits.size(-1))  # Safety check
                if top_k > 0:
                    # Remove all tokens with a probability less than the last token of the top-k
                    ignore_mask = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
                    logits[ignore_mask] = - float('Inf')

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                if top_p < 1.0:
                    sorted_logits, sorted_logit_indices = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_sorted_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_ignore_mask = cumulative_sorted_probs > top_p

                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_ignore_mask[..., 1:] = sorted_ignore_mask[..., :-1].clone()
                    sorted_ignore_mask[..., 0] = False

                    sorted_logits[sorted_ignore_mask] = - float('Inf')

                    # Then reverse the sorting process by mapping back sorted_logits to their original position
                    logits = torch.gather(sorted_logits, 1, sorted_logit_indices.argsort(-1))

                sampled_tokens = torch.multinomial(torch.softmax(logits, dim=-1), 1)

                # Sample from top-k tokens
                for i, (seq, score, sampled_token) in enumerate(zip(batch_sequences, batch_scores, sampled_tokens)):
                    # Sample token from top-k distribution and append to sequence
                    new_seq = seq + [sampled_token.item()]
                    new_score = score + original_scores[i][sampled_token.item()].item()  # type: ignore

                    # If token is <eos>, add to completed sequences
                    if sampled_token == self.expression_space.tokenizer['<eos>']:
                        completed_sequences.append(new_seq)
                        completed_scores.append(new_score)
                        continue

                    # Append incomplete sequence for next iteration
                    new_sequences.append(new_seq)
                    new_scores.append(new_score)

            # Filter out completed sequences for next iteration
            sequences = new_sequences
            scores = new_scores

            # If all sequences are completed, break early
            if len(sequences) == 0:
                break

            pbar.update(1)

        # Add any remaining incomplete sequences (truncate to max_len)
        for seq, score in zip(sequences, scores):
            completed_sequences.append(seq)
            completed_scores.append(score)

        pbar.close()

        # Simplify and filter for uniqueness
        filtered_sequences = []
        filtered_scores = []
        filtered_is_valid = []
        seen_expressions = set()

        for seq, score in zip(completed_sequences, completed_scores):
            encoded_expression, before, after = self.expression_space.extract_expression_from_beam(seq)

            # Decode the sequence
            expression = self.expression_space.tokenizer.decode(encoded_expression, special_tokens='<num>')

            # Only process valid expressions
            if self.expression_space.is_valid(expression) and len(expression) > 1:
                if simplify:
                    # Simplify the expression
                    expression = self.expression_space.simplify(expression)

                    # Skip if we've already seen this simplified expression
                if unique and tuple(expression) in seen_expressions:
                    continue

                # Encode the simplified expression
                filtered_sequence = self.expression_space.tokenizer.encode(expression, add_bos=True, add_eos=True)
                filtered_sequences.append(filtered_sequence)
                filtered_scores.append(score)
                filtered_is_valid.append(True)

                if unique:
                    seen_expressions.add(tuple(expression))

            elif not valid_only:
                filtered_sequences.append(seq)
                filtered_scores.append(score)
                filtered_is_valid.append(False)

        sorted_indices_by_score = sorted(range(len(filtered_scores)), key=lambda i: filtered_scores[i], reverse=True)
        filtered_sequences = [filtered_sequences[i] for i in sorted_indices_by_score]
        filtered_scores = [filtered_scores[i] for i in sorted_indices_by_score]
        filtered_is_valid = [filtered_is_valid[i] for i in sorted_indices_by_score]

        return filtered_sequences, filtered_scores, filtered_is_valid

    def save(self, directory: str, config: dict[str, Any] | str | None = None, reference: str = 'relative', recursive: bool = True, errors: Literal['raise', 'warn', 'ignore'] = 'warn') -> None:
        '''
        Save the model to a directory.

        Parameters
        ----------
        directory : str
            The directory to save the model to.
        config : dict[str, Any] or str, optional
            The configuration dictionary or file, by default None.
        reference : str, optional
            Determines the reference base path. One of
            - 'relative': relative to the specified directory
            - 'project': relative to the project root
            - 'absolute': absolute paths
        recursive : bool, optional
            Save any referenced configs too
        errors : {'raise', 'warn', 'ignore'}, optional
            How to handle errors, by default 'warn'.

        Raises
        ------
        ValueError
            If no config is specified and errors is 'raise'.
        '''
        directory = substitute_root_path(directory)

        os.makedirs(directory, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(directory, "state_dict.pt"))

        # Copy the config to the directory for best portability
        if config is None:
            if errors == 'raise':
                raise ValueError("No config specified, saving the model without a config file. Loading the model will require manual configuration.")
            if errors == 'warn':
                warnings.warn("No config specified, saving the model without a config file. Loading the model will require manual configuration.")
        else:
            save_config(
                load_config(config, resolve_paths=True),
                directory=directory,
                filename='nsr.yaml',
                reference=reference,
                recursive=recursive,
                resolve_paths=True)

    @classmethod
    def load(cls, directory: str) -> tuple[dict[str, Any], "FlashANSRTransformer"]:
        '''
        Load a model from a directory.

        Parameters
        ----------
        directory : str
            The directory to load the model from.

        Returns
        -------
        dict[str, Any]
            The configuration dictionary.
        FlashANSRTransformer
            The FlashANSRTransformer instance
        '''
        directory = substitute_root_path(directory)

        config_path = os.path.join(directory, 'nsr.yaml')

        model = cls.from_config(config_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True))

        return load_config(config_path), model

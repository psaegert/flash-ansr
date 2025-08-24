import os
import warnings
from typing import Any, Literal
import re

import torch
from torch import nn
from tqdm import tqdm

from simplipy import SimpliPyEngine

from flash_ansr.utils import load_config, save_config, substitute_root_path
from flash_ansr.model.transformer import Tokenizer
from flash_ansr.model.pre_encoder import IEEE75432PreEncoder
from flash_ansr.preprocess import FlashASNRPreprocessor
from flash_ansr.model.set_transformer import SetTransformer
from flash_ansr.model.transformer import TransformerDecoder

# Assuming these are defined in your refactored modules
USE_CHECKPOINTING = True


class FlashANSRModel(nn.Module):
    def __init__(
        self,
        simplipy_engine: SimpliPyEngine,
        tokenizer: Tokenizer,

        encoder_max_n_variables: int,
        encoder_dim: int = 512,
        encoder_n_heads: int = 8,
        encoder_n_isab: int = 2,
        encoder_n_sab: int = 1,
        encoder_n_inducing_points: int = 32,
        encoder_n_seeds: int = 1,
        encoder_ffn_hidden_dim: int = 2048,
        encoder_dropout: float = 0.1,

        decoder_input_dim: int = 512,
        decoder_model_dim: int = 512,
        decoder_n_layers: int = 6,
        decoder_n_heads: int = 8,
        decoder_max_seq_len: int = 4096,
        decoder_ffn_hidden_dim: int = None,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.simplipy_engine = simplipy_engine
        self.tokenizer = tokenizer
        self.encoder_max_n_variables = encoder_max_n_variables

        self.pre_encoder = IEEE75432PreEncoder(input_size=encoder_max_n_variables)

        self.pre_encoder_numeric_tokens = IEEE75432PreEncoder(input_size=1)
        self.numeric_embedding = nn.Linear(self.pre_encoder_numeric_tokens.output_size, decoder_input_dim)

        self.encoder = SetTransformer(
            input_dim=self.pre_encoder.output_size,
            output_dim=None,
            model_dim=encoder_dim,
            n_heads=encoder_n_heads,
            n_isab=encoder_n_isab,
            n_sab=encoder_n_sab,
            n_inducing_points=encoder_n_inducing_points,
            n_seeds=encoder_n_seeds,
            ffn_hidden_dim=encoder_ffn_hidden_dim,
            dropout=encoder_dropout
        )

        self.decoder = TransformerDecoder(
            vocab_size=len(tokenizer),
            input_dim=None,
            model_dim=decoder_model_dim,
            n_layers=decoder_n_layers,
            n_heads=decoder_n_heads,
            max_seq_len=decoder_max_seq_len,
            ffn_hidden_dim=decoder_ffn_hidden_dim,
            dropout=decoder_dropout
        )

        self.preprocessor = FlashASNRPreprocessor(simplipy_engine=simplipy_engine, tokenizer=tokenizer)

        self.memory: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "FlashANSRModel":
        config_ = load_config(config)

        if "nsr" in config_.keys():
            config_ = config_["nsr"]

        if isinstance(config, str) and isinstance(config_["simplipy_engine"], str):
            if config_["simplipy_engine"].startswith('.'):
                config_["simplipy_engine"] = os.path.join(os.path.dirname(config), config_["simplipy_engine"])

        simplipy_engine = SimpliPyEngine.from_config(config_["simplipy_engine"])
        tokenizer = Tokenizer.from_config(config_["tokenizer"])

        # Update the mapping of config keys to match the new __init__ signature
        return cls(
            simplipy_engine=simplipy_engine,
            tokenizer=tokenizer,
            encoder_max_n_variables=config_["encoder_max_n_variables"],
            encoder_dim=config_["encoder_dim"],
            encoder_n_heads=config_["encoder_n_heads"],
            encoder_n_isab=config_["encoder_n_isab"],
            encoder_n_sab=config_["encoder_n_sab"],
            encoder_n_inducing_points=config_["encoder_n_inducing_points"],
            encoder_n_seeds=config_["encoder_n_seeds"],
            encoder_ffn_hidden_dim=config_["encoder_ffn_hidden_dim"],
            encoder_dropout=config_["encoder_dropout"],
            decoder_input_dim=config_["decoder_input_dim"],
            decoder_model_dim=config_["decoder_model_dim"],
            decoder_n_layers=config_["decoder_n_layers"],
            decoder_n_heads=config_["decoder_n_heads"],
            decoder_max_seq_len=config_["decoder_max_seq_len"],
            decoder_ffn_hidden_dim=config_["decoder_ffn_hidden_dim"],
            decoder_dropout=config_["decoder_dropout"],
        )

    def _create_memory(self, data: torch.Tensor) -> torch.Tensor:
        # Pre-process input data
        data_pre_encodings: torch.Tensor = self.pre_encoder(data)
        B, M, D, E = data_pre_encodings.size()

        # Encoder forward pass
        memory = self.encoder(data_pre_encodings.view(B, M, D * E))

        if memory.ndim > 3:
            memory = memory.view(B, -1, memory.size(-1))

        return memory

    def forward(self, input_tokens: torch.Tensor, data: torch.Tensor, input_num: torch.Tensor | None = None, memory: torch.Tensor | None = None) -> torch.Tensor:
        if memory is not None:
            self.memory = memory
        else:
            self.memory = self._create_memory(data)

        # Add numeric token logic back
        # The new TransformerDecoder handles embedding and positional encoding internally.
        # We need to pass the numeric embeddings to it.
        if input_num is not None:

            input_num_pre_encodings = self.pre_encoder_numeric_tokens(input_num)
            input_num_pre_encodings[torch.isnan(input_num_pre_encodings)] = 0
            numeric_embeddings = self.numeric_embedding(input_num_pre_encodings)
        else:
            numeric_embeddings = None

        # Pass both symbolic and numeric inputs to the decoder
        # This requires modifying the decoder's forward method, but for this refactor,
        # we'll handle the combination here before calling the decoder.
        # The new TransformerDecoder's forward method needs to be modified to accept this.
        # For this refactoring, we'll assume a new argument `numeric_embeddings` is added.
        # However, the provided `TransformerDecoder` code doesn't support this.
        # A simple solution is to add the numeric embeddings to the symbolic embeddings
        # before passing them to the decoder.

        logits = self.decoder(tokens=input_tokens, encoder_memory=self.memory, extra_parallel_embeddings=numeric_embeddings)

        # Removed numeric head as it is not present in the new Decoder structure
        return logits

    def beam_search(self, data: torch.Tensor, beam_width: int = 4, max_len: int = 100, mini_batch_size: int = 128, equivalence_pruning: bool = True, complexity: int | float | None = None, verbose: bool = False) -> tuple[list[list[int]], list[float], list[bool]]:
        if complexity is None:
            initial_beam, input_num = [self.tokenizer['<bos>']], None
        else:
            initial_beam, input_num = self.preprocessor.format_complexity([self.tokenizer['<bos>']], complexity=complexity)

        # Step 1: Initialize the beam with the initial input sequence
        beams = [(initial_beam, 0.0)]  # each beam is a tuple: (sequence, score)
        completed_sequences = []  # store completed sequences here
        n_pruned = 0

        memory = self._create_memory(data)

        pbar = tqdm(total=max_len, disable=not verbose, desc=f"Generating beams (max length: {max_len})")

        for _ in range(max_len):
            all_sequences = []
            all_scores = []
            # Prepare batched sequences and their scores
            for seq, score in beams:
                if seq[-1] == self.tokenizer['<eos>']:
                    completed_sequences.append((seq, score))
                    continue  # Don't expand this sequence further
                all_sequences.append(seq)
                all_scores.append(score)

            if len(all_sequences) == 0:
                break  # All sequences completed

            # Step 2: Pad the sequences to ensure they all have the same length
            max_seq_len = max(len(seq) for seq in all_sequences)
            input_ids_padded = [seq + [self.tokenizer['<pad>']] * (max_seq_len - len(seq)) for seq in all_sequences]
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
                logits = self.forward(mini_batch, mini_batch_data, input_num=input_num_tensor, memory=memory)

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
                    if candidate[0][-1] == self.tokenizer['<eos>']:
                        candidate_expression, before, after = self.extract_expression_from_beam(candidate[0])
                        candidate_expression_decoded = self.tokenizer.decode(candidate_expression, special_tokens='<constant>')

                        if self.simplipy_engine.is_valid(candidate_expression_decoded) and len(candidate_expression_decoded) > 1:
                            candidate_simplified = self.tokenizer.encode(self.simplipy_engine.simplify(candidate_expression_decoded, max_pattern_length=4))

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

            for beam, score in beams:
                if beam[0] != self.tokenizer['<bos>']:
                    raise ValueError(f"Beam must start with <bos> token. Got {beam}.")

            # If no beams are left to expand, break early
            if not beams:
                break

            pbar.set_postfix({'completed': len(completed_sequences), 'pruned': n_pruned})
            pbar.update(1)

        # Step 7: Combine completed sequences with current beams (in case they are incomplete)
        completed_sequences.extend(beams)

        # Step 7.4: Constantify the expressions in completed sequences
        for i, (seq, score) in enumerate(completed_sequences):
            constantified_seq = self.constantify_expression(seq)
            completed_sequences[i] = (constantified_seq, score)

        # Step 8: Sort all sequences by score (highest scores first)
        completed_sequences = sorted(completed_sequences, key=lambda x: x[1], reverse=True)

        # Step 9: Return the top 'beam_size' sequences
        return [seq for seq, _ in completed_sequences[:beam_width]], [score for _, score in completed_sequences[:beam_width]], [True] * len(completed_sequences[:beam_width])

    def sample_top_kp(
            self, data: torch.Tensor, choices: int = 10, top_k: int = 0, top_p: float = 1, max_len: int = 100,
            mini_batch_size: int = 128, complexity: int | float | None = None,
            temperature: float = 1.0, valid_only: bool = True, simplify: bool = True, unique: bool = True, verbose: bool = False) -> tuple[list[int], list[float], list[bool]]:
        # Prepare initial token based on complexity
        if complexity is None:
            initial_token, input_num = [self.tokenizer['<bos>']], None
        else:
            initial_token, input_num = self.preprocessor.format_complexity([self.tokenizer['<bos>']], complexity=complexity)

        memory = self._create_memory(data)

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
                logits = self.forward(input_ids_tensor, batch_data, input_num=input_num_tensor, memory=memory)

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
                    if sampled_token == self.tokenizer['<eos>']:
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
            encoded_expression, before, after = self.extract_expression_from_beam(seq)
            encoded_expression = self.constantify_expression(encoded_expression)  # type: ignore

            # Decode the sequence
            expression = self.tokenizer.decode(encoded_expression, special_tokens='<constant>')

            # Only process valid expressions
            if self.simplipy_engine.is_valid(expression) and len(expression) > 1:
                if simplify:
                    # Simplify the expression
                    expression = self.simplipy_engine.simplify(expression, max_pattern_length=4)

                    # Skip if we've already seen this simplified expression
                if unique and tuple(expression) in seen_expressions:
                    continue

                # Encode the simplified expression
                filtered_sequence = self.tokenizer.encode(expression, add_bos=True, add_eos=True)
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

    def constantify_expression(self, expression: list[int] | list[str]) -> list[int] | list[str]:
        # Replace mult4, div3 etc by multiplication with <constant>

        # Find out if the expression is encoded or not
        if isinstance(expression, list) and all(isinstance(token, int) for token in expression):
            # If it's encoded, we need to convert it to the tokenizer's string representation
            constantified_expression = []
            for token in expression:
                if re.match(r"^mult\d+$", self.tokenizer[token]) or re.match(r"^div\d+$", self.tokenizer[token]):
                    # Replace with '*', '<constant>
                    constantified_expression.append(self.tokenizer['*'])
                    constantified_expression.append(self.tokenizer['<constant>'])
                else:
                    constantified_expression.append(token)

        elif isinstance(expression, list) and all(isinstance(token, str) for token in expression):
            # If it's already a string representation, we can directly replace the patterns
            constantified_expression = []
            for token in expression:
                if re.match(r"^mult\d+$", token) or re.match(r"^div\d+$", token):  # type: ignore
                    # Replace with '*', '<constant>'
                    constantified_expression.append('*')
                    constantified_expression.append('<constant>')
                else:
                    constantified_expression.append(token)
        else:
            raise ValueError("Expression must be a list of integers or strings.")
        return constantified_expression

    def extract_expression_from_beam(self, beam: list[int]) -> tuple[list[int], list[int], list[int]]:

        if self.tokenizer['<bos>'] not in beam or self.tokenizer['<eos>'] not in beam:
            raise ValueError(f"Beam must contain <bos> and <eos> tokens. Got {beam}.")

        bos_index = beam.index(self.tokenizer['<bos>'])
        eos_index = beam.index(self.tokenizer['<eos>'])

        before = beam[:bos_index + 1]
        after = beam[eos_index:]

        return beam[bos_index + 1:eos_index], before, after

    def save(self, directory: str, config: dict[str, Any] | str | None = None, reference: str = 'relative', recursive: bool = True, errors: Literal['raise', 'warn', 'ignore'] = 'warn') -> None:

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
                filename='model.yaml',
                reference=reference,
                recursive=recursive,
                resolve_paths=True)

    @classmethod
    def load(cls, directory: str) -> tuple[dict[str, Any], "FlashANSRModel"]:
        directory = substitute_root_path(directory)

        config_path = os.path.join(directory, 'model.yaml')

        model = cls.from_config(config_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True))

        return load_config(config_path), model

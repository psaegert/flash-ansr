import os
import warnings
from typing import Any, Callable, Literal, Optional, Tuple, TypeAlias

import torch
from torch import nn
from tqdm import tqdm

from simplipy import SimpliPyEngine

from flash_ansr.utils import load_config, save_config, substitute_root_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.model.pre_encoder import IEEE75432PreEncoder
from flash_ansr.preprocess import FlashASNRPreprocessor
from flash_ansr.model.set_transformer import SetTransformer
from flash_ansr.model.transformer import TransformerDecoder
from flash_ansr.decoding.mcts import MonteCarloTreeSearch, MCTSConfig, PolicyStep


ValueFunction: TypeAlias = Callable[[Tuple[int, ...]], float]
TerminalFunction: TypeAlias = Callable[[Tuple[int, ...]], bool]


class FlashANSRModel(nn.Module):
    def __init__(
        self,
        simplipy_engine: SimpliPyEngine,
        tokenizer: Tokenizer,

        pre_encoder_noise_scale: float,

        encoder_max_n_variables: int,
        encoder_dim: int = 512,
        encoder_n_heads: int = 8,
        encoder_n_isab: int = 2,
        encoder_n_sab: int = 1,
        encoder_n_inducing_points: int = 32,
        encoder_n_seeds: int = 1,
        encoder_ffn_hidden_dim: int = 2048,
        encoder_dropout: float = 0.1,
        encoder_attn_norm: str = "none",
        encoder_ffn_norm: str = "none",
        encoder_output_norm: str = "none",

        decoder_input_dim: int = 512,
        decoder_model_dim: int = 512,
        decoder_n_layers: int = 6,
        decoder_n_heads: int = 8,
        decoder_max_seq_len: int = 4096,
        decoder_ffn_hidden_dim: int = None,
        decoder_dropout: float = 0.1,
        decoder_block_self_attn_norm: str = "rms",
        decoder_block_cross_attn_norm: str = "rms",
        decoder_block_ffn_norm: str = "rms",
        decoder_cross_attn_kv_norm: str = "rms",
        decoder_output_norm: str = "rms",
        decoder_use_rope_self_attn: bool = False,
        decoder_use_rope_cross_attn: bool = False,

        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.simplipy_engine = simplipy_engine
        self.tokenizer = tokenizer
        self.encoder_max_n_variables = encoder_max_n_variables

        self.pre_encoder = IEEE75432PreEncoder(input_size=encoder_max_n_variables)

        self.pre_encoder_numeric_tokens = IEEE75432PreEncoder(input_size=1)
        self.pre_encoder_noise_scale = pre_encoder_noise_scale
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
            dropout=encoder_dropout,
            attn_norm=encoder_attn_norm,
            ffn_norm=encoder_ffn_norm,
            output_norm=encoder_output_norm,
            use_checkpointing=use_checkpointing
        )

        if self.encoder.output_dim != decoder_model_dim:
            decoder_input_dim = self.encoder.output_dim

        self.decoder = TransformerDecoder(
            vocab_size=len(tokenizer),
            input_dim=decoder_input_dim,
            model_dim=decoder_model_dim,
            n_layers=decoder_n_layers,
            n_heads=decoder_n_heads,
            max_seq_len=decoder_max_seq_len,
            ffn_hidden_dim=decoder_ffn_hidden_dim,
            dropout=decoder_dropout,
            block_self_attn_norm_type=decoder_block_self_attn_norm,
            block_cross_attn_norm_type=decoder_block_cross_attn_norm,
            block_ffn_norm_type=decoder_block_ffn_norm,
            cross_attn_kv_norm_type=decoder_cross_attn_kv_norm,
            output_norm_type=decoder_output_norm,
            use_rope_self_attn=decoder_use_rope_self_attn,
            use_rope_cross_attn=decoder_use_rope_cross_attn,
            use_checkpointing=use_checkpointing,
        )

        self.next_token_head = nn.Sequential(
            nn.Linear(decoder_model_dim, decoder_model_dim),
            nn.GELU(),
            nn.Dropout(p=decoder_dropout),
            nn.Linear(decoder_model_dim, len(self.tokenizer)))

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

        if "model" in config_.keys():
            config_ = config_["model"]

        if isinstance(config, str) and isinstance(config_["simplipy_engine"], str):
            if config_["simplipy_engine"].startswith('.'):
                config_["simplipy_engine"] = os.path.join(os.path.dirname(config), config_["simplipy_engine"])

        simplipy_engine = SimpliPyEngine.load(config_["simplipy_engine"], install=True)
        tokenizer = Tokenizer.from_config(config_["tokenizer"])

        return cls(
            simplipy_engine=simplipy_engine,
            tokenizer=tokenizer,

            pre_encoder_noise_scale=config_["pre_encoder_noise_scale"],

            encoder_max_n_variables=config_["encoder_max_n_variables"],
            encoder_dim=config_["encoder_dim"],
            encoder_n_heads=config_["encoder_n_heads"],
            encoder_n_isab=config_["encoder_n_isab"],
            encoder_n_sab=config_["encoder_n_sab"],
            encoder_n_inducing_points=config_["encoder_n_inducing_points"],
            encoder_n_seeds=config_["encoder_n_seeds"],
            encoder_ffn_hidden_dim=config_["encoder_ffn_hidden_dim"],
            encoder_dropout=config_["encoder_dropout"],
            encoder_attn_norm=config_["encoder_attn_norm"],
            encoder_ffn_norm=config_["encoder_ffn_norm"],
            encoder_output_norm=config_["encoder_output_norm"],

            decoder_input_dim=config_["decoder_input_dim"],
            decoder_model_dim=config_["decoder_model_dim"],
            decoder_n_layers=config_["decoder_n_layers"],
            decoder_n_heads=config_["decoder_n_heads"],
            decoder_max_seq_len=config_["decoder_max_seq_len"],
            decoder_ffn_hidden_dim=config_["decoder_ffn_hidden_dim"],
            decoder_dropout=config_["decoder_dropout"],
            decoder_block_self_attn_norm=config_["decoder_block_self_attn_norm"],
            decoder_block_cross_attn_norm=config_["decoder_block_cross_attn_norm"],
            decoder_block_ffn_norm=config_["decoder_block_ffn_norm"],
            decoder_cross_attn_kv_norm=config_["decoder_cross_attn_kv_norm"],
            decoder_output_norm=config_["decoder_output_norm"],
            decoder_use_rope_self_attn=config_["decoder_use_rope_self_attn"],
            decoder_use_rope_cross_attn=config_["decoder_use_rope_cross_attn"],

            use_checkpointing=config_["use_checkpointing"],
        )

    def _create_memory(self, data: torch.Tensor, data_attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if data.ndim != 3:
            data = data.unsqueeze(0)

        # Pre-process input data
        data_pre_encodings: torch.Tensor = self.pre_encoder(data)
        B, M, D, E = data_pre_encodings.size()

        # If in training, add a small amount of noise to the pre-encodings for regularization
        if self.training:
            noise = torch.randn_like(data_pre_encodings) * self.pre_encoder_noise_scale
            data_pre_encodings = data_pre_encodings + noise

        # Encoder forward pass
        memory = self.encoder(data_pre_encodings.view(B, M, D * E), data_attn_mask)

        if memory.ndim > 3:
            memory = memory.view(B, -1, memory.size(-1))

        return memory

    def forward(self, input_tokens: torch.Tensor, data: torch.Tensor | None, input_num: torch.Tensor | None = None, memory: torch.Tensor | None = None, data_attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if memory is not None:
            self.memory = memory
        elif data is not None:
            self.memory = self._create_memory(data, data_attn_mask)
        elif self.memory is None:
            raise ValueError("Either `data` or `memory` must be provided for the first forward pass.")

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
        decoder_output = self.decoder(tokens=input_tokens, encoder_memory=self.memory, extra_parallel_embeddings=numeric_embeddings)

        logits = self.next_token_head(decoder_output)

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

        with torch.no_grad():
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

                    # Forward pass for the mini-batch
                    logits = self.forward(mini_batch, None, input_num=input_num_tensor, memory=memory)

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
                            candidate_expression, before, after = self.tokenizer.extract_expression_from_beam(candidate[0])
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
            constantified_seq = self.tokenizer.constantify_expression(seq)
            completed_sequences[i] = (constantified_seq, score)

        # Step 8: Sort all sequences by score (highest scores first)
        completed_sequences = sorted(completed_sequences, key=lambda x: x[1], reverse=True)

        # Step 9: Return the top 'beam_size' sequences
        return [seq for seq, _ in completed_sequences[:beam_width]], [score for _, score in completed_sequences[:beam_width]], [True] * len(completed_sequences[:beam_width])

    def mcts_decode(
        self,
        data: torch.Tensor,
        config: MCTSConfig,
        beam_width: int = 16,
        complexity: int | float | None = None,
        value_fn: Optional[ValueFunction] = None,
        terminal_fn: Optional[TerminalFunction] = None,
        invalid_sequence_fn: Optional[Callable[[Tuple[int, ...]], bool]] = None,
        completion_sort: str = "reward",
        verbose: bool = False,
    ) -> tuple[list[list[int]], list[float], list[bool], list[float]]:
        """Decode expressions using Monte Carlo Tree Search."""

        device = data.device

        if complexity is None:
            initial_tokens, input_num = [self.tokenizer['<bos>']], None
        else:
            initial_tokens, input_num = self.preprocessor.format_complexity([self.tokenizer['<bos>']], complexity=complexity)

        base_input_num = input_num[:] if input_num is not None else None

        memory = self._create_memory(data)

        policy_cache: dict[Tuple[int, ...], torch.Tensor] = {}

        def build_input_num_tensor(length: int) -> Optional[torch.Tensor]:
            if base_input_num is None:
                return None

            if length <= len(base_input_num):
                values = base_input_num[:length]
            else:
                values = base_input_num + [float('nan')] * (length - len(base_input_num))

            return torch.tensor(values, device=device).unsqueeze(0).unsqueeze(-1)

        def policy_fn(tokens: Tuple[int, ...], _: Optional[Any]) -> PolicyStep:
            if tokens in policy_cache:
                return PolicyStep(log_probs=policy_cache[tokens])

            input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
            input_num_tensor = build_input_num_tensor(len(tokens))

            with torch.no_grad():
                logits = self.forward(input_ids, None, input_num=input_num_tensor, memory=memory)
                next_logits = logits[:, -1, :].squeeze(0)
                log_probs = torch.log_softmax(next_logits, dim=-1)

            policy_cache[tokens] = log_probs
            return PolicyStep(log_probs=log_probs)

        value_callable: ValueFunction
        if value_fn is None:
            def default_value(_: Tuple[int, ...], /) -> float:
                return 0.0

            value_callable = default_value
        else:
            value_callable = value_fn

        terminal_callable: TerminalFunction
        if terminal_fn is None:
            eos_token = self.tokenizer['<eos>']

            def default_terminal(tokens: Tuple[int, ...], /) -> bool:
                return bool(tokens) and tokens[-1] == eos_token

            terminal_callable = default_terminal
        else:
            terminal_callable = terminal_fn

        eos_token_id = self.tokenizer['<eos>']
        pad_token_id = self.tokenizer['<pad>'] if '<pad>' in self.tokenizer else None

        mcts = MonteCarloTreeSearch(
            policy_fn=policy_fn,
            value_fn=value_callable,
            terminal_fn=terminal_callable,
            config=config,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            invalid_sequence_fn=invalid_sequence_fn,
        )

        with torch.no_grad():
            mcts.run(
                initial_tokens,
                initial_state=None,
                progress=verbose,
                progress_desc=f"MCTS decode ({config.simulations} sims)",
            )

        completions = mcts.get_top_completions(limit=beam_width, by=completion_sort)

        if not completions:
            try:
                best_child = mcts.best_child(by="visits")
                fallback_tokens = list(best_child.tokens)
                if not fallback_tokens or fallback_tokens[-1] != eos_token_id:
                    fallback_tokens.append(eos_token_id)
                completions = [(tuple(fallback_tokens), 0.0, best_child.log_prob)]
            except Exception:
                fallback_tokens = initial_tokens + [eos_token_id]
                completions = [(tuple(fallback_tokens), 0.0, 0.0)]

        sequences: list[list[int]] = []
        log_probs: list[float] = []
        rewards: list[float] = []

        seen_sequences: set[tuple[int, ...]] = set()

        for tokens, reward, log_prob in completions:
            seq = list(tokens)
            simplified_seq = seq

            try:
                expression_tokens, before, after = self.tokenizer.extract_expression_from_beam(seq)
            except ValueError:
                expression_tokens = None
            else:
                decoded_expression = self.tokenizer.decode(expression_tokens, special_tokens='<constant>')
                if self.simplipy_engine.is_valid(decoded_expression) and len(decoded_expression) > 1:
                    simplified_expression = self.simplipy_engine.simplify(decoded_expression, max_pattern_length=4)
                    simplified_seq = before + self.tokenizer.encode(simplified_expression) + after

            constantified = self.tokenizer.constantify_expression(simplified_seq)
            sequence_key = tuple(constantified)  # type: ignore[arg-type]

            if sequence_key in seen_sequences:
                continue

            seen_sequences.add(sequence_key)
            sequences.append(constantified)  # type: ignore[arg-type]
            log_probs.append(float(log_prob))
            rewards.append(float(reward))

        completed_flags = [True] * len(sequences)

        return sequences, log_probs, completed_flags, rewards

    def sample_top_kp(
        self, data: torch.Tensor, choices: int = 10, top_k: int = 0, top_p: float = 1, max_len: int = 100,
        mini_batch_size: int = 128, complexity: int | float | None = None,
        temperature: float = 1.0, valid_only: bool = True, simplify: bool = True, unique: bool = True, verbose: bool = False
    ) -> tuple[list[list[int]], list[float], list[bool]]:

        device = data.device

        # --- 1. Vectorized Initialization ---
        if complexity is None:
            initial_token, input_num = [self.tokenizer['<bos>']], None
        else:
            initial_token, input_num = self.preprocessor.format_complexity([self.tokenizer['<bos>']], complexity=complexity)

        # Pre-allocate tensors on the target device
        sequences = torch.full((choices, max_len), self.tokenizer['<pad>'], device=device, dtype=torch.long)
        sequences[:, 0] = initial_token[0]

        scores = torch.zeros(choices, device=device, dtype=torch.float)
        is_finished = torch.zeros(choices, device=device, dtype=torch.bool)

        memory = self._create_memory(data)
        input_num_tensor = torch.tensor(input_num, device=device).unsqueeze(-1) if input_num is not None else None

        # --- 2. Vectorized Generation Loop with Mini-batching ---
        with torch.no_grad():
            pbar = tqdm(total=max_len - 1, disable=not verbose, desc="Generating tokens")
            for current_length in range(1, max_len):
                if is_finished.all():
                    break

                # Get indices of all sequences that are not yet finished
                active_indices = (~is_finished).nonzero(as_tuple=True)[0]

                # Process active sequences in mini-batches to manage memory
                for start_idx in range(0, len(active_indices), mini_batch_size):
                    batch_indices = active_indices[start_idx: start_idx + mini_batch_size]

                    # Prepare input for the current batch
                    input_ids_tensor = sequences[batch_indices, :current_length]

                    # Forward pass for the batch
                    logits = self.forward(input_ids_tensor, None, input_num=input_num_tensor, memory=memory)
                    next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

                    original_scores = torch.log_softmax(next_token_logits, dim=-1)

                    # --- Filtering (top-k, top-p, temperature) ---
                    if top_k > 0:
                        top_k_val = min(top_k, next_token_logits.size(-1))
                        ignore_mask = next_token_logits < torch.topk(next_token_logits, top_k_val, dim=1)[0][..., -1, None]
                        next_token_logits[ignore_mask] = -float('Inf')

                    if temperature != 1.0:
                        next_token_logits /= temperature

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('Inf')

                    # Sample next tokens for the batch
                    probs = torch.softmax(next_token_logits, dim=-1)
                    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    # Update the main tensors with the results for the batch
                    sequences[batch_indices, current_length] = sampled_tokens
                    scores[batch_indices] += torch.gather(original_scores, 1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
                    is_finished[batch_indices] |= (sampled_tokens == self.tokenizer['<eos>'])

                pbar.update(1)
            pbar.close()

        # Move results to CPU for post-processing
        completed_sequences = sequences.cpu().tolist()
        completed_scores = scores.cpu().tolist()

        # --- 3. Single-threaded Post-processing (same as original) ---
        filtered_sequences = []
        filtered_scores = []
        filtered_is_valid = []
        seen_expressions = set()

        pbar_post = tqdm(zip(completed_sequences, completed_scores), total=len(completed_sequences), disable=not verbose, desc="Post-processing")
        for seq, score in pbar_post:
            try:
                # Assuming these methods exist on your tokenizer/class
                encoded_expression, _, _ = self.tokenizer.extract_expression_from_beam(seq)
                encoded_expression = self.tokenizer.constantify_expression(encoded_expression)
            except (ValueError, IndexError):
                continue

            expression = self.tokenizer.decode(encoded_expression, special_tokens='<constant>')

            if self.simplipy_engine.is_valid(expression) and len(expression) > 1:
                if simplify:
                    expression = self.simplipy_engine.simplify(expression, max_pattern_length=4)

                expression_tuple = tuple(expression)
                if unique and expression_tuple in seen_expressions:
                    continue

                filtered_sequence = self.tokenizer.encode(expression, add_bos=True, add_eos=True)
                filtered_sequences.append(filtered_sequence)
                filtered_scores.append(score)
                filtered_is_valid.append(True)

                if unique:
                    seen_expressions.add(expression_tuple)

            elif not valid_only:
                # Clean up the sequence to remove padding before appending
                try:
                    end_idx = seq.index(self.tokenizer['<eos>']) + 1
                    filtered_sequences.append(seq[:end_idx])
                except ValueError:
                    filtered_sequences.append(seq)  # Append full sequence if no <eos>
                filtered_scores.append(score)
                filtered_is_valid.append(False)

        # Sort final results by score
        sorted_indices = sorted(range(len(filtered_scores)), key=lambda i: filtered_scores[i], reverse=True)  # type: ignore
        final_sequences = [filtered_sequences[i] for i in sorted_indices]
        final_scores = [filtered_scores[i] for i in sorted_indices]
        final_is_valid = [filtered_is_valid[i] for i in sorted_indices]

        return final_sequences, final_scores, final_is_valid

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

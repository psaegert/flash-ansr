import heapq
import math
import os
import warnings
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, TypeAlias

import torch
from torch import nn
from tqdm import tqdm

from simplipy import SimpliPyEngine

from flash_ansr.utils import load_config, save_config, substitute_root_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.model.pre_encoder import IEEE75432PreEncoder
from flash_ansr.preprocess import FlashANSRPreprocessor
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
        use_numeric_head: bool = False,
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

        self._configured_decoder_max_seq_len = int(decoder_max_seq_len)

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

        self.use_numeric_head = bool(use_numeric_head)
        self.constant_value_head: nn.Module | None
        if self.use_numeric_head:
            self.constant_value_head = nn.Sequential(
                nn.Linear(decoder_model_dim, decoder_model_dim),
                nn.GELU(),
                nn.Linear(decoder_model_dim, 1),
            )
        else:
            self.constant_value_head = None

        self.preprocessor = FlashANSRPreprocessor(simplipy_engine=simplipy_engine, tokenizer=tokenizer)

        self.memory: torch.Tensor | None = None
        self._latest_numeric_sequences: list[list[float]] | None = None

    @staticmethod
    def _split_forward_output(output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(output, (tuple, list)):
            logits, numeric = output
            return logits, numeric  # type: ignore[return-value]
        return output, None  # type: ignore[return-value]

    @staticmethod
    def _normalize_numeric_prefix(tokens: Sequence[int], numeric: Sequence[float] | torch.Tensor | None) -> list[float]:
        seq_len = len(tokens)
        if numeric is None:
            return [float('nan')] * seq_len

        if isinstance(numeric, torch.Tensor):
            numeric_list = [float(value) for value in numeric.view(-1).tolist()]
        else:
            numeric_list = [float(value) if value is not None else float('nan') for value in numeric]

        if len(numeric_list) < seq_len:
            numeric_list = numeric_list + [float('nan')] * (seq_len - len(numeric_list))
        elif len(numeric_list) > seq_len:
            numeric_list = numeric_list[:seq_len]

        return numeric_list

    @staticmethod
    def _numeric_sequences_to_tensor(numeric_sequences: Sequence[Sequence[float]], device: torch.device) -> torch.Tensor:
        numeric_tensor = torch.tensor(numeric_sequences, device=device, dtype=torch.float32)
        return numeric_tensor.unsqueeze(-1)

    def get_latest_numeric_sequences(self) -> list[list[float]] | None:
        """Return the numeric sequences from the most recent decode call."""
        if self._latest_numeric_sequences is None:
            return None
        return [list(sequence) for sequence in self._latest_numeric_sequences]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def decoder_max_seq_len(self) -> int:
        """Return the configured maximum decoder sequence length."""
        if hasattr(self, '_configured_decoder_max_seq_len'):
            return int(self._configured_decoder_max_seq_len)

        rope = getattr(getattr(self, 'decoder', None), 'rope', None)
        if rope is None or not hasattr(rope, 'max_seq_len'):
            raise AttributeError("Decoder does not expose a rotary embedding max_seq_len")

        return int(getattr(rope, 'max_seq_len'))

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
            use_numeric_head=config_.get("use_numeric_head", False),
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
            if numeric_embeddings.dim() == 4 and numeric_embeddings.size(-2) == 1:
                numeric_embeddings = numeric_embeddings.squeeze(-2)
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
        numeric_prediction = None
        if self.use_numeric_head and self.constant_value_head is not None:
            numeric_prediction = self.constant_value_head(decoder_output).squeeze(-1)

        if numeric_prediction is None:
            return logits
        return logits, numeric_prediction  # type: ignore[return-value]

    def _resolve_generation_prefix(
        self,
        *,
        initial_tokens: list[int] | None,
        input_num: Sequence[float] | None,
        complexity: int | float | None,
    ) -> tuple[list[int], list[float]]:
        if initial_tokens is not None:
            tokens = list(initial_tokens)
            numeric = self._normalize_numeric_prefix(tokens, input_num)
            return tokens, numeric

        if self.preprocessor is None:
            tokens = [self.tokenizer['<bos>']]
            return tokens, [float('nan')]

        serialized = self.preprocessor.serialize_prompt_prefix(complexity=complexity)

        tokens = [int(token) for token in serialized['input_ids']]
        numeric_values = self._normalize_numeric_prefix(tokens, serialized.get('input_num'))

        return tokens, numeric_values

    def beam_search(
        self,
        data: torch.Tensor,
        beam_width: int = 4,
        max_len: int = 100,
        mini_batch_size: int = 128,
        equivalence_pruning: bool = True,
        complexity: int | float | None = None,
        verbose: bool = False,
        limit_expansions: bool = True,
        *,
        initial_tokens: list[int] | None = None,
        input_num: Sequence[float] | None = None,
    ) -> tuple[list[list[int]], list[float], list[bool]]:
        base_tokens, base_numeric = self._resolve_generation_prefix(
            initial_tokens=initial_tokens,
            input_num=input_num,
            complexity=complexity,
        )

        equivalence_pruning = equivalence_pruning and not self.use_numeric_head

        beams: list[tuple[list[int], list[float], float]] = [(base_tokens, base_numeric, 0.0)]
        completed_sequences_heap: list[tuple[float, tuple[int, ...]]] = []
        completed_sequences_scores: dict[tuple[int, ...], float] = {}
        completed_sequences_numeric: dict[tuple[int, ...], list[float]] = {}
        simplify_cache: dict[tuple[int, ...], tuple[int, ...]] = {}
        n_pruned = 0

        memory = self._create_memory(data)

        eos_token_id = self.tokenizer['<eos>']
        pad_token_id = self.tokenizer['<pad>']
        constant_token_id = self.tokenizer['<constant>']

        pbar = tqdm(total=max_len, disable=not verbose, desc=f"Generating beams (max length: {max_len})", smoothing=0.0)

        def register_completed_sequence(seq_tuple: tuple[int, ...], numeric_seq: list[float], score: float) -> None:
            nonlocal n_pruned

            existing_score = completed_sequences_scores.get(seq_tuple)
            if existing_score is not None and score <= existing_score:
                n_pruned += 1
                return

            completed_sequences_scores[seq_tuple] = score
            completed_sequences_numeric[seq_tuple] = list(numeric_seq)
            heapq.heappush(completed_sequences_heap, (score, seq_tuple))

            while len(completed_sequences_scores) > beam_width:
                prune_score, prune_key = heapq.heappop(completed_sequences_heap)
                current_score = completed_sequences_scores.get(prune_key)
                if current_score is None:
                    continue
                if current_score != prune_score:
                    continue
                del completed_sequences_scores[prune_key]
                completed_sequences_numeric.pop(prune_key, None)
                n_pruned += 1
                break

        def predict_constant_value(seq: list[int], numeric_seq: list[float]) -> float:
            if not self.use_numeric_head or self.constant_value_head is None:
                return float('nan')

            input_ids = torch.tensor([seq], device=data.device, dtype=torch.long)
            numeric_tensor = self._numeric_sequences_to_tensor([numeric_seq], device=data.device)
            output = self.forward(input_ids, None, input_num=numeric_tensor, memory=memory)
            _, numeric_pred = self._split_forward_output(output)
            if numeric_pred is None:
                return float('nan')
            return float(numeric_pred[0, len(seq) - 1].item())

        with torch.no_grad():
            for _ in range(max_len):
                active_sequences: list[list[int]] = []
                active_numeric: list[list[float]] = []
                active_scores: list[float] = []

                for seq, numeric_seq, score in beams:
                    if seq[-1] == eos_token_id:
                        register_completed_sequence(tuple(seq), numeric_seq, score)
                        continue  # Don't expand this sequence further
                    active_sequences.append(seq)
                    active_numeric.append(numeric_seq)
                    active_scores.append(score)

                if len(active_sequences) == 0:
                    break  # All sequences completed

                max_seq_len = max(len(seq) for seq in active_sequences)
                input_ids_padded = [seq + [pad_token_id] * (max_seq_len - len(seq)) for seq in active_sequences]
                numeric_padded = [numeric_seq + [float('nan')] * (max_seq_len - len(numeric_seq)) for numeric_seq in active_numeric]
                input_ids_tensor = torch.tensor(input_ids_padded, device=data.device, dtype=torch.long)
                numeric_tensor = self._numeric_sequences_to_tensor(numeric_padded, device=data.device)

                all_next_token_logits: list[torch.Tensor] = []
                for start_idx in range(0, len(active_sequences), mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, len(active_sequences))
                    mini_batch = input_ids_tensor[start_idx:end_idx]
                    numeric_batch = numeric_tensor[start_idx:end_idx] if numeric_tensor is not None else None

                    output = self.forward(mini_batch, None, input_num=numeric_batch, memory=memory)
                    logits, _ = self._split_forward_output(output)

                    all_next_token_logits.append(logits[:, -1, :])  # Shape: (mini_batch_size, vocab_size)

                next_token_logits = torch.cat(all_next_token_logits, dim=0)
                next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

                vocab_size = next_token_log_probs.size(-1)

                if limit_expansions:
                    expansion_factor = 2 if equivalence_pruning else 1
                    expansion_per_beam = max(1, min(vocab_size, beam_width * expansion_factor))
                    top_log_probs, top_token_ids = torch.topk(next_token_log_probs, k=expansion_per_beam, dim=-1)
                else:
                    expansion_per_beam = vocab_size
                    top_log_probs = next_token_log_probs
                    top_token_ids = torch.arange(vocab_size, device=next_token_log_probs.device, dtype=torch.long).unsqueeze(0).expand(next_token_log_probs.size(0), -1)

                active_scores_tensor = torch.tensor(active_scores, device=data.device)
                candidate_scores = active_scores_tensor.unsqueeze(-1) + top_log_probs

                flat_scores = candidate_scores.reshape(-1)
                flat_parent_indices = torch.arange(len(active_sequences), device=data.device).repeat_interleave(expansion_per_beam)
                flat_token_indices = top_token_ids.reshape(-1)

                sorted_scores, sorted_indices = torch.sort(flat_scores, descending=True)

                next_beams: list[tuple[list[int], list[float], float]] = []
                next_beam_set: set[tuple[int, ...]] = set()

                for rank_idx in range(sorted_indices.numel()):
                    parent_idx = int(flat_parent_indices[sorted_indices[rank_idx]])
                    token_id = int(flat_token_indices[sorted_indices[rank_idx]])
                    base_seq = active_sequences[parent_idx]
                    base_numeric_seq = active_numeric[parent_idx]
                    new_seq = base_seq + [token_id]
                    new_numeric = base_numeric_seq + [float('nan')]
                    new_score = float(sorted_scores[rank_idx].item())

                    if token_id == eos_token_id:
                        numeric_for_completion = new_numeric
                        if equivalence_pruning:
                            candidate_expression, before, after = self.tokenizer.extract_expression_from_beam(new_seq)
                            expr_key = tuple(candidate_expression)

                            simplified_tuple = simplify_cache.get(expr_key)
                            if simplified_tuple is None:
                                candidate_expression_decoded = self.tokenizer.decode(candidate_expression, special_tokens='<constant>')

                                if not self.simplipy_engine.is_valid(candidate_expression_decoded) or len(candidate_expression_decoded) <= 1:
                                    n_pruned += 1
                                    continue

                                simplified_tokens = self.tokenizer.encode(
                                    self.simplipy_engine.simplify(candidate_expression_decoded, max_pattern_length=4)
                                )
                                simplified_tuple = tuple(before + simplified_tokens + after)
                                simplify_cache[expr_key] = simplified_tuple
                        else:
                            simplified_tuple = tuple(new_seq)

                        register_completed_sequence(simplified_tuple, numeric_for_completion, new_score)
                        continue
                    else:
                        seq_tuple = tuple(new_seq)
                        if equivalence_pruning:
                            if seq_tuple in next_beam_set:
                                n_pruned += 1
                                continue
                            next_beam_set.add(seq_tuple)

                        if token_id == constant_token_id:
                            predicted_value = predict_constant_value(new_seq, new_numeric)
                            if math.isfinite(predicted_value):
                                new_numeric[-1] = predicted_value

                        next_beams.append((new_seq, new_numeric, new_score))

                    if len(next_beams) >= beam_width:
                        break

                beams = next_beams[:beam_width]

                if not beams:
                    break

                pbar.set_postfix({'completed': len(completed_sequences_scores), 'pruned': n_pruned})
                pbar.update(1)

        # Step 7: Combine completed sequences with current beams (in case they are incomplete)
        combined_sequences: list[tuple[list[int], list[float], float]] = [
            (list(seq_tuple), list(completed_sequences_numeric.get(seq_tuple, [float('nan')] * len(seq_tuple))), score)
            for seq_tuple, score in completed_sequences_scores.items()
        ]
        combined_sequences.extend(beams)

        if not self.use_numeric_head:
            for i, (seq, numeric_seq, score) in enumerate(combined_sequences):
                constantified_seq = self.tokenizer.constantify_expression(seq)
                combined_sequences[i] = (constantified_seq, numeric_seq, score)

        # Step 8: Sort all sequences by score (highest scores first)
        combined_sequences = sorted(combined_sequences, key=lambda x: x[2], reverse=True)

        # Step 9: Return the top 'beam_size' sequences
        top_sequences = combined_sequences[:beam_width]
        if self.use_numeric_head:
            self._latest_numeric_sequences = [list(numeric_seq) for _, numeric_seq, _ in top_sequences]
        else:
            self._latest_numeric_sequences = None

        return (
            [seq for seq, _, _ in top_sequences],
            [score for _, _, score in top_sequences],
            [True] * len(top_sequences),
        )

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
        *,
        initial_tokens: list[int] | None = None,
        input_num: list[float] | None = None,
    ) -> tuple[list[list[int]], list[float], list[bool], list[float]]:
        """Decode expressions using Monte Carlo Tree Search."""

        device = data.device

        base_tokens, base_input_num = self._resolve_generation_prefix(
            initial_tokens=initial_tokens,
            input_num=input_num,
            complexity=complexity,
        )

        memory = self._create_memory(data)

        policy_cache: dict[Tuple[int, ...], torch.Tensor] = {}

        def build_input_num_tensor(length: int) -> Optional[torch.Tensor]:
            if length <= len(base_input_num):
                values = base_input_num[:length]
            else:
                values = base_input_num + [float('nan')] * (length - len(base_input_num))

            return torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        def policy_fn(tokens: Tuple[int, ...], _: Optional[Any]) -> PolicyStep:
            if tokens in policy_cache:
                return PolicyStep(log_probs=policy_cache[tokens])

            input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
            input_num_tensor = build_input_num_tensor(len(tokens))

            with torch.no_grad():
                output = self.forward(input_ids, None, input_num=input_num_tensor, memory=memory)
                logits, _ = self._split_forward_output(output)
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
                base_tokens,
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
                fallback_tokens = base_tokens + [eos_token_id]
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

        self._latest_numeric_sequences = None

        return sequences, log_probs, completed_flags, rewards

    def sample_top_kp(
        self,
        data: torch.Tensor,
        choices: int = 10,
        top_k: int = 0,
        top_p: float = 1,
        max_len: int = 100,
        mini_batch_size: int = 128,
        complexity: int | float | None = None,
        temperature: float = 1.0,
        valid_only: bool = True,
        simplify: bool = True,
        unique: bool = True,
        verbose: bool = False,
        *,
        initial_tokens: list[int] | None = None,
        input_num: Sequence[float] | None = None,
    ) -> tuple[list[list[int]], list[float], list[bool]]:

        device = data.device

        # --- 1. Vectorized Initialization ---
        base_tokens, base_numeric = self._resolve_generation_prefix(
            initial_tokens=initial_tokens,
            input_num=input_num,
            complexity=complexity,
        )

        prefix_length = len(base_tokens)
        if prefix_length > max_len:
            raise ValueError(f"Initial token prefix length ({prefix_length}) exceeds max_len ({max_len}).")

        # Pre-allocate tensors on the target device
        sequences = torch.full((choices, max_len), self.tokenizer['<pad>'], device=device, dtype=torch.long)
        if prefix_length > 0:
            prefix_tensor = torch.tensor(base_tokens, device=device, dtype=torch.long)
            sequences[:, :prefix_length] = prefix_tensor

        numeric_sequences = torch.full((choices, max_len), float('nan'), device=device, dtype=torch.float32)
        if prefix_length > 0:
            numeric_prefix = torch.tensor(base_numeric, device=device, dtype=torch.float32)
            numeric_sequences[:, :prefix_length] = numeric_prefix

        scores = torch.zeros(choices, device=device, dtype=torch.float)
        is_finished = torch.zeros(choices, device=device, dtype=torch.bool)

        eos_token = self.tokenizer['<eos>']
        constant_token_id = self.tokenizer['<constant>']
        if prefix_length > 0 and base_tokens[-1] == eos_token:
            is_finished[:] = True

        memory = self._create_memory(data)

        def predict_constant_value(seq_tensor: torch.Tensor, numeric_tensor: torch.Tensor) -> float:
            if not self.use_numeric_head or self.constant_value_head is None:
                return float('nan')

            input_ids = seq_tensor.to(device=device, dtype=torch.long).unsqueeze(0)
            numeric_input = numeric_tensor.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            output = self.forward(input_ids, None, input_num=numeric_input, memory=memory)
            _, numeric_pred = self._split_forward_output(output)
            if numeric_pred is None:
                return float('nan')
            return float(numeric_pred[0, input_ids.shape[1] - 1].item())

        # --- 2. Vectorized Generation Loop with Mini-batching ---
        with torch.no_grad():
            total_steps = max(0, max_len - prefix_length)
            pbar = tqdm(total=total_steps, disable=not verbose, desc="Generating tokens", smoothing=0.0)
            for current_length in range(prefix_length, max_len):
                if is_finished.all():
                    break

                active_indices = (~is_finished).nonzero(as_tuple=True)[0]
                if active_indices.numel() == 0:
                    break

                for start_idx in range(0, len(active_indices), mini_batch_size):
                    batch_indices = active_indices[start_idx: start_idx + mini_batch_size]

                    input_ids_tensor = sequences[batch_indices, :current_length]
                    input_num_tensor: torch.Tensor | None
                    if current_length > 0:
                        input_num_tensor = numeric_sequences[batch_indices, :current_length].clone().unsqueeze(-1)
                    else:
                        input_num_tensor = None

                    output = self.forward(input_ids_tensor, None, input_num=input_num_tensor, memory=memory)
                    logits, _ = self._split_forward_output(output)
                    next_token_logits = logits[:, -1, :]

                    original_scores = torch.log_softmax(next_token_logits, dim=-1)

                    if top_k > 0:
                        top_k_val = min(top_k, next_token_logits.size(-1))
                        ignore_mask = next_token_logits < torch.topk(next_token_logits, top_k_val, dim=1)[0][..., -1, None]
                        next_token_logits[ignore_mask] = -float('inf')

                    if temperature != 1.0:
                        next_token_logits /= temperature

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('inf')

                    probs = torch.softmax(next_token_logits, dim=-1)
                    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    sequences[batch_indices, current_length] = sampled_tokens
                    scores[batch_indices] += torch.gather(original_scores, 1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
                    is_finished[batch_indices] |= (sampled_tokens == eos_token)

                    if self.use_numeric_head and self.constant_value_head is not None:
                        for local_idx, global_idx in enumerate(batch_indices.tolist()):
                            token_id = int(sampled_tokens[local_idx].item())
                            if token_id != constant_token_id:
                                continue

                            seq_len = current_length + 1
                            seq_tensor = sequences[global_idx, :seq_len].clone()
                            numeric_tensor = numeric_sequences[global_idx, :seq_len].clone()
                            numeric_tensor[-1] = float('nan')
                            predicted_value = predict_constant_value(seq_tensor, numeric_tensor)
                            if math.isfinite(predicted_value):
                                numeric_sequences[global_idx, current_length] = float(predicted_value)

                pbar.update(1)
            pbar.close()

        completed_sequences = sequences.cpu().tolist()
        completed_scores = scores.cpu().tolist()
        completed_numeric = numeric_sequences.cpu().tolist()

        filtered_sequences: list[list[int]] = []
        filtered_scores: list[float] = []
        filtered_is_valid: list[bool] = []
        filtered_numeric: list[list[float]] = []
        seen_expressions: set[tuple[str, ...]] = set()

        post_iterable = zip(completed_sequences, completed_scores, completed_numeric)
        pbar_post = tqdm(post_iterable, total=len(completed_sequences), disable=not verbose, desc="Post-processing", smoothing=0.0)
        for seq, score, numeric_values in pbar_post:
            try:
                encoded_expression, before, after = self.tokenizer.extract_expression_from_beam(seq)
            except (ValueError, IndexError):
                continue

            encoded_expression = self.tokenizer.constantify_expression(encoded_expression)
            expression = self.tokenizer.decode(encoded_expression, special_tokens='<constant>')

            if self.simplipy_engine.is_valid(expression) and len(expression) > 1:
                if simplify:
                    expression = self.simplipy_engine.simplify(expression, max_pattern_length=4)

                expression_tuple = tuple(expression)
                if unique and expression_tuple in seen_expressions:
                    continue

                expression_tokens = self.tokenizer.encode(expression)
                reconstructed_sequence = before + expression_tokens + after
                filtered_sequences.append(reconstructed_sequence)
                filtered_scores.append(score)
                filtered_is_valid.append(True)
                filtered_numeric.append(numeric_values)

                if unique:
                    seen_expressions.add(expression_tuple)

            elif not valid_only:
                try:
                    end_idx = seq.index(eos_token) + 1
                    filtered_sequences.append(seq[:end_idx])
                except ValueError:
                    filtered_sequences.append(seq)
                filtered_scores.append(score)
                filtered_is_valid.append(False)
                filtered_numeric.append(numeric_values)

        pbar_post.close()

        sorted_order = sorted(range(len(filtered_scores)), key=lambda i: filtered_scores[i], reverse=True)  # type: ignore[arg-type]
        final_sequences = [filtered_sequences[i] for i in sorted_order]
        final_scores = [filtered_scores[i] for i in sorted_order]
        final_is_valid = [filtered_is_valid[i] for i in sorted_order]
        if self.use_numeric_head:
            self._latest_numeric_sequences = [list(filtered_numeric[i]) for i in sorted_order]
        else:
            self._latest_numeric_sequences = None

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

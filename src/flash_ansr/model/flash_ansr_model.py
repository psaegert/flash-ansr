"""The :class:`FlashANSRModel` transformer backbone and its decoding routines."""
import heapq
import os
import warnings
from typing import Any, Callable, Literal, Optional, Tuple, TypeAlias

import torch
from torch import nn
from tqdm import tqdm

from simplipy import SimpliPyEngine

from flash_ansr.utils.config_io import load_config, save_config
from flash_ansr.utils.paths import substitute_root_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.model.pre_encoder import IEEE75432PreEncoder, IEEE75416PreEncoder
from flash_ansr.preprocessing import FlashANSRPreprocessor, PromptPrefix
from flash_ansr.model.encoders import SetTransformer
from flash_ansr.model.decoders import TransformerDecoder
from flash_ansr.model.decoders.static_kv import StaticKVCache
from flash_ansr.decoding.mcts import MonteCarloTreeSearch, MCTSConfig, PolicyStep
from flash_ansr.utils.sympy_timeout import _sympy_simplify_with_timeout


ValueFunction: TypeAlias = Callable[[Tuple[int, ...]], float]
TerminalFunction: TypeAlias = Callable[[Tuple[int, ...]], bool]

# Engagement counter for the static-decode path: incremented every time `_sample_top_kp_static` runs.
# Gates read this to PROVE the static path actually engaged (vs `static_decode` silently failing to
# propagate through generate()'s arms -> dynamic fallback, which would false-pass a quality/equality gate
# as dynamic-vs-dynamic). This is the assertion gate_overlap.py enforces via `_overlap_supported()`.
STATIC_DECODE_CALL_COUNT = 0

# Global kill-switch for the deployed static-decode default (default ON, 2026-06-24). When True,
# `static_decode=None` (the config/library default) resolves to static iff the model is capable AND the
# decode is MULTI-chunk (the resolved static batch < choices) -- see `_resolve_static_decode`. Set False to
# force the dynamic path everywhere. RATIONALE: the cross-GPU SPEED grid (consumer Turing/Ampere + datacenter
# A100/H100/H200) showed static wins in the multi-chunk regime on EVERY measured card; the only losses are
# single-chunk small-model cells, which the `batch < choices` predicate excludes by construction. So this one
# runtime predicate replaces the old per-(scale,c) enable table + scale-class/c-band + hardware gate -- the
# regime is computed from each card's own free VRAM, so it is hardware-general with no table to maintain.
# (Was _DEPLOYED_STATIC_DEFAULT.) See PREREG_overlap_default_and_static_global.md + INFERENCE_SPEED_FINDINGS.md.
_DEPLOYED_STATIC_ENABLED = True

# The runtime spill-guard trips when a static-decode chunk's live ALLOCATED working set exceeds this
# fraction of the card's physical memory. Set near the physical limit: the auto chunk size
# (suggest_batch_size_dims) is already conservative (~0.7 of FREE), so this is a backstop for an explicit
# oversized batch, not a second-guess of the cap. 0.9 of FREE passes the validated static b256 (~9.5 GB
# transient of ~18.6 free) with margin and trips on a gross over-budget chunk.
_VRAM_GUARD_FRACTION = 0.9

# Per-row pad the spill-guard uses, DELIBERATELY LARGER than suggest_batch_size_dims's cap overhead (1.6)
# so the guard is an INDEPENDENT backstop: a guard self-consistent with the cap could never catch a cap
# under-estimate. At 2.0 the guard passes the validated static b256 but trips b384+ (the measured spill
# onset) -- tighter than the cap, looser than reality, exactly a backstop.
_GUARD_OVERHEAD = 2.0


class FlashANSRModel(nn.Module):
    """Transformer backbone that maps ``(X, y)`` data to expression-token sequences.

    Couples a :class:`~flash_ansr.model.encoders.SetTransformer` set encoder (fed by an IEEE-754
    bit pre-encoder) with an autoregressive :class:`~flash_ansr.model.decoders.TransformerDecoder`,
    and exposes the beam-search / top-k-p / MCTS decoding routines used to generate candidate
    expression skeletons.
    """

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
        decoder_use_xsa_self_attn: bool = False,

        decoder_block_norm_position: str = "pre",
        encoder_block_norm_position: str = "pre",
        pre_encoder_bits: int = 32,

        use_checkpointing: bool = False,

        optional_condition: bool = False,
        null_memory_init_seed: int = 0,
    ) -> None:
        super().__init__()

        self.simplipy_engine = simplipy_engine
        self.tokenizer = tokenizer
        self.encoder_max_n_variables = encoder_max_n_variables

        if pre_encoder_bits == 32:
            pre_encoder_cls = IEEE75432PreEncoder
        elif pre_encoder_bits == 16:
            pre_encoder_cls = IEEE75416PreEncoder
        else:
            raise ValueError(f"pre_encoder_bits must be 16 or 32, got {pre_encoder_bits}")
        self.pre_encoder_bits = pre_encoder_bits

        self.pre_encoder = pre_encoder_cls(input_size=encoder_max_n_variables)

        self.pre_encoder_numeric_tokens = pre_encoder_cls(input_size=1)
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
            norm_position=encoder_block_norm_position,
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
            use_xsa_self_attn=decoder_use_xsa_self_attn,
            block_norm_position=decoder_block_norm_position,
            use_checkpointing=use_checkpointing,
        )

        self.next_token_head = nn.Sequential(
            nn.Linear(decoder_model_dim, decoder_model_dim),
            nn.GELU(),
            nn.Dropout(p=decoder_dropout),
            nn.Linear(decoder_model_dim, len(self.tokenizer)))

        self.preprocessor = FlashANSRPreprocessor(simplipy_engine=simplipy_engine, tokenizer=tokenizer)

        self.memory: torch.Tensor | None = None

        # Optional condition (classifier-free guidance), default OFF -> exactly the original behavior.
        # When enabled, the model carries a learned `null_memory` and `forward` accepts a per-example
        # `condition_mask` (True = conditioned, use the (X, y) encoder memory; False = unconditioned, use
        # `null_memory`). Conditioning is thus a FIRST-CLASS, DATA-DRIVEN input mode: the data marks each
        # example conditioned or not, and train/inference are symmetric (NO self.training gating, NOT a
        # train-time dropout). The `null_memory` parameter is created ONLY when enabled, so existing
        # checkpoints/configs load unchanged, and is initialized from a LOCAL generator so enabling the
        # feature consumes zero global RNG. (PROPOSAL_cfg_condition_dropout.md)
        self.optional_condition = bool(optional_condition)
        self.null_memory_init_seed = int(null_memory_init_seed)
        if self.optional_condition:
            _null_gen = torch.Generator(device="cpu").manual_seed(self.null_memory_init_seed)
            null_init = torch.randn(1, encoder_n_seeds, self.encoder.output_dim, generator=_null_gen) * 0.01
            self.null_memory = nn.Parameter(null_init)

    @property
    def device(self) -> torch.device:
        """The device on which the model's parameters currently reside."""
        return next(self.parameters()).device

    @property
    def n_params(self) -> int:
        """The total number of trainable (``requires_grad``) parameters in the model."""
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

    def supports_static_decode(self) -> tuple[bool, str]:
        """Whether the static-shape decode path (`_sample_top_kp_static`) is faithful for THIS model.

        Returns ``(ok, reason)``; ``reason`` names the first failing precondition (empty when ok). The
        static path was VALIDATED (greedy bit-identical + stochastic-logit gate) ONLY for the deployed
        v23.0 config: every decoder layer pre-norm, RoPE-self / no-cross-RoPE. XSA is supported (its
        own-value orthogonalization is bit-identical static-vs-dynamic; see test_xsa_static.py).
        The static forward enforces some of these (cross-RoPE raises inside ``forward_static_*``;
        self-RoPE is applied conditionally on ``use_rope``), but a post-norm or otherwise-untested config
        could diverge with NO error. So this is a POSITIVE, every-layer capability gate -- only run the path
        on a config it was validated for -- broader than the inside raises, which remain a hard backstop.
        Pure attribute reads; the deployed default and the cert harness both gate on this."""
        decoder = getattr(self, 'decoder', None)
        if decoder is None or not getattr(decoder, 'layers', None):
            return False, "no decoder layers"
        try:
            if not (self.decoder_max_seq_len and self.decoder_max_seq_len > 0):
                return False, "decoder_max_seq_len not positive/finite"
        except AttributeError:
            return False, "decoder_max_seq_len unavailable"
        for i, layer in enumerate(decoder.layers):
            if getattr(layer, 'norm_position', 'pre') != 'pre':
                return False, f"layer {i} is post-norm (static decode assumes pre-norm)"
            sa = getattr(layer, 'self_attention', None)
            ca = getattr(layer, 'cross_attention', None)
            if sa is None or ca is None:
                return False, f"layer {i} missing self/cross attention"
            if not getattr(sa, 'use_rope', False):
                return False, f"layer {i} self-attention has no RoPE (static applies it unconditionally)"
            if getattr(ca, 'use_rope', False):
                return False, f"layer {i} cross-attention uses RoPE (static drops it)"
        return True, ""

    def _resolve_static_decode(self, static_decode: bool | None, choices: int | None = None,
                               static_batch: int | None = None) -> bool:
        """Resolve the tri-state `static_decode` to a concrete bool.

        Explicit ``True`` on an unsupported model -> warn + fall back to dynamic (never let a deployed
        library raise from an opt-in speed flag; the inside-`_sample_top_kp_static` raises remain a hard
        backstop). Explicit ``False`` -> dynamic (direct callers needing the dynamic path's activation hooks
        -- recorder / interventions -- pass ``False``). ``None`` (the config/library default) resolves via
        the MULTI-CHUNK regime predicate: static iff the model is capable AND the global kill-switch
        (`_DEPLOYED_STATIC_ENABLED`) is on AND the decode spans multiple chunks (``static_batch < choices``).
        The cross-GPU SPEED grid (2026-06-24) showed static wins in the multi-chunk regime on every measured
        GPU class (consumer + datacenter); the only losses are single-chunk small-model cells, which this
        predicate excludes by construction -- so one runtime test replaces the old per-(scale,c) table +
        hardware gate. ``static_batch`` is the chunk batch the static path WOULD use (resolved by the caller,
        who threads the SAME value into the decode); ``None`` (direct callers / CPU / unknown) falls through
        to dynamic, the validated path."""
        if static_decode is not None:
            if static_decode:
                ok, reason = self.supports_static_decode()
                if not ok:
                    warnings.warn(f"static_decode=True requested but unsupported ({reason}); using dynamic path")
                    return False
                return True
            return False
        # None (deployed default): capability AND kill-switch AND the multi-chunk regime (batch < choices).
        ok, _ = self.supports_static_decode()
        if not (ok and _DEPLOYED_STATIC_ENABLED):
            return False
        if choices is None or static_batch is None:   # regime undeterminable -> dynamic (validated path)
            return False
        return int(static_batch) < int(choices)        # multi-chunk -> static

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "FlashANSRModel":
        """Instantiate a model from a config dict or a path to a config file.

        Parameters
        ----------
        config : dict[str, Any] or str
            A configuration mapping, or a path to a YAML config file. A top-level ``"model"``
            key is unwrapped if present.

        Returns
        -------
        FlashANSRModel
            The constructed model (with freshly initialised weights).

        Raises
        ------
        KeyError
            If the resolved config is missing any required constructor key.
        """
        config_ = load_config(config)

        if "model" in config_.keys():
            config_ = config_["model"]

        # Validate the required keys up front so a missing/renamed field yields one clear, actionable
        # error instead of an opaque bare KeyError from deep inside the constructor call below. Keys
        # read via .get(...) with a default (optional_condition, pre_encoder_bits, the *_norm_position
        # and xsa/rope-optional flags, ...) are intentionally NOT required here.
        required_keys = (
            "simplipy_engine", "tokenizer", "pre_encoder_noise_scale",
            "encoder_max_n_variables", "encoder_dim", "encoder_n_heads", "encoder_n_isab",
            "encoder_n_sab", "encoder_n_inducing_points", "encoder_n_seeds", "encoder_ffn_hidden_dim",
            "encoder_dropout", "encoder_attn_norm", "encoder_ffn_norm", "encoder_output_norm",
            "decoder_input_dim", "decoder_model_dim", "decoder_n_layers", "decoder_n_heads",
            "decoder_max_seq_len", "decoder_ffn_hidden_dim", "decoder_dropout",
            "decoder_block_self_attn_norm", "decoder_block_cross_attn_norm", "decoder_block_ffn_norm",
            "decoder_cross_attn_kv_norm", "decoder_output_norm", "decoder_use_rope_self_attn",
            "decoder_use_rope_cross_attn", "use_checkpointing",
        )
        missing_keys = [key for key in required_keys if key not in config_]
        if missing_keys:
            raise KeyError(
                f"FlashANSRModel config is missing required key(s): {missing_keys}. This usually means "
                f"the model config is incomplete or from an incompatible flash-ansr version. "
                f"Present keys: {sorted(config_.keys())}."
            )

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
            decoder_use_xsa_self_attn=config_.get("decoder_use_xsa_self_attn", False),

            decoder_block_norm_position=config_.get("decoder_block_norm_position", "pre"),
            encoder_block_norm_position=config_.get("encoder_block_norm_position", "pre"),
            pre_encoder_bits=config_.get("pre_encoder_bits", 32),

            use_checkpointing=config_["use_checkpointing"],

            optional_condition=config_.get("optional_condition", False),
            null_memory_init_seed=config_.get("null_memory_init_seed", 0),
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

    def forward(self, input_tokens: torch.Tensor, data: torch.Tensor | None, input_num: torch.Tensor | None = None, memory: torch.Tensor | None = None, data_attn_mask: torch.Tensor | None = None, condition_mask: torch.Tensor | None = None, past_key_values: list | None = None, use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, list]:
        """Compute next-token logits for ``input_tokens`` conditioned on the ``(X, y)`` support set.

        Encoder memory is taken from ``memory`` if given, otherwise built from ``data`` (or reused
        from a previous call). ``condition_mask`` enables classifier-free routing between the data
        memory and the learned null memory on optional-condition models.

        Parameters
        ----------
        input_tokens : torch.Tensor
            The decoder input token indices.
        data : torch.Tensor or None
            The ``(X, y)`` support set to encode into memory; may be ``None`` when ``memory`` is
            supplied or memory was set on a previous call.
        input_num : torch.Tensor, optional
            Numeric values aligned with ``input_tokens`` for the numeric embedding.
        memory : torch.Tensor, optional
            Precomputed encoder memory, bypassing re-encoding of ``data``.
        data_attn_mask : torch.Tensor, optional
            Attention mask over the support set when building memory.
        condition_mask : torch.Tensor, optional
            Per-example boolean mask selecting data memory (True) vs the learned null memory
            (False); only valid on optional-condition models.
        past_key_values : list, optional
            Cached decoder key/value states for incremental decoding.
        use_cache : bool, optional
            If True, also return updated key/value caches. Defaults to False.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, list]
            The next-token logits, or ``(logits, past_key_values)`` when ``use_cache`` is True.
        """
        if memory is not None:
            # Never register a passed nn.Parameter (e.g. an optional-condition model's `null_memory`)
            # as the 'memory' attribute: nn.Module.__setattr__ would move it into `_parameters` and the
            # next plain-tensor forward would raise. A detached view keeps the cross-attention math
            # identical (inference reads memory read-only; the train-time null routing uses torch.where
            # on `null_memory` directly, never this assignment).
            self.memory = memory.detach() if isinstance(memory, nn.Parameter) else memory
        elif data is not None:
            self.memory = self._create_memory(data, data_attn_mask)
        elif self.memory is None:
            raise ValueError("Either `data` or `memory` must be provided for the first forward pass.")

        # Optional-condition routing (classifier-free guidance): per example, keep the (X, y) encoder
        # memory where `condition_mask` is True and substitute the learned `null_memory` where it is
        # False (unconditioned). DATA-DRIVEN and applied in BOTH train and inference (no self.training
        # gate) -- this is the first-class optional condition, NOT a train-time dropout. `torch.where`
        # routes gradients to the encoder for conditioned rows and to `null_memory` for unconditioned
        # rows. `condition_mask=None` -> every example conditioned (backward-compatible default).
        if condition_mask is not None:
            if not self.optional_condition:
                raise ValueError(
                    "condition_mask was provided but this model has no `null_memory` "
                    "(set optional_condition=True in the model config to enable the unconditional mode)."
                )
            cm = condition_mask.to(device=self.memory.device, dtype=torch.bool).view(-1, 1, 1)
            self.memory = torch.where(cm, self.memory, self.null_memory.to(self.memory.dtype))

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
        decoder_output = self.decoder(tokens=input_tokens, encoder_memory=self.memory, extra_parallel_embeddings=numeric_embeddings, past_key_values=past_key_values, use_cache=use_cache)

        if use_cache:
            decoder_output, new_past_key_values = decoder_output
            logits = self.next_token_head(decoder_output)
            return logits, new_past_key_values

        logits = self.next_token_head(decoder_output)

        # Removed numeric head as it is not present in the new Decoder structure
        return logits

    def forward_static(
        self,
        input_tokens: torch.Tensor,
        input_num: torch.Tensor | None,
        memory: torch.Tensor,
        static_cache: StaticKVCache,
        position: int,
    ) -> torch.Tensor:
        """Static-shape (graph-capturable) single-token decode step. Mirrors ``forward`` (numeric
        embedding routing identical) but the decoder writes K/V into ``static_cache`` at ``position``
        and reads the full buffer under a causal mask, instead of the dynamic cat-grow path. Returns
        logits only (the cache is mutated in place). v23.0 decoder path only (see static_kv.py)."""
        self.memory = memory
        if input_num is not None:
            input_num_pre_encodings = self.pre_encoder_numeric_tokens(input_num)
            input_num_pre_encodings[torch.isnan(input_num_pre_encodings)] = 0
            numeric_embeddings = self.numeric_embedding(input_num_pre_encodings)
            if numeric_embeddings.dim() == 4 and numeric_embeddings.size(-2) == 1:
                numeric_embeddings = numeric_embeddings.squeeze(-2)
        else:
            numeric_embeddings = None

        decoder_output = self.decoder.forward_static(input_tokens, self.memory, numeric_embeddings, static_cache, position)
        return self.next_token_head(decoder_output)

    def _resolve_generation_prefix(
        self,
        *,
        prompt_prefix: PromptPrefix | None,
        initial_tokens: list[int] | None = None,
        input_num: list[float] | None = None,
    ) -> tuple[list[int], list[float] | None]:
        if prompt_prefix is not None:
            tokens = [int(token) for token in prompt_prefix.tokens]
            numeric_values = [float(value) for value in prompt_prefix.numeric]
            return tokens, numeric_values

        if initial_tokens is not None:
            tokens = list(initial_tokens)
            numeric = [float(value) for value in input_num] if input_num is not None else None
            return tokens, numeric

        if self.preprocessor is None:
            return [self.tokenizer['<bos>']], None

        serialized = self.preprocessor.serialize_prompt_prefix()

        tokens = [int(token) for token in serialized['input_ids']]
        numeric_values = [float(value) for value in serialized['input_num']]

        return tokens, numeric_values

    def beam_search(
        self,
        data: torch.Tensor,
        beam_width: int = 4,
        max_len: int = 100,
        batch_size: int = 128,
        unique: bool = True,
        verbose: bool = False,
        limit_expansions: bool = True,
        use_cache: bool = False,
        *,
        prompt_prefix: PromptPrefix | None = None,
        initial_tokens: list[int] | None = None,
        input_num: list[float] | None = None,
    ) -> tuple[list[list[int]], list[float], list[bool]]:
        """Decode candidate expressions from ``data`` with beam search.

        Parameters
        ----------
        data : torch.Tensor
            The encoded ``(X, y)`` support set conditioning the decoder.
        beam_width : int, optional
            Number of beams kept at each expansion step. Defaults to 4.
        max_len : int, optional
            Maximum decoded sequence length (including the prompt prefix). Defaults to 100.
        batch_size : int, optional
            Number of beams scored per forward pass. Defaults to 128.
        unique : bool, optional
            Deduplicate completed sequences by their constantified skeleton. Defaults to True.
        verbose : bool, optional
            Show a progress bar over decoding steps. Defaults to False.
        limit_expansions : bool, optional
            Restrict expansions to grammar-valid continuations. Defaults to True.
        use_cache : bool, optional
            Reuse decoder key/value caches across steps. Defaults to False.
        prompt_prefix : PromptPrefix, optional
            Prompt prefix (e.g. feature hints) prepended before decoding.
        initial_tokens : list[int], optional
            Explicit initial token prefix to start decoding from.
        input_num : list[float], optional
            Numeric values aligned with ``initial_tokens`` for the numeric embedding.

        Returns
        -------
        tuple[list[list[int]], list[float], list[bool]]
            The decoded token sequences, their log-probabilities, and per-sequence completion
            flags. A flag is ``True`` for sequences that terminated with ``<eos>`` and ``False``
            for active-beam fallbacks appended when fewer than ``beam_width`` completed sequences
            were found.
        """
        device = data.device

        base_tokens, base_input_num = self._resolve_generation_prefix(
            prompt_prefix=prompt_prefix,
            initial_tokens=initial_tokens,
            input_num=input_num,
        )

        if isinstance(base_input_num, torch.Tensor):
            base_input_num = base_input_num.tolist()

        prefix_length = len(base_tokens)
        if prefix_length >= max_len:
            raise ValueError(f"Initial token prefix length ({prefix_length}) exceeds max_len ({max_len}).")

        memory = self._create_memory(data)

        eos_token_id = self.tokenizer['<eos>']
        pad_token_id = self.tokenizer['<pad>']

        pbar = tqdm(total=max_len - prefix_length, disable=not verbose, desc=f"Generating beams (max length: {max_len})", smoothing=0.0)

        completed_sequences_heap: list[tuple[float, tuple[int, ...]]] = []
        completed_sequences_scores: dict[tuple[int, ...], float] = {}
        simplify_cache: dict[tuple[int, ...], tuple[int, ...]] = {}
        n_pruned = 0

        numeric_template: torch.Tensor | None = None
        if base_input_num is not None:
            numeric_template = torch.full((max_len,), float('nan'), device=device, dtype=torch.float32)
            numeric_template[:len(base_input_num)] = torch.tensor(base_input_num, device=device, dtype=torch.float32)

        def build_input_num_tensor(current_length: int, batch_size: int) -> torch.Tensor | None:
            if numeric_template is None:
                return None
            return numeric_template[:current_length].unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)

        sequences = torch.full((beam_width, max_len), pad_token_id, device=device, dtype=torch.long)
        if prefix_length:
            sequences[:, :prefix_length] = torch.tensor(base_tokens, device=device, dtype=torch.long)

        lengths = torch.full((beam_width,), prefix_length, device=device, dtype=torch.long)
        scores = torch.full((beam_width,), float('-inf'), device=device, dtype=torch.float)
        scores[0] = 0.0
        finished = torch.zeros(beam_width, dtype=torch.bool, device=device)

        if prefix_length and base_tokens[-1] == eos_token_id:
            finished[0] = True

        def register_completed_sequence(seq_tuple: tuple[int, ...], score: float) -> None:
            nonlocal n_pruned

            existing_score = completed_sequences_scores.get(seq_tuple)
            if existing_score is not None and score <= existing_score:
                n_pruned += 1
                return

            completed_sequences_scores[seq_tuple] = score
            heapq.heappush(completed_sequences_heap, (score, seq_tuple))

            while len(completed_sequences_scores) > beam_width:
                prune_score, prune_key = heapq.heappop(completed_sequences_heap)
                current_score = completed_sequences_scores.get(prune_key)
                if current_score is None:
                    continue
                if current_score != prune_score:
                    continue
                del completed_sequences_scores[prune_key]
                n_pruned += 1
                break

        with torch.no_grad():
            kv_cache: list | None = None

            for current_length in range(prefix_length, max_len):
                active_mask = (~finished) & torch.isfinite(scores)
                if not torch.any(active_mask):
                    break

                active_indices = active_mask.nonzero(as_tuple=True)[0]

                candidate_scores_list: list[torch.Tensor] = []
                candidate_parents: list[torch.Tensor] = []
                candidate_tokens: list[torch.Tensor] = []
                step_kv_parts: list[list] = []

                for start_idx in range(0, active_indices.numel(), batch_size):
                    batch_indices = active_indices[start_idx:start_idx + batch_size]

                    if use_cache and kv_cache is not None:
                        # Incremental: only the last token
                        input_ids_tensor = sequences[batch_indices, current_length - 1:current_length]
                        # Pass numeric embedding for the new token position to preserve bias contribution
                        if numeric_template is not None:
                            input_num_tensor = numeric_template[current_length - 1:current_length].unsqueeze(0).expand(len(batch_indices), -1).unsqueeze(-1)
                        else:
                            input_num_tensor = None
                        batch_past = [
                            (
                                (lc[0][0][batch_indices], lc[0][1][batch_indices]),
                                (lc[1][0][batch_indices], lc[1][1][batch_indices]),
                            )
                            for lc in kv_cache
                        ]
                    else:
                        input_ids_tensor = sequences[batch_indices, :current_length]
                        input_num_tensor = build_input_num_tensor(current_length, len(batch_indices))
                        batch_past = None

                    result = self.forward(input_ids_tensor, None, input_num=input_num_tensor, memory=memory, past_key_values=batch_past, use_cache=use_cache)

                    if use_cache:
                        logits, new_past = result
                        step_kv_parts.append(new_past)
                    else:
                        logits = result

                    next_token_log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

                    vocab_size = next_token_log_probs.size(-1)
                    if limit_expansions:
                        expansion_factor = 2 if unique else 1
                        expansion_per_beam = max(1, min(vocab_size, beam_width * expansion_factor))
                        top_log_probs, top_token_ids = torch.topk(next_token_log_probs, k=expansion_per_beam, dim=-1)
                    else:
                        expansion_per_beam = vocab_size
                        top_log_probs = next_token_log_probs
                        top_token_ids = torch.arange(vocab_size, device=next_token_log_probs.device, dtype=torch.long).unsqueeze(0).expand(next_token_log_probs.size(0), -1)

                    candidate_scores_list.append(scores[batch_indices].unsqueeze(1) + top_log_probs)
                    candidate_parents.append(batch_indices.repeat_interleave(expansion_per_beam))
                    candidate_tokens.append(top_token_ids.reshape(-1))

                # Phase 3: declare before the block so it is always bound;
                # assigned in each branch without a repeated type annotation.
                active_to_pos: dict[int, int] = {}

                # Assemble step-level KV cache from mini-batch parts (indexed 0..n_active-1)
                if use_cache and step_kv_parts:
                    if len(step_kv_parts) == 1:
                        step_kv_cache = step_kv_parts[0]
                    else:
                        n_layers = len(step_kv_parts[0])
                        step_kv_cache = [
                            (
                                (
                                    torch.cat([p[li][0][0] for p in step_kv_parts], dim=0),
                                    torch.cat([p[li][0][1] for p in step_kv_parts], dim=0),
                                ),
                                (
                                    torch.cat([p[li][1][0] for p in step_kv_parts], dim=0),
                                    torch.cat([p[li][1][1] for p in step_kv_parts], dim=0),
                                ),
                            )
                            for li in range(n_layers)
                        ]
                    # Build mapping: beam_idx -> position in step_kv_cache
                    active_to_pos = {int(idx): pos for pos, idx in enumerate(active_indices)}

                flat_scores = torch.cat(candidate_scores_list).reshape(-1)
                flat_parents = torch.cat(candidate_parents)
                flat_tokens = torch.cat(candidate_tokens)

                sorted_scores, sorted_indices = torch.sort(flat_scores, descending=True)

                # Phase 2: single bulk GPU→CPU transfer; avoids O(N) individual syncs
                sorted_parents_cpu: list[int] = flat_parents[sorted_indices].tolist()
                sorted_tokens_cpu: list[int] = flat_tokens[sorted_indices].tolist()
                sorted_scores_cpu: list[float] = sorted_scores.tolist()

                next_sequences = torch.full_like(sequences, pad_token_id)
                next_lengths = torch.zeros_like(lengths)
                next_scores = torch.full_like(scores, float('-inf'))
                next_finished = torch.zeros_like(finished)

                next_beam_set: set[tuple[int, ...]] = set()
                next_count = 0
                parent_cache_indices: list[int] = []

                for rank_idx in range(len(sorted_scores_cpu)):
                    parent_idx = sorted_parents_cpu[rank_idx]
                    token_id = sorted_tokens_cpu[rank_idx]
                    new_score = sorted_scores_cpu[rank_idx]

                    if token_id == eos_token_id:
                        new_seq = sequences[parent_idx, :current_length].tolist() + [token_id]
                        if unique:
                            try:
                                candidate_expression, before, after = self.tokenizer.extract_expression_from_beam(new_seq)
                            except ValueError:
                                n_pruned += 1
                                continue
                            else:
                                expr_key = tuple(candidate_expression)

                                tentative_simplified_tuple = simplify_cache.get(expr_key)
                                if tentative_simplified_tuple is None:
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
                                    simplified_tuple = tentative_simplified_tuple
                        else:
                            simplified_tuple = tuple(new_seq)

                        register_completed_sequence(simplified_tuple, new_score)
                        continue

                    # Phase 1: if the beam is already full, this non-EOS candidate has
                    # nowhere to go.  EOS candidates are handled above and `continue` before
                    # reaching here, so we only land here for non-EOS tokens.
                    # Keep the loop running so lower-ranked EOS candidates are still processed;
                    # bail out early only when the score has dropped below the worst score in
                    # the (already-full) completed pool — no future candidate can improve it.
                    if next_count >= beam_width:
                        if (completed_sequences_heap
                                and len(completed_sequences_scores) >= beam_width
                                and new_score < completed_sequences_heap[0][0]):
                            break
                        continue

                    if unique:
                        seq_tuple = (*sequences[parent_idx, :current_length].tolist(), token_id)
                        if seq_tuple in next_beam_set:
                            n_pruned += 1
                            continue
                        next_beam_set.add(seq_tuple)

                    # Direct tensor copy: avoids Python list → torch.tensor round-trip
                    next_sequences[next_count, :current_length] = sequences[parent_idx, :current_length]
                    next_sequences[next_count, current_length] = token_id
                    next_lengths[next_count] = current_length + 1
                    next_scores[next_count] = new_score
                    next_finished[next_count] = False
                    if use_cache:
                        parent_cache_indices.append(active_to_pos[parent_idx])
                    next_count += 1

                # Reindex KV-cache for the surviving beams
                if use_cache and step_kv_parts and parent_cache_indices:
                    reindex = torch.tensor(parent_cache_indices, device=device, dtype=torch.long)
                    # Pad to beam_width (unused slots get index 0, harmless)
                    if len(reindex) < beam_width:
                        reindex = torch.cat([reindex, reindex.new_zeros(beam_width - len(reindex))])
                    kv_cache = [
                        (
                            (lc[0][0][reindex].clone(), lc[0][1][reindex].clone()),
                            (lc[1][0][reindex].clone(), lc[1][1][reindex].clone()),
                        )
                        for lc in step_kv_cache
                    ]

                if next_count == 0 and completed_sequences_scores:
                    break

                sequences = next_sequences
                lengths = next_lengths
                scores = next_scores
                finished = next_finished

                pbar.set_postfix({'completed': len(completed_sequences_scores), 'pruned': n_pruned})
                pbar.update(1)
            pbar.close()

        # Sequences from completed_sequences_scores are EOS-terminated; active-beam
        # fallbacks are not.  Track the flag separately so the return value is accurate.
        combined_sequences: list[tuple[list[int], float, bool]] = [
            (list(seq_tuple), score, True) for seq_tuple, score in completed_sequences_scores.items()
        ]

        # Only include active (non-EOS) beams as a true last resort. Active beams have
        # no EOS log-probability penalty so their scores are systematically higher than
        # completed sequences. Including them unconditionally causes them to displace
        # EOS-terminated sequences in the final sort, returning truncated beams that
        # lack </expression> and crash downstream.
        if len(combined_sequences) < beam_width:
            for beam_idx in range(beam_width):
                if torch.isfinite(scores[beam_idx]):
                    seq_len = int(lengths[beam_idx].item())
                    if seq_len == 0:
                        continue
                    seq = sequences[beam_idx, :seq_len].tolist()
                    combined_sequences.append((seq, float(scores[beam_idx].item()), False))

        # Look up expression delimiters once; may be None if not in the vocabulary.
        expr_start_token_id = self.tokenizer.token2idx.get('<expression>')
        expr_end_token_id = self.tokenizer.token2idx.get('</expression>')

        combined_sequences_final: list[tuple[list[int], float, bool]] = []
        for seq, score, is_complete in combined_sequences:
            # Repair sequences that opened <expression> but never closed it.
            # Two sub-cases:
            #   - EOS-terminated (unique=False degenerate): [bos <expression> X <eos>]
            #     → insert </expression> before <eos> so EOS remains the final token.
            #   - Active beam truncated at max_len: [bos <expression> X Y Z]
            #     → append </expression> at the end.
            # Sequences that never opened an expression cannot be parsed and are dropped.
            if expr_end_token_id is not None and expr_end_token_id not in seq:
                if expr_start_token_id is None or expr_start_token_id not in seq:
                    continue
                if eos_token_id in seq:
                    eos_pos = seq.index(eos_token_id)
                    seq = seq[:eos_pos] + [expr_end_token_id] + seq[eos_pos:]
                else:
                    seq = seq + [expr_end_token_id]
            constantified_seq = self.tokenizer.constantify_expression(seq)
            combined_sequences_final.append((constantified_seq, score, is_complete))

        combined_sequences_final = sorted(combined_sequences_final, key=lambda x: x[1], reverse=True)
        top = combined_sequences_final[:beam_width]

        return [seq for seq, _, _ in top], [score for _, score, _ in top], [flag for _, _, flag in top]

    def mcts_decode(
        self,
        data: torch.Tensor,
        config: MCTSConfig,
        beam_width: int = 16,
        value_fn: Optional[ValueFunction] = None,
        terminal_fn: Optional[TerminalFunction] = None,
        invalid_sequence_fn: Optional[Callable[[Tuple[int, ...]], bool]] = None,
        completion_sort: str = "reward",
        verbose: bool = False,
        *,
        prompt_prefix: PromptPrefix | None = None,
        initial_tokens: list[int] | None = None,
        input_num: list[float] | None = None,
    ) -> tuple[list[list[int]], list[float], list[bool], list[float]]:
        """Decode expressions using Monte Carlo Tree Search."""

        device = data.device

        base_tokens, base_input_num = self._resolve_generation_prefix(
            prompt_prefix=prompt_prefix,
            initial_tokens=initial_tokens,
            input_num=input_num,
        )

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
                assert isinstance(logits, torch.Tensor)
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

        return sequences, log_probs, completed_flags, rewards

    def sample_top_kp(
        self,
        data: torch.Tensor,
        choices: int = 10,
        top_k: int = 0,
        top_p: float = 1,
        max_len: int = 100,
        batch_size: int = 128,
        temperature: float = 1.0,
        valid_only: bool = True,
        simplify: bool | str = True,
        unique: bool = True,
        verbose: bool = False,
        use_cache: bool = False,
        return_raw: bool = False,
        static_decode: bool | None = None,
        *,
        prompt_prefix: PromptPrefix | None = None,
        initial_tokens: list[int] | None = None,
        input_num: list[float] | None = None,
        memory: torch.Tensor | None = None,
        guidance_weight: float | None = None,
    ) -> tuple[list[list[int]], list[float], list[bool]] | tuple[list[list[int]], list[float]]:
        """Decode candidate expressions from ``data`` by top-k / top-p (nucleus) sampling.

        Draws ``choices`` samples per problem, optionally routing through the static-shape decode
        path and classifier-free guidance. Post-processing extracts, simplifies, deduplicates and
        sorts the sampled sequences unless ``return_raw`` short-circuits it.

        Parameters
        ----------
        data : torch.Tensor
            The encoded ``(X, y)`` support set conditioning the decoder.
        choices : int, optional
            Number of independent samples to draw per problem. Defaults to 10.
        top_k : int, optional
            Restrict sampling to the ``top_k`` most likely tokens (0 disables). Defaults to 0.
        top_p : float, optional
            Nucleus threshold: sample from the smallest set of tokens whose cumulative
            probability exceeds ``top_p``. Defaults to 1.
        max_len : int, optional
            Maximum decoded sequence length. Defaults to 100.
        batch_size : int, optional
            Number of samples decoded per forward pass. Defaults to 128.
        temperature : float, optional
            Softmax temperature applied to the logits before sampling. Defaults to 1.0.
        valid_only : bool, optional
            Keep only grammar-valid completed sequences. Defaults to True.
        simplify : bool or str, optional
            Simplify decoded expressions during post-processing. Defaults to True.
        unique : bool, optional
            Deduplicate by constantified skeleton. Defaults to True.
        verbose : bool, optional
            Show a progress bar over decoding steps. Defaults to False.
        use_cache : bool, optional
            Reuse decoder key/value caches across steps. Defaults to False.
        return_raw : bool, optional
            Return the raw ``(sequences, scores)`` before post-processing. Defaults to False.
        static_decode : bool or None, optional
            Tri-state selector for the static-shape decode path. ``None`` uses the deployed
            default for capable models; ``True``/``False`` force it on/off (an unsupported
            ``True`` warns and falls back to dynamic). Defaults to ``None``.
        prompt_prefix : PromptPrefix, optional
            Prompt prefix prepended before decoding.
        initial_tokens : list[int], optional
            Explicit initial token prefix to start decoding from.
        input_num : list[float], optional
            Numeric values aligned with ``initial_tokens`` for the numeric embedding.
        memory : torch.Tensor, optional
            Precomputed encoder memory, bypassing re-encoding of ``data``.
        guidance_weight : float, optional
            Classifier-free guidance weight (optional-condition models only): ``1`` is a pure
            conditioned decode, ``0`` a pure prior decode, ``>1`` amplifies the data signal.

        Returns
        -------
        tuple
            When ``return_raw`` is False, ``(sequences, scores, is_valid)`` after
            post-processing; when ``return_raw`` is True, the raw ``(sequences, scores)``.
        """

        # Static-shape, position-indexed-KV decode (Stage 1b): route to the chunk-major static loop.
        # `static_decode` is tri-state: None -> the deployed default for capable models, True/False explicit
        # (resolved + capability-gated by `_resolve_static_decode`; unsupported True -> warn + dynamic).
        # Direct callers needing the dynamic path's activation hooks (recorder/interventions) pass False.
        # The dynamic path below is byte-unchanged when the resolved value is False.
        static_decode = self._resolve_static_decode(static_decode)

        # Classifier-free guidance (optional-condition models only). The guidance weight w combines the
        # conditioned next-token logits (real (X, y) encoder memory) with the unconditioned logits (the
        # learned `null_memory`):  guided = uncond + w * (cond - uncond).
        #   w == 1 -> pure conditioned decode (byte-identical to the plain path; the cond fast-path);
        #   w == 0 -> pure unconditional / prior decode (substitute null_memory; the null fast-path);
        #   w  > 1 -> amplify the data signal; 0 < w < 1 -> interpolate toward the prior.
        # The general (w not in {0, 1}) path runs a SECOND forward per step and is not KV-cache / static
        # compatible yet, so it forces the dynamic, cache-free loop. The w in {0, 1} fast-paths keep the
        # static / cache path and give the exact correctness floors.
        guided = False
        uncond_memory: torch.Tensor | None = None
        if guidance_weight is not None:
            if not self.optional_condition:
                raise ValueError(
                    "guidance_weight was provided but this model has no `null_memory` "
                    "(set optional_condition=True in the model config to enable guided decoding)."
                )
            guidance_weight = float(guidance_weight)
            # `.detach()` is load-bearing: `null_memory` is an nn.Parameter and `.to()` returns it
            # unchanged when already on the target device, so assigning it to `self.memory` inside
            # `forward` would register 'memory' as a Parameter and poison the next plain-tensor forward.
            if guidance_weight == 0.0:
                # Pure unconditional / prior decode: substitute the learned null_memory, decode plainly.
                memory = self.null_memory.detach().to(device=data.device)
            elif guidance_weight == 1.0:
                pass  # Pure conditioned decode == the plain path; nothing to substitute.
            else:
                guided = True
                static_decode = False  # the two-forward combine is dynamic + cache-free
                use_cache = False
                uncond_memory = self.null_memory.detach().to(device=data.device)

        if static_decode:
            return self._sample_top_kp_static(
                data, choices=choices, top_k=top_k, top_p=top_p, max_len=max_len,
                batch_size=batch_size, temperature=temperature, valid_only=valid_only,
                simplify=simplify, unique=unique, verbose=verbose, return_raw=return_raw,
                prompt_prefix=prompt_prefix, initial_tokens=initial_tokens,
                input_num=input_num, memory=memory)

        device = data.device

        # --- 1. Vectorized Initialization ---
        base_tokens, base_input_num = self._resolve_generation_prefix(
            prompt_prefix=prompt_prefix,
            initial_tokens=initial_tokens,
            input_num=input_num,
        )

        prefix_length = len(base_tokens)
        if prefix_length > max_len:
            raise ValueError(f"Initial token prefix length ({prefix_length}) exceeds max_len ({max_len}).")

        # Pre-allocate tensors on the target device
        sequences = torch.full((choices, max_len), self.tokenizer['<pad>'], device=device, dtype=torch.long)
        if prefix_length > 0:
            prefix_tensor = torch.tensor(base_tokens, device=device, dtype=torch.long)
            sequences[:, :prefix_length] = prefix_tensor

        scores = torch.zeros(choices, device=device, dtype=torch.float)
        is_finished = torch.zeros(choices, device=device, dtype=torch.bool)

        eos_token = self.tokenizer['<eos>']
        if prefix_length > 0 and base_tokens[-1] == eos_token:
            is_finished[:] = True

        if memory is None:
            memory = self._create_memory(data)

        # Pre-allocate numeric template once (mirrors beam_search optimisation)
        numeric_template: torch.Tensor | None = None
        if base_input_num is not None:
            numeric_template = torch.full((max_len,), float('nan'), device=device, dtype=torch.float32)
            numeric_template[:len(base_input_num)] = torch.tensor(base_input_num, device=device, dtype=torch.float32)

        def build_input_num_tensor(current_length: int, batch_size: int) -> torch.Tensor | None:
            if numeric_template is None:
                return None
            return numeric_template[:current_length].unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)

        # --- 2. Vectorized Generation Loop with Mini-batching ---
        # KV-cache state: list of per-layer caches, each holding tensors of shape
        # (choices, n_heads, cached_len, head_dim).  Indexed by batch_indices per mini-batch.
        kv_cache: list | None = None

        with torch.no_grad():
            total_steps = max(0, max_len - prefix_length)
            pbar = tqdm(total=total_steps, disable=not verbose, desc="Generating tokens", smoothing=0.0)
            for current_length in range(prefix_length, max_len):
                if is_finished.all():
                    break

                active_indices = (~is_finished).nonzero(as_tuple=True)[0]
                if active_indices.numel() == 0:
                    break

                step_kv_parts: list[tuple[torch.Tensor, list]] = []

                for start_idx in range(0, len(active_indices), batch_size):
                    batch_indices = active_indices[start_idx: start_idx + batch_size]

                    if use_cache and kv_cache is not None:
                        # Incremental: only feed the last token
                        input_ids_tensor = sequences[batch_indices, current_length - 1: current_length]
                        # Pass numeric embedding for the new token position to preserve bias contribution
                        if numeric_template is not None:
                            input_num_tensor = numeric_template[current_length - 1:current_length].unsqueeze(0).expand(len(batch_indices), -1).unsqueeze(-1)
                        else:
                            input_num_tensor = None
                        batch_past = [
                            (
                                (layer_cache[0][0][batch_indices], layer_cache[0][1][batch_indices]),
                                (layer_cache[1][0][batch_indices], layer_cache[1][1][batch_indices]),
                            )
                            for layer_cache in kv_cache
                        ]
                    else:
                        input_ids_tensor = sequences[batch_indices, :current_length]
                        input_num_tensor = build_input_num_tensor(current_length, len(batch_indices))
                        batch_past = None

                    result = self.forward(input_ids_tensor, None, input_num=input_num_tensor, memory=memory, past_key_values=batch_past, use_cache=use_cache)

                    if use_cache:
                        logits, new_past = result
                        step_kv_parts.append((batch_indices, new_past))
                    else:
                        logits = result

                    next_token_logits = logits[:, -1, :]

                    # Recorded candidate score = the CONDITIONED model log-likelihood, captured BEFORE
                    # guidance / top_k / temperature / top_p. The sampler invariant is that `original_scores`
                    # excludes every sampling-shaping transform; CFG is one such transform (it steers WHICH
                    # tokens are drawn), so it must not contaminate the persisted log-prob that feeds candidate
                    # ranking + the save-all-raw pipeline -- and for w > 1 the guided logits are not a
                    # normalized likelihood anyway.
                    original_scores = torch.log_softmax(next_token_logits, dim=-1)

                    if guided:
                        # Second (unconditional) forward on the SAME partial sequences, with null_memory,
                        # combined BEFORE the top_k/top_p/temperature shaping. use_cache is False here, so
                        # both forwards recompute the full prefix and stay aligned step-for-step. This shapes
                        # only the SAMPLING distribution; `original_scores` above stays the conditioned score.
                        logits_uncond = self.forward(input_ids_tensor, None, input_num=input_num_tensor, memory=uncond_memory, use_cache=False)
                        assert isinstance(logits_uncond, torch.Tensor)  # use_cache=False -> Tensor (narrow the union for mypy)
                        next_token_uncond = logits_uncond[:, -1, :]
                        next_token_logits = next_token_uncond + guidance_weight * (next_token_logits - next_token_uncond)

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

                # Assemble updated KV-cache from mini-batch parts
                if use_cache and step_kv_parts:
                    n_layers = len(step_kv_parts[0][1])
                    new_kv_cache = []
                    for li in range(n_layers):
                        if kv_cache is None:
                            # Prefill: concatenate all parts in sample order
                            sa_k = torch.cat([p[1][li][0][0] for p in step_kv_parts], dim=0)
                            sa_v = torch.cat([p[1][li][0][1] for p in step_kv_parts], dim=0)
                            ca_k = torch.cat([p[1][li][1][0] for p in step_kv_parts], dim=0)
                            ca_v = torch.cat([p[1][li][1][1] for p in step_kv_parts], dim=0)
                        else:
                            # Incremental: self-attn cache grows by 1 token, cross-attn is static
                            old_sa_k, old_sa_v = kv_cache[li][0]
                            ca_k, ca_v = kv_cache[li][1]
                            # Pad self-attention cache by 1 position for all samples
                            pad_shape = (old_sa_k.shape[0], old_sa_k.shape[1], 1, old_sa_k.shape[3])
                            sa_k = torch.cat([old_sa_k, old_sa_k.new_zeros(pad_shape)], dim=2)
                            sa_v = torch.cat([old_sa_v, old_sa_v.new_zeros(pad_shape)], dim=2)
                            # Scatter active results into the padded cache
                            for batch_idx, new_past in step_kv_parts:
                                sa_k[batch_idx] = new_past[li][0][0]
                                sa_v[batch_idx] = new_past[li][0][1]
                        new_kv_cache.append(((sa_k, sa_v), (ca_k, ca_v)))
                    kv_cache = new_kv_cache

                pbar.update(1)
            pbar.close()

        completed_sequences = sequences.cpu().tolist()
        completed_scores = scores.cpu().tolist()

        if return_raw:
            # Raw exit: the GPU sampling output BEFORE post-processing (extract/simplify/dedup/sort).
            # FlashANSR.generate's chunked path uses this to run ONE parallel simplify pass over all
            # chunks' candidates, then calls _postprocess_sampled with a precomputed simplify_map.
            return completed_sequences, completed_scores

        return self._postprocess_sampled(
            completed_sequences, completed_scores,
            simplify=simplify, unique=unique, valid_only=valid_only,
            eos_token=eos_token, verbose=verbose,
        )

    def _sample_top_kp_static(
        self,
        data: torch.Tensor,
        choices: int = 10,
        top_k: int = 0,
        top_p: float = 1,
        max_len: int = 100,
        batch_size: int = 128,
        temperature: float = 1.0,
        valid_only: bool = True,
        simplify: bool | str = True,
        unique: bool = True,
        verbose: bool = False,
        return_raw: bool = False,
        early_stop_every: int = 1,
        *,
        prompt_prefix: PromptPrefix | None = None,
        initial_tokens: list[int] | None = None,
        input_num: list[float] | None = None,
        memory: torch.Tensor | None = None,
    ) -> tuple[list[list[int]], list[float], list[bool]] | tuple[list[list[int]], list[float]]:
        """Static-shape, position-indexed-KV variant of ``sample_top_kp`` (Stage 1b).

        CHUNK-major (vs the dynamic path's step-major): each chunk of ``batch_size`` candidates runs the
        full decode at FIXED width with a per-chunk ``StaticKVCache`` -- in-place K/V writes at
        ``position`` (no ``torch.cat`` grow, no active-row gather). The per-step forward is
        ``forward_static``; sampling is eager on the ACTIVE subset (finished rows kept full-width but
        never sampled/overwritten -- avoids the finished-row-corruption footgun). Prefill stays dynamic,
        then ``seed_from_dynamic``.

        Quality bar = the ``gate_overlap`` pinned-seed FLOOR, NOT bit-identity: full-width + active-subset
        sampling shifts the RNG trajectory vs serial's shrinking batch (greedy/top_k=1 IS bit-identical --
        argmax of per-row-independent logits -- which is the cheapest loop-wiring falsifier). Deployed
        v23.0 path only: pre-norm, RoPE-self (XSA supported; cross-RoPE still asserted off).

        ``early_stop_every``: check ``all-finished`` every K steps (K=1 mirrors the dynamic path's
        per-step sync for an apples-to-apples Stage-1b measurement; Stage 2 bumps it for the graph).
        """
        global STATIC_DECODE_CALL_COUNT
        STATIC_DECODE_CALL_COUNT += 1  # engagement proof for gates (see module-level definition)
        device = data.device

        block0 = self.decoder.layers[0]

        # --- 1. init (mirrors sample_top_kp) ---
        base_tokens, base_input_num = self._resolve_generation_prefix(
            prompt_prefix=prompt_prefix, initial_tokens=initial_tokens, input_num=input_num)
        prefix_length = len(base_tokens)
        if prefix_length > max_len:
            raise ValueError(f"Initial token prefix length ({prefix_length}) exceeds max_len ({max_len}).")
        if prefix_length == 0:
            raise ValueError("static decode requires a non-empty prefix (dynamic prefill seeds the cache).")

        sequences = torch.full((choices, max_len), self.tokenizer['<pad>'], device=device, dtype=torch.long)
        sequences[:, :prefix_length] = torch.tensor(base_tokens, device=device, dtype=torch.long)
        scores = torch.zeros(choices, device=device, dtype=torch.float)
        is_finished = torch.zeros(choices, device=device, dtype=torch.bool)

        eos_token = self.tokenizer['<eos>']
        if base_tokens[-1] == eos_token:
            is_finished[:] = True

        if memory is None:
            memory = self._create_memory(data)
        if memory.shape[0] not in (1, choices):
            raise ValueError(f"memory batch dim {memory.shape[0]} must be 1 (broadcast) or choices ({choices}).")

        numeric_template: torch.Tensor | None = None
        if base_input_num is not None:
            numeric_template = torch.full((max_len,), float('nan'), device=device, dtype=torch.float32)
            numeric_template[:len(base_input_num)] = torch.tensor(base_input_num, device=device, dtype=torch.float32)

        n_heads = block0.self_attention.n_heads
        head_dim = block0.self_attention.head_dim
        n_layers = len(self.decoder.layers)

        # Runtime VRAM spill-guard (WSL2/unified memory has NO clean OOM: an over-budget batch spills to
        # system RAM and runs ~28x slower -- measured). The auto chunk size is picked conservatively, so
        # this is a BACKSTOP for an explicit oversized batch_size: after the FIRST chunk, if this decode's
        # live working set climbs past _VRAM_GUARD_FRACTION of physical VRAM, raise loudly rather than
        # crawl. Use max_memory_ALLOCATED (live tensors), NOT reserved: the caching-allocator RESERVED pool
        # is a process-global high-water that a PRIOR run (e.g. a wide dynamic decode) can leave inflated
        # and never release, which would false-trip the guard on an innocent later static call. reset the
        # allocated peak here so it reflects THIS decode (resident + the static transient).
        _guard = None
        if device.type == 'cuda':
            try:
                # Budget = what this process can still allocate without spilling = physical FREE + its own
                # REUSABLE held pool (reserved - allocated). Using free alone would under-budget on a card
                # whose allocator pool already grew; using total would over-budget on a SHARED card.
                _free0 = torch.cuda.mem_get_info(device)[0]
                _alloc0 = torch.cuda.memory_allocated(device)
                _avail = _free0 + (torch.cuda.memory_reserved(device) - _alloc0)
                # Pre-flight: reject a grossly oversized FIRST chunk BEFORE running it. Use the ACTUAL first
                # chunk width min(batch_size, choices) (the single-shot arm passes batch_size>=choices but
                # runs only `choices` rows). Pad per-row with _GUARD_OVERHEAD (LARGER than the cap's 1.6 ->
                # an independent backstop). decoder_max_seq_len + fp32 (4 B): the deployed path is fp32; under
                # half/autocast this over-projects (safe). The auto cap (0.7 of avail) stays well under this.
                _chunk0 = min(int(batch_size), int(choices))
                _per_row = 2 * n_layers * n_heads * head_dim * self.decoder_max_seq_len * 4 * _GUARD_OVERHEAD
                if _chunk0 * _per_row > _VRAM_GUARD_FRACTION * _avail:
                    raise RuntimeError(
                        f"static decode batch={_chunk0} projected {_chunk0 * _per_row / 1024**3:.1f} GB > "
                        f"{_VRAM_GUARD_FRACTION:.0%} of {_avail / 1024**3:.1f} GB available VRAM; reduce "
                        f"batch_size or set static_decode=False (WSL2 would spill to system RAM, ~28x slower).")
                torch.cuda.reset_peak_memory_stats(device)
                _guard = (_avail, _alloc0)
            except RuntimeError:
                raise
            except Exception:
                _guard = None

        # --- 2. chunk-major static generation loop ---
        with torch.no_grad():
            pbar = tqdm(total=choices, disable=not verbose, desc="Generating tokens (static)", smoothing=0.0)
            for chunk_start in range(0, choices, batch_size):
                chunk_stop = min(chunk_start + batch_size, choices)
                rows = slice(chunk_start, chunk_stop)
                bsz = chunk_stop - chunk_start
                mem_chunk = memory if memory.shape[0] == 1 else memory[rows]

                # dynamic prefill for this chunk seeds the static cache
                prefix_ids = sequences[rows, :prefix_length]
                if numeric_template is not None:
                    prefill_num = numeric_template[:prefix_length].unsqueeze(0).expand(bsz, -1).unsqueeze(-1)
                else:
                    prefill_num = None
                logits_pre, prefill_cache = self.forward(
                    prefix_ids, None, input_num=prefill_num, memory=mem_chunk, use_cache=True)

                static_cache = StaticKVCache(
                    n_layers=n_layers, batch=bsz, n_heads=n_heads, head_dim=head_dim,
                    max_len=max_len, device=device, dtype=logits_pre.dtype)
                static_cache.seed_from_dynamic(prefill_cache)
                del prefill_cache

                next_logits = logits_pre[:, -1, :]              # predicts slot `prefix_length`
                chunk_finished = is_finished[rows].clone()      # local (bsz,) finished flags

                for current_length in range(prefix_length, max_len):
                    if (current_length - prefix_length) % early_stop_every == 0 and bool(chunk_finished.all()):
                        break
                    active = (~chunk_finished).nonzero(as_tuple=True)[0]
                    if active.numel() == 0:
                        break

                    # sample on the ACTIVE subset (eager); score from the RAW (pre-mask) distribution
                    active_logits = next_logits.index_select(0, active)            # (n_active, vocab)
                    original_scores = torch.log_softmax(active_logits, dim=-1)     # BEFORE top_k/p/temp

                    work = active_logits
                    if top_k > 0:
                        top_k_val = min(top_k, work.size(-1))
                        ignore = work < torch.topk(work, top_k_val, dim=1)[0][..., -1, None]
                        work = work.masked_fill(ignore, -float('inf'))
                    if temperature != 1.0:
                        work = work / temperature
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(work, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_to_remove = cumulative_probs > top_p
                        sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
                        sorted_to_remove[..., 0] = 0
                        indices_to_remove = sorted_to_remove.scatter(1, sorted_indices, sorted_to_remove)
                        work = work.masked_fill(indices_to_remove, -float('inf'))

                    probs = torch.softmax(work, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)   # (n_active,)

                    global_active = active + chunk_start
                    sequences[global_active, current_length] = sampled
                    scores[global_active] += torch.gather(original_scores, 1, sampled.unsqueeze(-1)).squeeze(-1)
                    chunk_finished[active] |= (sampled == eos_token)

                    if current_length == max_len - 1:
                        break  # last slot filled; no further forward needed

                    # full-width static forward of the just-written token (finished rows feed <pad>, K/V
                    # written but never read -- masked out of `active` next step)
                    token_full = sequences[rows, current_length].unsqueeze(-1)     # (bsz, 1)
                    if numeric_template is not None:
                        num_step = numeric_template[current_length:current_length + 1].unsqueeze(0).expand(bsz, -1).unsqueeze(-1)
                    else:
                        num_step = None
                    logits_full = self.forward_static(token_full, num_step, mem_chunk, static_cache, position=current_length)
                    next_logits = logits_full[:, -1, :]

                is_finished[rows] = chunk_finished
                del static_cache

                # spill-guard backstop: after the FIRST chunk, fail loud if this decode's ADDED working set
                # (peak allocated DELTA over the decode-start baseline -- the part that grew the pool) is over
                # budget. Delta vs available, both ALLOCATED-based (not the process-global reserved pool, which
                # a prior wide run can leave inflated -> false-fire).
                if _guard is not None and chunk_start == 0:
                    _added = torch.cuda.max_memory_allocated(device) - _guard[1]
                    if _added > _VRAM_GUARD_FRACTION * _guard[0]:
                        raise RuntimeError(
                            f"static decode batch={bsz} added {_added / 1024**3:.1f} GB > "
                            f"{_VRAM_GUARD_FRACTION:.0%} of {_guard[0] / 1024**3:.1f} GB available VRAM; on "
                            f"WSL2/unified memory this spills to system RAM (~28x slower) instead of OOM. "
                            f"Pass a smaller batch_size or static_decode=False.")
                pbar.update(bsz)
            pbar.close()

        completed_sequences = sequences.cpu().tolist()
        completed_scores = scores.cpu().tolist()

        if return_raw:
            return completed_sequences, completed_scores

        return self._postprocess_sampled(
            completed_sequences, completed_scores,
            simplify=simplify, unique=unique, valid_only=valid_only,
            eos_token=eos_token, verbose=verbose,
        )

    def _postprocess_sampled(
        self,
        completed_sequences: list[list[int]],
        completed_scores: list[float],
        *,
        simplify: bool | str = True,
        unique: bool = True,
        valid_only: bool = True,
        eos_token: int | None = None,
        verbose: bool = False,
        simplify_map: dict[tuple, list] | None = None,
    ) -> tuple[list[list[int]], list[float], list[bool]]:
        """Post-process raw sampled sequences -> (sequences, scores, is_valid): extract the
        <expression>, (optionally) simplify, dedup on the (simplified) expression, re-encode, sort by
        score desc. EXTRACTED VERBATIM from ``sample_top_kp`` so the default (``simplify_map=None``)
        path is byte-identical to the legacy inline post-loop.

        ``simplify_map`` (raw-expression-tuple -> simplified-expression) lets the caller precompute
        every simplipy.simplify() call in ONE parallel pass and pass the results in; the per-expression
        engine.simplify() is then a dict lookup with the SAME values, so the output stays byte-identical.
        """
        if eos_token is None:
            eos_token = self.tokenizer['<eos>']

        filtered_sequences: list[list[int]] = []
        filtered_scores: list[float] = []
        filtered_is_valid: list[bool] = []
        seen_expressions: set[tuple[str, ...]] = set()

        pbar_post = tqdm(zip(completed_sequences, completed_scores), total=len(completed_sequences), disable=not verbose, desc="Post-processing", smoothing=0.0)
        for seq, score in pbar_post:
            try:
                encoded_expression, before, after = self.tokenizer.extract_expression_from_beam(seq)
            except (ValueError, IndexError):
                continue

            encoded_expression = self.tokenizer.constantify_expression(encoded_expression)
            expression = self.tokenizer.decode(encoded_expression, special_tokens='<constant>')

            if self.simplipy_engine.is_valid(expression) and len(expression) > 1:
                if simplify is True:
                    raw_key = tuple(expression)
                    if simplify_map is not None and raw_key in simplify_map:
                        expression = simplify_map[raw_key]            # precomputed (parallel) -> same value
                    else:
                        expression = self.simplipy_engine.simplify(expression, max_pattern_length=4)
                elif simplify == 'sympy':
                    try:
                        from simplipy.utils import numbers_to_constant
                        infix = self.simplipy_engine.prefix_to_infix(expression, power='**')
                        result = _sympy_simplify_with_timeout(infix, timeout_seconds=1.0)
                        if result is not None:
                            simplified_infix = result[0].replace('Abs', 'abs')
                            parsed = self.simplipy_engine.parse(simplified_infix)
                            parsed = numbers_to_constant(parsed, inplace=True)
                            if self.simplipy_engine.is_valid(parsed):
                                expression = parsed
                    except Exception:
                        pass  # keep unsimplified expression on failure

                expression_tuple = tuple(expression)
                if unique and expression_tuple in seen_expressions:
                    continue

                try:
                    expression_tokens = self.tokenizer.encode(expression)
                except KeyError:
                    continue

                reconstructed_sequence = before + expression_tokens + after
                filtered_sequences.append(reconstructed_sequence)
                filtered_scores.append(score)
                filtered_is_valid.append(True)

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

        sorted_order = sorted(range(len(filtered_scores)), key=lambda i: filtered_scores[i], reverse=True)  # type: ignore[arg-type]
        final_sequences = [filtered_sequences[i] for i in sorted_order]
        final_scores = [filtered_scores[i] for i in sorted_order]
        final_is_valid = [filtered_is_valid[i] for i in sorted_order]

        return final_sequences, final_scores, final_is_valid

    def extract_valid_raw_expressions(self, completed_sequences: list[list[int]]) -> list[list[str]]:
        """The cheap (serial) half of post-processing: the RAW (pre-simplify) decoded expressions of
        the sequences that are valid (is_valid and len>1). Used by the chunked path to gather every
        expression that WILL be simplified, so they can be simplified in one parallel pass. The
        returned values are exactly the ``expression`` at the simplify() call site (keys into the map)."""
        out: list[list[str]] = []
        for seq in completed_sequences:
            try:
                enc, _before, _after = self.tokenizer.extract_expression_from_beam(seq)
            except (ValueError, IndexError):
                continue
            enc = self.tokenizer.constantify_expression(enc)
            expression = self.tokenizer.decode(enc, special_tokens='<constant>')
            if self.simplipy_engine.is_valid(expression) and len(expression) > 1:
                out.append(expression)
        return out

    def save(self, directory: str, config: dict[str, Any] | str | None = None, reference: str = 'relative', recursive: bool = True, errors: Literal['raise', 'warn', 'ignore'] = 'warn') -> None:
        """Save the model's state dict (and optionally its config) to ``directory``.

        Parameters
        ----------
        directory : str
            Destination directory (created if missing). The weights are written to
            ``state_dict.pt`` and, when ``config`` is given, the config to ``model.yaml``.
        config : dict[str, Any] or str or None, optional
            Config mapping or path to copy alongside the weights. If ``None``, no config is
            written and ``errors`` governs the response.
        reference : str, optional
            How referenced sub-config paths are stored (``'relative'`` by default).
        recursive : bool, optional
            Recursively resolve and copy referenced configs. Defaults to True.
        errors : {'raise', 'warn', 'ignore'}, optional
            Behaviour when ``config`` is ``None``: raise, warn (default), or silently proceed.
        """

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
        """Load a model and its config from a directory previously written by :meth:`save`.

        Parameters
        ----------
        directory : str
            Directory containing ``model.yaml`` and ``state_dict.pt``.

        Returns
        -------
        tuple[dict[str, Any], FlashANSRModel]
            The loaded config and the model with its weights restored.
        """
        directory = substitute_root_path(directory)

        config_path = os.path.join(directory, 'model.yaml')

        model = cls.from_config(config_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True))

        return load_config(config_path), model

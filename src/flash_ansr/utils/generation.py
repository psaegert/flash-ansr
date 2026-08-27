"""Generation configuration helpers with method-specific signatures."""
from typing import Any, Iterator, Literal, Mapping


def validate_simplify(value: Any) -> bool:
    """Return ``value`` if it is a valid ``simplify`` selector, else raise ``ValueError``.

    ``simplify`` is a two-state switch: ``True`` canonicalizes with SimpliPy (the product
    simplifier), ``False`` leaves the decoded expression alone. It used to accept the string
    ``'sympy'`` as a third state; that path was an ablation of the product simplifier and was
    removed (owner ruling, 2026-08-18) under the rule that ablations do not live in production
    code.

    A removed selector is REFUSED, never quietly mapped onto another one: silently answering
    ``simplify='sympy'`` with SimpliPy would change which canonicalizer a config named without
    telling anyone, which is a behaviour change wearing a removal's clothes.
    """
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"Invalid simplify={value!r}. flash-ansr removed the simplify='sympy' path (2026-08-18): "
        "SymPy simplification was an ablation of the product simplifier, and ablations do not live "
        "in production code. Use simplify=True (SimpliPy, the product default) or simplify=False "
        "(no simplification). There is no fallback -- this request is refused rather than silently "
        "served by a different simplifier."
    )


class GenerationConfigBase(Mapping[str, Any]):
    """Common interface implemented by all generation configuration objects."""

    __slots__ = ('method',)
    method: Literal['softmax_sampling', 'prior_sampling']

    def to_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments appropriate for the configured method."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_kwargs())

    def __len__(self) -> int:
        return len(self.to_kwargs())

    def __getitem__(self, key: str) -> Any:
        return self.to_kwargs()[key]

    def as_dict(self) -> dict[str, Any]:
        """Alias for :meth:`to_kwargs` mirroring the Mapping protocol."""
        return self.to_kwargs()

    def __repr__(self) -> str:
        params = ', '.join(f"{key}={value!r}" for key, value in self.to_kwargs().items())
        return f"{self.__class__.__name__}({params})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.to_kwargs() == other.to_kwargs()


class SoftmaxSamplingConfig(GenerationConfigBase):
    """Configuration for softmax sampling generation."""

    __slots__ = (
        'choices',
        'top_k',
        'top_p',
        'max_len',
        'batch_size',
        'temperature',
        'valid_only',
        'simplify',
        'unique',
        'use_cache',
        'static_decode',
        'guidance_weight',
        'constrain_ieee754',
        'compact_ieee754',
    )

    method: Literal['softmax_sampling']
    choices: int
    top_k: int
    top_p: float
    max_len: int
    batch_size: int | str
    temperature: float
    valid_only: bool
    simplify: bool
    unique: bool
    use_cache: bool
    static_decode: bool | None
    guidance_weight: float | None
    constrain_ieee754: bool
    compact_ieee754: bool

    def __init__(
        self,
        *,
        choices: int = 1024,
        top_k: int = 0,
        top_p: float = 1.0,
        max_len: int = 64,
        batch_size: int | str = 'auto',   # c-adaptive chunk size (suggest_batch_size); int overrides
        temperature: float = 1.0,
        valid_only: bool = True,
        simplify: bool = True,
        unique: bool = True,
        use_cache: bool = True,   # KV cache ON by default (quality-equivalent; the inference speed win)
        static_decode: bool | None = None,   # tri-state: None=deployed default for capable models, True/False explicit
        guidance_weight: float | None = None,   # classifier-free guidance (optcond models only); None=plain conditioned decode
        constrain_ieee754: bool = False,   # v24 grammar mask over <ieee754> spans (T6/T7)
        compact_ieee754: bool = False,     # in-decode span compaction (static path only)
    ) -> None:
        self.method = 'softmax_sampling'
        self.choices = choices
        self.top_k = top_k
        self.top_p = top_p
        self.max_len = max_len
        self.batch_size = batch_size
        self.temperature = temperature
        self.valid_only = valid_only
        self.simplify = validate_simplify(simplify)
        self.unique = unique
        self.use_cache = use_cache
        self.static_decode = static_decode
        self.guidance_weight = guidance_weight
        self.constrain_ieee754 = constrain_ieee754
        # In-decode span compaction: the static path's per-row position rewind. Validated HERE
        # rather than inside the decode, where the traceback tells a user nothing about what to
        # change (the same reason the retired beam config validated it at construction).
        if compact_ieee754:
            if not constrain_ieee754:
                raise ValueError(
                    "compact_ieee754=True requires constrain_ieee754=True: compaction decodes a "
                    "span the grammar guarantees is well-formed.")
            if static_decode is False:
                raise ValueError(
                    "compact_ieee754=True requires the static decode path: compaction is a per-row "
                    "position rewind on the static position-indexed cache.")
        self.compact_ieee754 = compact_ieee754

    def to_kwargs(self) -> dict[str, Any]:
        """Return the softmax-sampling keyword arguments (``choices``, ``top_k``, ``top_p``, ...)."""
        return {
            'choices': self.choices,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'max_len': self.max_len,
            'batch_size': self.batch_size,
            'temperature': self.temperature,
            'valid_only': self.valid_only,
            'simplify': self.simplify,
            'unique': self.unique,
            'use_cache': self.use_cache,
            'static_decode': self.static_decode,
            'guidance_weight': self.guidance_weight,
            'constrain_ieee754': self.constrain_ieee754,
            'compact_ieee754': self.compact_ieee754,
        }


GenerationConfig = SoftmaxSamplingConfig


def create_generation_config(*, method: Literal['softmax_sampling'] = 'softmax_sampling',
                             **kwargs: Any) -> GenerationConfig:
    """Factory that builds the generation configuration.

    Softmax sampling is the only method. Beam search and MCTS were retired once the
    capability that justified beam -- in-decode span compaction -- was shown to need a
    per-row decode position rather than a beam (see tests/test_models/
    test_static_position_rewind.py). The keyword survives so a stale config fails loudly.
    """
    if method.lower() != 'softmax_sampling':
        raise ValueError(
            f"Invalid generation method: {method}. Beam search and MCTS were retired; "
            f"'softmax_sampling' is the only method.")
    return SoftmaxSamplingConfig(**kwargs)


# Hardware gate for the c-adaptive caps in ``suggest_batch_size``: those caps are validated only on a
# 24 GiB RTX 4090. At or above this many GiB the measured caps apply; below it they are replaced by a
# conservative default and the runtime spill-guard becomes the backstop. (total_memory/1e9 reports ~25.8
# for a 24 GiB card, ~21.5 for 20 GB, ~17.2 for 16 GB -- so 24.0 cleanly separates 24 GiB-class from below.)
_FULL_CAP_MIN_VRAM_GB = 24.0
# Conservative chunk cap for an untested sub-24 GiB card (users wanting more pass an explicit int batch_size).
_SMALL_CARD_BATCH_CAP = 64


def suggest_batch_size(choices: int, n_params: int, vram_gb: float = 24.0) -> int:
    """Conservative c-adaptive chunk size for softmax sampling: the largest power-of-2 <= ``choices``,
    capped at a MEASURED-safe value per model size, applied only on cards >= 24 GiB.

    This is a *lookup*, NOT an analytic KV estimator: peak generation memory is dominated by
    problem-dependent candidate-expansion intermediates (e.g. batch 1024 @ c=1024 hit 22.7 GB on a
    120M model — nowhere near a clean KV estimate), so an estimator would be a new OOM source. The
    caps below are the largest batch with peak < ~21 GB at high c, measured on an RTX 4090 (24 GB):
    1B -> 128, 120M -> 512, smaller -> 1024/2048. Always overridable by an explicit int ``batch_size``.
    ``vram_gb`` should be the card's TOTAL memory (the caps already include the model footprint). The caps
    apply ONLY on cards >= ``_FULL_CAP_MIN_VRAM_GB``; a smaller, untested card falls to the conservative
    ``_SMALL_CARD_BATCH_CAP`` (the model footprint is fixed, not VRAM-proportional, so rescaling the cap by
    VRAM would over-estimate what fits) -- the runtime dynamic spill-guard backstops it.
    """
    if n_params >= 5e8:      # ~1B
        cap = 128
    elif n_params >= 5e7:    # ~120M
        cap = 512
    elif n_params >= 5e6:    # ~20M (untested -> conservative)
        cap = 1024
    else:                    # ~3M (untested -> conservative)
        cap = 2048
    # Hardware gate: the caps above are MEASURED on a 24 GiB 4090, and the model footprint is fixed (not
    # VRAM-proportional), so a linear vram_gb/24 rescale over-estimates what fits on a smaller card. Apply
    # the measured caps only at >= _FULL_CAP_MIN_VRAM_GB; otherwise fall to a conservative default (the
    # runtime dynamic spill-guard is the backstop). An explicit int batch_size bypasses this entirely.
    if vram_gb < _FULL_CAP_MIN_VRAM_GB:
        cap = min(cap, _SMALL_CARD_BATCH_CAP)
    target = min(int(choices), cap)
    b = 1
    while b * 2 <= target:
        b *= 2
    return max(1, min(int(choices), b))


def suggest_batch_size_dims(
    choices: int,
    *,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    max_len: int,
    static_decode: bool,
    free_bytes: int | None,
    dtype_bytes: int = 4,
    safety_fraction: float = 0.7,
) -> int:
    """VRAM-GENERAL, architecture-driven chunk size for the STATIC-decode path -- the largest power-of-2
    <= ``choices`` whose conservative per-row memory footprint fits in ``safety_fraction`` of FREE VRAM.

    Replaces the hardcoded 24GB lookup of :func:`suggest_batch_size` for the static path (the dynamic path
    keeps that measured lookup -- it is already validated and a conservative dims estimate would REGRESS its
    cap). Generalizes to any GPU by querying ``free_bytes`` (``torch.cuda.mem_get_info(dev)[0]``) at call
    time and scaling per-row memory by the model's own dims, so it is correct for 3M..1B on any card.

    Per-row estimate = analytic self-attn static-KV floor x an empirical overhead factor (cross-attn KV +
    activations + cuBLAS/SDPA workspace). The floor ALONE under-counts (measured static ~37 MB/row reserved
    at 1B fp32 vs ~26 MB analytic), so the overhead factor is a conservative pad calibrated to GATE-1e; the
    dynamic factor is larger (the per-step active-row gather copies the cache). On CPU / unknown VRAM
    (``free_bytes is None``) -> a conservative floor (<=128). An explicit int ``batch_size`` bypasses this.
    """
    # analytic self-attn static-KV bytes per candidate row (K and V, all layers/heads, full max_len).
    # dtype_bytes defaults to 4 (fp32) -- the deployed dtype; under half/autocast pass dtype_bytes=2.
    kv_floor = 2 * n_layers * n_heads * head_dim * max_len * dtype_bytes
    # Empirical overhead over the KV floor (the floor under-counts cross-attn KV + activations + workspace).
    # CALIBRATED AT A SINGLE POINT (GATE-1e, fp32 1B: static 37/26~=1.4 -> 1.6; dynamic 107/26~=4.1 -> 4.2).
    # The static factor must be re-checked cross-scale (3M/20M/120M) BEFORE the deployed default flips on --
    # non-KV costs shrink slower than KV, so smaller models may need a larger factor (the runtime VRAM guard
    # is the backstop until then). The dynamic factor is unused by the deployed path (it keeps the validated
    # suggest_batch_size lookup); it is here only for completeness.
    overhead = 1.6 if static_decode else 4.2
    per_row = kv_floor * overhead

    if free_bytes is None or per_row <= 0:
        # CPU / unknown card: keep a conservative power-of-2 floor, never extrapolate.
        target = min(int(choices), 128)
    else:
        budget = safety_fraction * float(free_bytes)
        cap = int(budget // per_row)
        target = min(int(choices), cap)

    b = 1
    while b * 2 <= target:
        b *= 2
    return max(1, min(int(choices), b))


def _spill_over_budget(added_bytes: float, avail_bytes: float, fraction: float) -> bool:
    """True if a decode's added working set ``added_bytes`` exceeds ``fraction`` of ``avail_bytes``.

    The pure threshold rule behind the dynamic-path VRAM spill-guard, extracted so it is unit-testable
    without a GPU. A non-positive ``avail_bytes`` (unknown / degenerate reading) returns False -- never
    block a decode on a bad memory reading.
    """
    if avail_bytes <= 0:
        return False
    return added_bytes > fraction * avail_bytes

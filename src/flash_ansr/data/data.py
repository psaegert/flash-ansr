"""The :class:`FlashANSRDataset` training-data wrapper.

Wraps a :class:`symbolic_data.ProblemSource` (backed by a catalog) so that on-the-fly sampled or
pre-generated symbolic-regression problems can be tokenized, preprocessed and collated into the
padded tensor batches consumed during training.
"""
import copy
import os
import time
import warnings
import types
from typing import Any, Callable, Generator, Literal, Sequence

import numpy as np
import torch
from datasets import Dataset, disable_progress_bars, load_from_disk
from simplipy import SimpliPyEngine
from tqdm import tqdm

from flash_ansr.data.collate import BatchFormatter
from flash_ansr.data.serialization import (
    HYPOTHESIS_TOKEN,
    MASK_MODE_TOKENS,
    PREDICT_CONSTANTS_TOKENS,
    COMPLEXITY_TOKENS,
    PREDICT_RESIDUAL_TOKENS,
    PREDICT_Y_TOKENS,
    COMPACT_CONSTANT_TOKEN,
    TAGGED_DELIMITER_TOKENS,
    TARGET_DIALECT_EXPLICIT,
    TARGET_DIALECT_TAGGED,
    TARGET_DIALECTS,
)
from flash_ansr.data.streaming import SharedMemoryWorkerPool
from flash_ansr.utils.ieee754 import IEEE754_SPECIAL_TOKENS
from flash_ansr.utils.numeric import NUMERIC_DTYPE
from symbolic_data import LampleChartonCatalog, ProblemSource
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing import FlashANSRPreprocessor
from flash_ansr.utils.config_io import load_config, save_config
from flash_ansr.utils.paths import substitute_root_path
from flash_ansr.utils.metrics import (
    build_expression_callable,
    estimate_curvature_metric,
    estimate_fisher_metric,
)


def _validate_task_block(raw: "dict[str, Any] | None", *, name: str,
                         probability_keys: tuple[str, ...],
                         int_keys: tuple[str, ...] = ()) -> "dict[str, Any] | None":
    """Validate a v24 task-block config mapping: exact keys, probabilities in [0, 1].

    Priors are pinned explicitly, never defaulted -- a missing or unknown key is refused
    loudly, like the noise-mixture spec on the symbolic-data side."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping, got {raw!r}")
    required = set(probability_keys) | set(int_keys)
    if set(raw) != required:
        raise ValueError(
            f"{name} must carry exactly {sorted(required)} (got {sorted(raw)}); "
            f"priors are pinned explicitly, never defaulted")
    validated: dict[str, Any] = {}
    for key in probability_keys:
        probability = float(raw[key])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name}.{key} must be a probability in [0, 1], got {raw[key]!r}")
        validated[key] = probability
    for key in int_keys:
        value = int(raw[key])
        if value < 1:
            raise ValueError(f"{name}.{key} must be a positive integer, got {raw[key]!r}")
        validated[key] = value
    return validated


class FlashANSRDataset:
    """Dataset wrapper for amortized neural symbolic regression training.

    Manages skeleton sampling, support point generation, optional prompt
    preprocessing, and collation into model-ready batches. Can also compile
    streaming output into an on-disk `datasets.Dataset` for deterministic
    iteration.

    Parameters
    ----------
    source : ProblemSource
        symbolic-data problem source streaming ready-to-use Problems (skeleton
        + support points) from its underlying generative catalog.
    tokenizer : Tokenizer
        Tokenizer used for expression serialization and padding.
    padding : {"random", "zero"}
        Strategy for padding numeric support points.
    preprocessor : FlashANSRPreprocessor, optional
        Prompt-aware preprocessor; when provided, prompt metadata can be
        injected during sampling or in worker processes.

    Notes
    -----
    This object owns a multiprocessing worker pool. Call `dataset.shutdown()`
    when done, or use it as a context manager
    (`with FlashANSRDataset(...) as dataset:`) so the pool is shut down
    automatically. If neither is done, a warning is emitted at garbage
    collection.
    """

    def __init__(
        self,
        source: ProblemSource,
        tokenizer: Tokenizer,
        padding: Literal["random", "zero"],
        preprocessor: FlashANSRPreprocessor | None = None,
        unconditional_prob: float = 0.0,
        condition_dropout: float | None = None,
        target_dialect: str = TARGET_DIALECT_EXPLICIT,
        complexity_block: "dict[str, Any] | None" = None,
        predict_y_block: "dict[str, Any] | None" = None,
        residual_block: "dict[str, Any] | None" = None,
        mask_block: "dict[str, Any] | None" = None,
    ) -> None:
        self.source = source
        self.tokenizer = tokenizer
        self.padding = padding
        self.preprocessor = preprocessor
        # Fraction of generated examples emitted UNCONDITIONED (no condition) -> first-class optional
        # condition (CFG). 0.0 = every example conditioned (original behavior). Set only on the TRAIN
        # dataset; keep 0.0 on val so validation CE stays a pure conditioned metric.
        #
        # `condition_dropout` is the v24 canonical name for the same probability (owner ruling:
        # "10% condition dropout ... the model has to learn to predict unconditionally"). In this
        # cross-attention architecture a dropped instance routes to the model's learned null_memory
        # (see streaming.draw_condition_mask); under the planned v24 self-attn-only decoder it
        # becomes span omission -- the data tokens omitted from the sequence -- the cleaner
        # formulation. Both spellings accepted; giving both with different values is an error.
        if condition_dropout is not None:
            condition_dropout = float(condition_dropout)
            if unconditional_prob and float(unconditional_prob) != condition_dropout:
                raise ValueError(
                    f"Conflicting condition_dropout={condition_dropout} and "
                    f"unconditional_prob={unconditional_prob}: give one (they name the same "
                    f"per-instance dropout probability)."
                )
            self.unconditional_prob = condition_dropout
        else:
            self.unconditional_prob = float(unconditional_prob)
        # Expression constants ride <ieee754> spans; <float> carries the numbers the CALLER
        # supplies (predict_y's x, a prompted complexity). Both token sets are required.
        required_tokens = (*IEEE754_SPECIAL_TOKENS, COMPACT_CONSTANT_TOKEN)
        missing_tokens = [token for token in required_tokens if token not in tokenizer]
        if missing_tokens:
            raise ValueError(
                f"the v24 numeric format requires the special tokens "
                f"{list(required_tokens)}, but the tokenizer is missing {missing_tokens}."
            )
        # Target-dialect gate: 'explicit' (default) targets the binary-prefix expression;
        # 'tagged' targets the engine's TAGGED CANONICAL output (contract A3) and requires
        # the tagged delimiter tokens in the tokenizer.
        if target_dialect not in TARGET_DIALECTS:
            raise ValueError(
                f"Unknown target_dialect {target_dialect!r}; expected one of {TARGET_DIALECTS}."
            )
        if target_dialect == TARGET_DIALECT_TAGGED:
            missing_delimiters = [token for token in TAGGED_DELIMITER_TOKENS if token not in tokenizer]
            if missing_delimiters:
                raise ValueError(
                    f"target_dialect 'tagged' requires the tagged delimiter tokens "
                    f"{list(TAGGED_DELIMITER_TOKENS)}, but the tokenizer is missing "
                    f"{missing_delimiters}."
                )
        self.target_dialect = target_dialect
        # v24 task blocks (owner ruling 2026-08-24): the optional <complexity> block and the
        # <predict_y> auxiliary block, requiring the expression wrappers and their block
        # tokens up front. A PROMPTED complexity is compact, a HYPOTHESIZED one is spelled
        # in bits, and no instance carries both (owner ruling 2026-08-27).
        self.complexity_block = _validate_task_block(
            complexity_block, name="complexity_block",
            probability_keys=("p_present", "p_hypothesize"))
        if self.complexity_block is not None:
            if self.complexity_block["p_present"] + self.complexity_block["p_hypothesize"] > 1.0:
                raise ValueError("complexity_block: p_present + p_hypothesize must not exceed 1.0 "
                                 "(the remainder is the block-absent fraction)")
            if self.complexity_block["p_hypothesize"] > 0.0 and HYPOTHESIS_TOKEN not in tokenizer:
                raise ValueError(f"complexity_block.p_hypothesize > 0 requires the {HYPOTHESIS_TOKEN} "
                                 f"token in the tokenizer")
        self.predict_y_block = _validate_task_block(
            predict_y_block, name="predict_y_block", probability_keys=("p_present", "p_conditional"),
            int_keys=("min_n_support",))
        # <predict_residual>: the observed-minus-predicted displacement at one point. It
        # needs a noise mixture to have a target at all -- without one the residual is
        # identically 0.0 and the block teaches only "emit eight zero bytes" -- so the
        # source is checked here rather than letting a run train on a degenerate task.
        self.residual_block = _validate_task_block(
            residual_block, name="residual_block", probability_keys=("p_present", "p_conditional"),
            int_keys=("min_n_support",))
        if self.residual_block is not None and getattr(source, "noise_spec", None) is None:
            raise ValueError(
                "residual_block requires a source with a noise mixture: without one the "
                "observed targets ARE the clean ones, every residual is exactly 0.0, and the "
                "block has nothing to teach. Configure sampling.noise, or remove the block.")
        # The promptable-mask + constant-infilling feature (owner rulings 2026-08-24):
        # 'all'/'fittable' emission formats behind harness-owned flags, the unflagged
        # per-slot partial circumstance, and the <predict_constants> block probabilities.
        # All six priors pinned, never defaulted.
        self.mask_block = _validate_task_block(
            mask_block, name="mask_block",
            probability_keys=("p_mask_all", "p_mask_fittable", "p_partial", "p_placeheld",
                              "p_predict_constants_flagged", "p_predict_constants_partial"))
        if self.mask_block is not None:
            if self.mask_block["p_mask_all"] + self.mask_block["p_mask_fittable"] > 1.0:
                raise ValueError("mask_block: p_mask_all + p_mask_fittable must not exceed 1.0 "
                                 "(the remainder is the unmasked fraction)")
            required = sorted(MASK_MODE_TOKENS.values()) + ["<constant>",
                                                            *PREDICT_CONSTANTS_TOKENS]
            missing_tokens = [t for t in required if t not in tokenizer]
            if missing_tokens:
                raise ValueError(f"mask_block requires tokens {required}, missing {missing_tokens}.")
            if target_dialect != TARGET_DIALECT_TAGGED:
                raise ValueError(
                    "mask_block requires target_dialect='tagged': the per-slot site walk "
                    "is defined on the tagged canonical (the explicit path's site filters "
                    "diverge on np.pi/np.e and would break slot alignment).")
        if (self.complexity_block is not None or self.predict_y_block is not None
                or self.mask_block is not None or self.residual_block is not None):
            missing_wrappers = [t for t in ("<expression>", "</expression>") if t not in tokenizer]
            if missing_wrappers:
                raise ValueError(f"task blocks require the expression wrappers, missing {missing_wrappers}.")
        if self.complexity_block is not None:
            missing_tokens = [t for t in COMPLEXITY_TOKENS if t not in tokenizer]
            if missing_tokens:
                raise ValueError(f"complexity_block requires tokens {list(COMPLEXITY_TOKENS)}, missing {missing_tokens}.")
        if self.predict_y_block is not None:
            missing_tokens = [t for t in PREDICT_Y_TOKENS if t not in tokenizer]
            if missing_tokens:
                raise ValueError(f"predict_y_block requires tokens {list(PREDICT_Y_TOKENS)}, missing {missing_tokens}.")
        if self.residual_block is not None:
            missing_tokens = [t for t in PREDICT_RESIDUAL_TOKENS if t not in tokenizer]
            if missing_tokens:
                raise ValueError(f"residual_block requires tokens {list(PREDICT_RESIDUAL_TOKENS)}, missing {missing_tokens}.")
        self.data = None
        #: Monotone worker-counter sums (skipped blocks, restructure gate, drops).
        self.stream_counters: dict[str, int] = {}

        self._collator = BatchFormatter(tokenizer=tokenizer)
        self._stream = SharedMemoryWorkerPool(
            source=source,
            tokenizer=tokenizer,
            padding=padding,
            target_dialect=target_dialect,
            complexity_block=self.complexity_block,
            predict_y_block=self.predict_y_block,
            residual_block=self.residual_block,
            mask_block=self.mask_block,
        )
        self._preprocessor_prompt_config = (
            copy.deepcopy(preprocessor.prompt_config) if preprocessor is not None else None
        )

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        # __init__ validates before it builds the stream, so a rejected config leaves the
        # instance without `_stream`; the destructor must not raise on top of that.
        stream = getattr(self, "_stream", None)
        if stream is not None and stream.is_initialized:
            warnings.warn(
                "FlashANSRDataset was not explicitly shut down. "
                "Call `dataset.shutdown()` for cleaner resource management. Shutting down in destructor.",
            )
            self.shutdown()

    def __enter__(self) -> "FlashANSRDataset":
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None, exc_tb: types.TracebackType | None) -> None:  # pragma: no cover - convenience helper
        self.shutdown()

    @property
    def simplipy_engine(self) -> SimpliPyEngine:
        """The :class:`~simplipy.SimpliPyEngine` used by this dataset's underlying catalog."""
        return self.source.catalog.simplipy_engine

    @property
    def condition_dropout(self) -> float:
        """The per-instance condition-dropout probability (v24 name for ``unconditional_prob``)."""
        return self.unconditional_prob

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "FlashANSRDataset":
        """Instantiate from a YAML/dict config.

        Paths are normalized via `load_config` and `substitute_root_path`. The
        config carries a `source:` block: `{catalog: <path-to-catalog-yaml OR
        inline dict>, sampling: {...}}`. The catalog (a generative
        `lample_charton` catalog) is loaded into a dict and handed to a
        `ProblemSource`.

        Parameters
        ----------
        config : dict or str
            Dataset config or path to a YAML file.

        Returns
        -------
        FlashANSRDataset
            Dataset wrapper with tokenizer and optional preprocessor wired.
        """
        config_ = load_config(config)

        if "dataset" in config_.keys():
            config_ = config_["dataset"]

        for key in ("source", "tokenizer", "padding"):
            if key not in config_:
                raise ValueError(f"Dataset config is missing required key {key!r}.")

        source_cfg = config_["source"]
        if "catalog" not in source_cfg:
            raise ValueError("Dataset config `source` block is missing required key 'catalog'.")
        catalog_cfg = source_cfg["catalog"]

        if isinstance(config, str) and isinstance(catalog_cfg, str) and catalog_cfg.startswith('.'):
            catalog_cfg = os.path.join(os.path.dirname(config), catalog_cfg)  # pragma: no cover - config guard
        if isinstance(catalog_cfg, str):
            catalog_cfg = substitute_root_path(catalog_cfg)

        # `source.catalog` may be: a curated NAME[@version] (resolved from HF), a catalog config path,
        # an inline generative-catalog dict, or a DIRECTORY holding a saved generative catalog (a fixed
        # validation pool). ProblemSource resolves names / paths / inline configs via build_catalog; only
        # the saved-directory form is loaded into an instance first (build_catalog has no saved-dir loader).
        catalog_spec: Any
        if isinstance(catalog_cfg, str) and os.path.isdir(catalog_cfg):
            catalog_spec = LampleChartonCatalog.load(catalog_cfg)
        else:
            catalog_spec = catalog_cfg

        source_obj = ProblemSource({"catalog": catalog_spec, "sampling": source_cfg.get("sampling", {})})

        # The catalog's `simplify` is symbolic-data's parameter, which flash-ansr only passes
        # through -- and symbolic-data still offers its own `simplify='sympy'` skeleton path.
        # flash-ansr removed SymPy simplification (owner ruling, 2026-08-18: it is an ablation of
        # the product simplifier, and ablations do not live in production code), so the removal on
        # this side is a REFUSAL at the boundary: flash-ansr will not build a data source that
        # routes into it. Checked on the constructed catalog rather than the raw config so every
        # spelling is covered (inline dict, config path, curated name, saved directory).
        # Construction does not sample, so nothing has run through the removed path yet.
        catalog_simplify = getattr(source_obj.catalog, "simplify", None)
        if catalog_simplify is not None and not isinstance(catalog_simplify, bool):
            raise ValueError(
                f"The catalog config requests simplify={catalog_simplify!r}. flash-ansr removed the "
                "simplify='sympy' path (2026-08-18): SymPy simplification was an ablation of the "
                "product simplifier, and ablations do not live in production code. symbolic-data "
                "still implements it, but flash-ansr refuses to build a data source that routes into "
                "it. Set `simplify: true` (SimpliPy, the product default) or `simplify: false` in the "
                "catalog config. There is no fallback -- this config is refused rather than silently "
                "served by a different simplifier."
            )

        tokenizer = Tokenizer.from_config(config_["tokenizer"])

        preprocessor_cfg = config_.get("preprocessor") if isinstance(config_, dict) else None
        preprocessor: FlashANSRPreprocessor | None = None
        if preprocessor_cfg is not None:
            preprocessor = FlashANSRPreprocessor.from_config(
                preprocessor_cfg,
                simplipy_engine=source_obj.catalog.simplipy_engine,
                tokenizer=tokenizer,
                catalog=source_obj.catalog,
            )

        return cls(
            source=source_obj,
            tokenizer=tokenizer,
            padding=config_["padding"],
            preprocessor=preprocessor,
            unconditional_prob=config_.get("unconditional_prob", 0.0),
            # v24 canonical key for the same probability; the constructor rejects a conflict.
            condition_dropout=config_.get("condition_dropout"),
            target_dialect=config_.get("target_dialect", TARGET_DIALECT_EXPLICIT),
            complexity_block=config_.get("complexity_block"),
            predict_y_block=config_.get("predict_y_block"),
            residual_block=config_.get("residual_block"),
            mask_block=config_.get("mask_block"),
        )

    def save(
        self,
        directory: str,
        *args: Any,
        config: dict[str, Any] | str | None = None,
        reference: str = "relative",
        recursive: bool = True,
        **kwargs: Any,
    ) -> None:
        """Persist the compiled dataset and its config.

        Parameters
        ----------
        directory : str
            Target directory for `dataset/` artifacts and `dataset.yaml`.
        config : dict or str, optional
            Config to save alongside the dataset. When omitted a warning is
            raised and only the data is stored.
        reference : str, default "relative"
            How to normalize paths when writing the config.
        recursive : bool, default True
            Whether to recursively resolve nested configs.
        *args, **kwargs : Any
            Passed to `datasets.Dataset.save_to_disk`.
        """
        if self.data is None:
            raise ValueError("No dataset to save. Please generate or load a dataset first.")

        directory = substitute_root_path(directory)
        os.makedirs(directory, exist_ok=True)

        self.data.save_to_disk(os.path.join(directory, "dataset"), *args, **kwargs)

        if config is None:
            warnings.warn(
                "No config specified, saving the model without a config file. "
                "Loading the model will require manual configuration.",
            )
        else:
            save_config(
                load_config(config, resolve_paths=True),
                directory=directory,
                filename="dataset.yaml",
                reference=reference,
                recursive=recursive,
                resolve_paths=True,
            )

    @classmethod
    def load(cls, directory: str) -> tuple[dict[str, Any], "FlashANSRDataset"]:
        """Load a saved dataset and its config from disk.

        Parameters
        ----------
        directory : str
            Directory containing `dataset.yaml` and `dataset/`.

        Returns
        -------
        tuple
            `(resolved_config, dataset)` with the dataset ready for iteration.

        Notes
        -----
        Unlike `FlashANSR.load`, which returns the model object directly, this
        method returns a `(config, dataset)` tuple. Unpack the result, e.g.
        `config, dataset = FlashANSRDataset.load(directory)`.
        """
        config_path = os.path.join(directory, "dataset.yaml")
        resolved_directory = substitute_root_path(directory)

        dataset = cls.from_config(config_path)
        dataset.data = load_from_disk(os.path.join(resolved_directory, "dataset"))

        return load_config(config_path), dataset

    def collate(self, batch: dict[str, Any], device: str | torch.device | int = "cpu") -> dict[str, Any]:
        """Format a raw batch into tensors expected by the model.

        Parameters
        ----------
        batch : dict
            Raw batch containing support points, targets, and metadata.
        device : str or torch.device or int, default "cpu"
            Device to place returned tensors on.

        Returns
        -------
        dict
            Collated batch with padded tensors and ensured numeric channel.
        """
        return self._collator.collate(batch, device=device)

    def compile(
        self,
        size: int | None = None,
        steps: int | None = None,
        batch_size: int | None = None,
        n_support: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Materialize a streaming iterator into an on-disk dataset.

        Parameters
        ----------
        size : int, optional
            Total number of samples to generate (used if `steps` is None).
        steps : int, optional
            Number of iteration steps (overrides `size` when provided).
        batch_size : int, optional
            Per-step generation batch size; defaults to 1.
        n_support : int, optional
            Number of support points per equation; falls back to pool defaults.
        verbose : bool, default False
            Enable progress reporting.
        """
        disable_progress_bars()
        if size is None and steps is None:
            size = self.source.size_hint()
            if size is None:
                raise ValueError(
                    "Cannot infer a dataset size from an unbounded ProblemSource. "
                    "Pass an explicit `size` or `steps` to `compile()`."
                )

        self.data = Dataset.from_list(
            list(
                self.iterate(
                    size=size,
                    steps=steps,
                    batch_size=batch_size,
                    n_support=n_support,
                    verbose=verbose,
                    persistent=True,  # clone tensors out of worker shared memory before shutdown frees it (avoids use-after-free)
                )
            )
        )

    @staticmethod
    def _inject_preprocessed_fields(batch: dict[str, Any], samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        for key in samples[0].keys():
            batch[key] = [sample[key] for sample in samples]

    def _compute_expression_metrics(self, batch: dict[str, Any], metrics: Sequence[str] | str) -> None:
        expressions = batch.get("expression")
        x_tensors = batch.get("x_tensors")
        data_attn_mask = batch.get("data_attn_mask")
        if not expressions or x_tensors is None:
            return

        if isinstance(metrics, str):
            if metrics.lower() == "all":
                metrics_set = {"fisher", "hessian"}
            else:
                metrics_set = {metrics.lower()}
        else:
            metrics_set = set(m.lower() for m in metrics)
        compute_fisher = "fisher" in metrics_set
        compute_hessian = "hessian" in metrics_set
        if not (compute_fisher or compute_hessian):
            return

        if data_attn_mask is None:
            data_attn_mask = torch.ones(
                x_tensors.shape[:2],
                device=x_tensors.device,
                dtype=torch.bool,
            )

        compiled_cache: dict[tuple[str, ...], Callable[[torch.Tensor], torch.Tensor] | None] = {}
        fisher_vals: list[float] = []
        hessian_vals: list[float] = []

        for idx, expression_tokens in enumerate(expressions):
            expr_key = tuple(str(token) for token in expression_tokens)
            compiled_fn = compiled_cache.get(expr_key)
            if compiled_fn is None:
                try:
                    compiled_fn = build_expression_callable(
                        self.source.catalog.simplipy_engine,
                        expression_tokens,
                        self.source.catalog.variables,
                    )
                except Exception:
                    compiled_fn = None
                compiled_cache[expr_key] = compiled_fn

            if compiled_fn is None:
                if compute_fisher:
                    fisher_vals.append(float("nan"))
                if compute_hessian:
                    hessian_vals.append(float("nan"))
                continue

            mask = data_attn_mask[idx]
            X = x_tensors[idx]
            X = X[mask] if mask is not None else X
            X = X.to(dtype=NUMERIC_DTYPE)

            try:
                if compute_fisher:
                    fisher = estimate_fisher_metric(compiled_fn, X)
                    fisher_vals.append(float(fisher.detach().cpu().item()))
                if compute_hessian:
                    curvature = estimate_curvature_metric(compiled_fn, X)
                    hessian_vals.append(float(curvature.detach().cpu().item()))
            except Exception:
                if compute_fisher:
                    fisher_vals.append(float("nan"))
                if compute_hessian:
                    hessian_vals.append(float("nan"))

        if compute_fisher:
            batch["fisher_metric"] = torch.tensor(fisher_vals, dtype=NUMERIC_DTYPE)
        if compute_hessian:
            batch["curvature_metric"] = torch.tensor(hessian_vals, dtype=NUMERIC_DTYPE)

    def _initialize_stream(
        self,
        *,
        prefetch_factor: int,
        batch_size: int,
        n_per_equation: int,
        max_seq_len: int,
        max_n_support: int | None,
        num_workers: int | None,
        tokenizer_oov: Literal["unk", "raise"],
        worker_preprocess: bool,
        unconditional_prob: float = 0.0,
    ) -> None:
        self._stream.initialize(
            prefetch_factor=prefetch_factor,
            batch_size=batch_size,
            n_per_equation=n_per_equation,
            max_seq_len=max_seq_len,
            max_n_support=max_n_support,
            num_workers=num_workers,
            tokenizer_oov=tokenizer_oov,
            worker_preprocess=worker_preprocess,
            preprocessor_prompt_config=self._preprocessor_prompt_config,
            unconditional_prob=unconditional_prob,
        )

    def shutdown(self) -> None:
        """Release multiprocessing workers and shared buffers."""
        self._stream.shutdown()

    def iterate(
        self,
        size: int | None = None,
        steps: int | None = None,
        batch_size: int | None = None,
        n_support: int | None = None,
        max_seq_len: int = 512,
        max_n_support: int | None = None,
        n_per_equation: int = 1,
        preprocess: bool = False,
        preprocess_in_worker: bool | None = None,
        include_metrics: Sequence[str] | str | None = None,
        tokenizer_oov: Literal["unk", "raise"] = "raise",
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        persistent: bool = False,
        unconditional_prob: float | None = None,
        keep_alive: bool = False,
        tqdm_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream batches of synthetic data.

        Parameters
        ----------
        size : int, optional
            Total number of samples to generate (used if `steps` is None).
        steps : int, optional
            Number of generation steps; overrides `size` when set.
        batch_size : int, optional
            Samples per step; defaults to 1.
        n_support : int, optional
            Support points per equation; pool default when None.
        max_seq_len : int, default 512
            Maximum prefix length for generated expressions.
        max_n_support : int, optional
            Upper bound for support points; used for padding.
        n_per_equation : int, default 1
            Number of datasets to draw per skeleton before moving on.
        preprocess : bool, default False
            Whether to run the preprocessor on generated batches.
        preprocess_in_worker : bool, optional
            Force preprocessing inside workers (True), main process (False), or auto-select (None).
        include_metrics : Sequence[str] or str or None, default None
            Metrics to compute for each sampled expression. Supported values: "fisher", "hessian".
        tokenizer_oov : {"unk", "raise"}, default "raise"
            How to handle tokens missing from the tokenizer.
        num_workers : int, optional
            Worker count for multiprocessing; defaults to CPU count when None.
        prefetch_factor : int, default 2
            Jobs per worker to pre-schedule.
        persistent : bool, default False
            Clone tensors to detach from shared memory buffers.
        keep_alive : bool, default False
            Leave the worker pool running after a fully drained stream so the next
            iterate with identical settings reuses it. The caller then owns the pool
            and must call `shutdown()`.
        tqdm_kwargs : dict, optional
            Additional arguments forwarded to tqdm progress bars.
        verbose : bool, default False
            Enable progress reporting.

        Yields
        ------
        dict
            Model-ready batch with tensors and optional prompt metadata.
        """
        if batch_size is None:
            batch_size = 1

        tqdm_kwargs = dict(tqdm_kwargs) if tqdm_kwargs else {}

        if preprocess:
            # The prompt serializer rebuilds the expression body from the RAW skeleton
            # (PromptFeatures.expression_tokens), which would silently discard the mixed
            # constant serialization. Refuse loudly until the prompt path is threaded.
            raise NotImplementedError(
                "preprocess=True is not supported yet: prompt serialization would rebuild the "
                "expression body from the raw skeleton and drop the mixed constant forms."
            )

        use_worker_preprocess = False
        if preprocess:
            if self.preprocessor is None:
                if preprocess_in_worker:
                    warnings.warn(
                        "worker preprocessing requested but no preprocessor configured; falling back to main process.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                if preprocess_in_worker is None:
                    use_worker_preprocess = True
                else:
                    use_worker_preprocess = bool(preprocess_in_worker)

        if self._stream.is_initialized and self._stream.worker_preprocess_enabled != use_worker_preprocess:
            raise RuntimeError(
                "Cannot switch worker preprocessing mode while workers are active. "
                "Call `dataset.shutdown()` before iterating with a new mode."
            )

        if self.data is not None:
            if include_metrics:
                warnings.warn(
                    "Metric computation is only supported for streaming datasets; ignoring include_metrics.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            precompiled_kwargs = tqdm_kwargs.copy()
            precompiled_kwargs.setdefault("desc", "Iterating over pre-compiled dataset")
            precompiled_kwargs.setdefault("disable", not verbose)
            precompiled_kwargs.setdefault("smoothing", 0.0)
            yield from tqdm(self.data, **precompiled_kwargs)
            return

        if steps is None and size is None:
            raise ValueError("Either size or steps must be specified.")

        if steps is None:
            assert size is not None
            steps = (size + batch_size - 1) // batch_size

        effective_unconditional_prob = self.unconditional_prob if unconditional_prob is None else float(unconditional_prob)
        self._initialize_stream(
            prefetch_factor=prefetch_factor,
            batch_size=batch_size,
            n_per_equation=n_per_equation,
            max_seq_len=max_seq_len,
            max_n_support=max_n_support,
            num_workers=num_workers,
            tokenizer_oov=tokenizer_oov,
            worker_preprocess=use_worker_preprocess,
            unconditional_prob=effective_unconditional_prob,
        )

        if not self._stream.buffers:
            raise RuntimeError("Multiprocessing resources are not properly initialized.")

        pool_size = self._stream.pool_size
        drained = False

        progress_kwargs = tqdm_kwargs.copy()
        progress_kwargs.setdefault("total", steps)
        progress_kwargs.setdefault("desc", "Generating Batches")
        progress_kwargs.setdefault("disable", not verbose)
        progress_kwargs.setdefault("smoothing", 0.0)
        pbar = tqdm(**progress_kwargs)

        try:
            for _ in range(min(pool_size, steps)):
                slot_idx = self._stream.acquire_slot()
                self._stream.submit_job(slot_idx, n_support)

            for step_id in range(steps):
                completed_slot_idx, metadata_and_constants = self._stream.get_completed_slot()
                if metadata_and_constants is None:
                    raise RuntimeError("Worker returned empty payload.")

                metadata_batch = metadata_and_constants["metadata"]
                metadata_fields: dict[str, list[Any]] = {}
                if metadata_batch:
                    for key in metadata_batch[0]:
                        metadata_fields[key] = [entry[key] for entry in metadata_batch]

                # Worker health counters ("counted, never silent" -- audit 2026-08-24:
                # they were shipped in the payload and read by nobody). Monotone sums
                # over the run, logged by the trainer.
                for counter_key in ("n_skipped_task_blocks",
                                    "n_dropped_nonfinite", "n_dropped_truncation"):
                    value = metadata_and_constants.get(counter_key)
                    if value is not None:
                        self.stream_counters[counter_key] = (
                            self.stream_counters.get(counter_key, 0) + int(value))

                batch_dict = {
                    "x_tensors": torch.from_numpy(self._stream.buffers["x_tensors"][completed_slot_idx]),
                    "y_tensors": torch.from_numpy(self._stream.buffers["y_tensors"][completed_slot_idx]),
                    "data_attn_mask": torch.from_numpy(self._stream.buffers["data_attn_mask"][completed_slot_idx]).to(torch.bool),
                    "input_ids": torch.from_numpy(self._stream.buffers["input_ids"][completed_slot_idx]),
                    "constants": [
                        torch.from_numpy(c)
                        for c in metadata_and_constants["constants"]
                    ],
                }
                # T0 contract (key present <=> feature on): the contamination labels are
                # emitted only when the source runs a noise mixture.
                if getattr(self._stream.source, "noise_spec", None) is not None:
                    batch_dict["outlier_mask"] = torch.from_numpy(
                        self._stream.buffers["outlier_mask"][completed_slot_idx]).to(torch.bool)
                    # clone(), NOT .to(): a same-dtype cast is a no-op that would hand back a
                    # VIEW onto the worker ring, which the pool recycles on refill --
                    # dereferencing one after the next batch is a hard SIGSEGV, not an
                    # exception. Anything that outlives its batch must COPY. outlier_mask
                    # escapes this only because its bool cast copies.
                    batch_dict["residual"] = torch.from_numpy(
                        self._stream.buffers["residual"][completed_slot_idx]).clone()
                batch_dict.update(metadata_fields)

                preprocessed_batch = metadata_and_constants.get("preprocessed")
                if preprocess:
                    if use_worker_preprocess:
                        if preprocessed_batch is not None:
                            self._inject_preprocessed_fields(batch_dict, preprocessed_batch)
                        elif self.preprocessor:
                            batch_dict = self.preprocessor.format(batch_dict)
                    elif self.preprocessor:
                        batch_dict = self.preprocessor.format(batch_dict)

                self._collator.ensure_numeric_channel(batch_dict)

                if include_metrics:
                    self._compute_expression_metrics(batch_dict, include_metrics)

                if persistent:
                    cloned_batch: dict[str, Any] = {}
                    for key, value in batch_dict.items():
                        if isinstance(value, torch.Tensor):
                            cloned_batch[key] = value.clone()
                        elif key == "constants" and isinstance(value, list):
                            cloned_batch[key] = [tensor.clone() for tensor in value]
                        elif key == "constants":
                            cloned_batch[key] = value
                        else:
                            cloned_batch[key] = value
                    batch_dict = cloned_batch

                yield batch_dict

                pbar.update(1)

                self._stream.release_slot(completed_slot_idx)
                if step_id + pool_size < steps:
                    slot_to_refill = self._stream.acquire_slot()
                    self._stream.submit_job(slot_to_refill, n_support)
            drained = True
        finally:
            pbar.close()
            # Keeping the pool alive across calls saves the workers' catalog parse, which
            # dominates a short run. Only a fully drained stream is safe to reuse: an
            # abandoned generator leaves jobs in flight, so that case still shuts down.
            if not (keep_alive and drained):
                self.shutdown()

    def _benchmark(self, n_samples: int, batch_size: int, verbose: bool = False) -> dict[str, Any]:
        iteration_times = []
        time_1 = time.time()
        for _ in self.iterate(
            size=n_samples,
            steps=None,
            batch_size=batch_size,
            n_support=None,
            verbose=verbose,
        ):
            iteration_times.append(time.time() - time_1)
            time_1 = time.time()

        iteration_times_array = np.array(iteration_times)

        return {
            "mean_iteration_time": iteration_times_array.mean(),
            "std_iteration_time": iteration_times_array.std(),
            "min_iteration_time": iteration_times_array.min(),
            "max_iteration_time": iteration_times_array.max(),
        }

    def __len__(self) -> int:
        if self.data is None:
            raise ValueError("No dataset to get the length of. Please generate or load a dataset first.")

        return len(self.data)

"""Shared-memory streaming of procedurally generated training samples."""

import multiprocessing as mp
import os
import signal
import warnings
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.managers import ListProxy, SyncManager
from typing import Any, Literal

import numpy as np

from symbolic_data import ProblemSource
from simplipy.utils import substitute_constants
from symbolic_data.token_ops import tagged_canonical
from flash_ansr.data.serialization import (
    COMPACT_CONSTANT_TOKEN,
    HYPOTHESIS_TOKEN,
    MASK_MODE_TOKENS,
    MASKED_CONSTANT_TOKEN,
    PREDICT_CONSTANTS_END_TOKEN,
    PREDICT_CONSTANTS_START_TOKEN,
    COMPLEXITY_END_TOKEN,
    COMPLEXITY_START_TOKEN,
    POINT_END_TOKEN,
    POINT_START_TOKEN,
    PREDICT_Y_END_TOKEN,
    PREDICT_Y_START_TOKEN,
    TARGET_DIALECT_EXPLICIT,
    TARGET_DIALECT_TAGGED,
    serialize_constant_tokens,
    truncation_cuts_ieee754_span,
)
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.numeric import NUMERIC_DTYPE_NP
from flash_ansr.preprocessing import FlashANSRPreprocessor
from flash_ansr.utils.ieee754 import (
    IEEE754_END_TOKEN,
    IEEE754_N_NIBBLES,
    IEEE754_SPAN_LENGTH,
    IEEE754_START_TOKEN,
    float32_to_nibble_tokens,
)
from flash_ansr.utils.skeleton import (
    NonFiniteExpressionError, fittable_slots, mask_literals_positional,
    mask_selected_sites, nonspecial_site_positions)
from flash_ansr.utils.tensor_ops import mask_unused_variable_columns


# Per-position task-segment ids (metadata "task_segments"): the trainer splits the CE
# by these so each task's learning curve is visible on wandb.
TASK_SEGMENT_EXPRESSION = 0
TASK_SEGMENT_COMPLEXITY = 1
TASK_SEGMENT_PREDICT_Y = 2
TASK_SEGMENT_PREDICT_CONSTANTS = 3


def draw_condition_mask(rng: np.random.Generator, condition_dropout: float) -> bool:
    """One per-instance condition-dropout draw (owner ruling: 10% condition dropout).

    Returns ``True`` (conditioned: keep the (X, y) data) with probability
    ``1 - condition_dropout`` and ``False`` (dropped: the model must predict
    unconditionally) with probability ``condition_dropout``. In the current
    cross-attention architecture a ``False`` routes the example to the learned
    ``null_memory`` (see FlashANSRModel.forward's condition_mask); under the planned
    v24 self-attn-only decoder this becomes SPAN OMISSION -- the data tokens are
    literally omitted from the sequence -- the cleaner formulation of the same ruling.

    Extracted as a seam so the draw is unit-testable with a seeded ``rng`` (rate and
    determinism); the streaming worker drives it with its per-worker post-fork rng.
    """
    return bool(rng.random() >= condition_dropout)


@dataclass
class WorkerConfig:
    """Configuration passed to worker processes generating samples."""
    source_config: dict[str, Any]
    tokenizer: Tokenizer
    padding: Literal["random", "zero"]
    n_per_equation: int
    batch_size: int
    tokenizer_oov: Literal["unk", "raise"]
    worker_preprocess: bool
    max_seq_len: int
    preprocessor_prompt_config: dict[str, Any] | None
    unconditional_prob: float = 0.0
    target_dialect: str = TARGET_DIALECT_EXPLICIT
    complexity_block: dict[str, Any] | None = None
    predict_y_block: dict[str, Any] | None = None
    mask_block: dict[str, Any] | None = None


class SharedMemoryWorkerPool:
    """Manage worker processes that stream samples into shared memory."""

    def __init__(
        self,
        *,
        source: ProblemSource,
        tokenizer: Tokenizer,
        padding: Literal["random", "zero"],
        target_dialect: str = TARGET_DIALECT_EXPLICIT,
        complexity_block: dict[str, Any] | None = None,
        predict_y_block: dict[str, Any] | None = None,
        mask_block: dict[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.tokenizer = tokenizer
        self.padding = padding
        self.target_dialect = target_dialect
        self.complexity_block = complexity_block
        self.predict_y_block = predict_y_block
        self.mask_block = mask_block

        self._manager: SyncManager | None = None
        self._shms: dict[str, shared_memory.SharedMemory] = {}
        self.buffers: dict[str, np.ndarray] = {}
        self.metadata_pool: ListProxy | None = None
        self._work_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None
        self._available_slots_queue: mp.Queue | None = None
        self._workers: list[mp.Process] = []
        self._num_workers = 0
        self.pool_size = 0
        self.worker_preprocess_enabled = False
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def initialize(
        self,
        *,
        prefetch_factor: int,
        batch_size: int,
        n_per_equation: int,
        max_seq_len: int,
        max_n_support: int | None = None,
        num_workers: int | None = None,
        tokenizer_oov: Literal["unk", "raise"] = "raise",
        worker_preprocess: bool = False,
        preprocessor_prompt_config: dict[str, Any] | None = None,
        unconditional_prob: float = 0.0,
    ) -> None:
        """Allocate shared buffers and spin up producer workers."""
        if self._is_initialized:
            return

        self.worker_preprocess_enabled = worker_preprocess
        self._num_workers = os.cpu_count() or 1 if num_workers is None else num_workers
        self.pool_size = self._num_workers * prefetch_factor

        if max_n_support is None:
            max_n_support = self.source.max_n_support
            if max_n_support is None:
                raise ValueError(
                    "Support sampler configuration must define a maximum support size via "
                    "'n_support_prior.kwargs.max_value' or an equivalent field."
                )

        shm_configs: dict[str, dict[str, Any]] = {
            "x_tensors": {
                "shape": (self.pool_size, batch_size, max_n_support, len(self.source.catalog.variables)),
                "dtype": NUMERIC_DTYPE_NP,
            },
            "y_tensors": {
                "shape": (self.pool_size, batch_size, max_n_support, 1),
                "dtype": NUMERIC_DTYPE_NP,
            },
            "outlier_mask": {
                # per-point contamination labels from the source's noise mixture (all-zero
                # without one); float32 like data_attn_mask, bool-cast downstream
                "shape": (self.pool_size, batch_size, max_n_support),
                "dtype": np.float32,
            },
            "residual": {
                # per-point y_observed - f(x): what the residual head predicts. Exactly zero
                # everywhere when the source runs without a noise spec (y_noisy IS y_clean),
                # which is the correct target, not a missing one. Stored RAW and unscaled --
                # any ruler is a train-side choice (see Trainer.residual_scale), so the buffer
                # never bakes one in.
                "shape": (self.pool_size, batch_size, max_n_support),
                "dtype": NUMERIC_DTYPE_NP,
            },
            "data_attn_mask": {
                "shape": (self.pool_size, batch_size, max_n_support),
                "dtype": np.float32,
            },
            "input_ids": {
                "shape": (self.pool_size, batch_size, max_seq_len),
                "dtype": np.int64,
            },
        }

        self._shms = {
            name: shared_memory.SharedMemory(
                create=True,
                size=int(np.prod(cfg["shape"]) * np.dtype(cfg["dtype"]).itemsize),
            )
            for name, cfg in shm_configs.items()
        }
        for name, shm in self._shms.items():
            shm_configs[name]["name"] = shm.name

        self.buffers = {
            name: np.ndarray(cfg["shape"], dtype=cfg["dtype"], buffer=self._shms[name].buf)
            for name, cfg in shm_configs.items()
        }

        self._manager = mp.Manager()
        self.metadata_pool = self._manager.list([None] * self.pool_size)
        self._work_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._available_slots_queue = mp.Queue()
        for idx in range(self.pool_size):
            self._available_slots_queue.put(idx)

        # Each worker rebuilds its OWN ProblemSource from this config (with its own post-fork rng)
        # for decorrelation; never pickle a live source. `problems_per_expression` carries the old
        # `n_per_equation` grouping so consecutive problems share a skeleton when n_per_equation > 1.
        # Shallow-copy the config + a fresh sampling sub-dict (do NOT deep-copy: a loaded validation
        # catalog instance can live under "catalog" and is shared/pickled to workers, not copied here).
        source_config = dict(self.source.config)
        source_config["sampling"] = {**self.source.config.get("sampling", {}), "problems_per_expression": n_per_equation}

        worker_config = WorkerConfig(
            source_config=source_config,
            tokenizer=self.tokenizer,
            padding=self.padding,
            n_per_equation=n_per_equation,
            batch_size=batch_size,
            tokenizer_oov=tokenizer_oov,
            worker_preprocess=worker_preprocess,
            max_seq_len=max_seq_len,
            preprocessor_prompt_config=preprocessor_prompt_config,
            unconditional_prob=unconditional_prob,
            target_dialect=self.target_dialect,
            complexity_block=self.complexity_block,
            predict_y_block=self.predict_y_block,
            mask_block=self.mask_block,
        )

        self._workers = []
        for _ in range(self._num_workers):
            process = mp.Process(
                target=_producer_worker,
                args=(self._work_queue, self._result_queue, shm_configs, self.metadata_pool, worker_config),
                daemon=True,
            )
            process.start()
            self._workers.append(process)

        self._is_initialized = True

    def shutdown(self) -> None:
        """Tear down workers and release shared resources."""
        if not self._is_initialized:
            return

        if self._work_queue is None or self._result_queue is None or self._available_slots_queue is None:
            raise RuntimeError("Multiprocessing resources are not properly initialized.")

        try:
            for _ in range(self._num_workers):
                self._work_queue.put(None)

            for process in self._workers:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()

            if self._manager is not None:
                self._manager.shutdown()

            for shm in self._shms.values():
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        finally:
            self._is_initialized = False
            self._manager = None
            self._shms.clear()
            self.buffers = {}
            self.metadata_pool = None
            self._work_queue = None
            self._result_queue = None
            self._available_slots_queue = None
            self._workers.clear()
            self._num_workers = 0
            self.pool_size = 0
            self.worker_preprocess_enabled = False

    def acquire_slot(self) -> int:
        """Reserve a buffer slot for a forthcoming job."""
        if self._available_slots_queue is None:
            raise RuntimeError("Multiprocessing resources are not properly initialized.")
        return self._available_slots_queue.get()

    def submit_job(self, slot_idx: int, n_support: int | None) -> None:
        """Queue a work item for a specific slot."""
        if self._work_queue is None:
            raise RuntimeError("Multiprocessing resources are not properly initialized.")
        self._work_queue.put((slot_idx, n_support))

    def get_completed_slot(self) -> int:
        """Block until a filled slot is available."""
        if self._result_queue is None:
            raise RuntimeError("Multiprocessing resources are not properly initialized.")
        return self._result_queue.get()

    def release_slot(self, slot_idx: int) -> None:
        """Return a slot to the available pool after consumption."""
        if self._available_slots_queue is None:
            raise RuntimeError("Multiprocessing resources are not properly initialized.")
        self._available_slots_queue.put(slot_idx)


def _producer_worker(
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    shm_configs: dict[str, dict[str, Any]],
    metadata_list: list,
    worker_config: WorkerConfig,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # One per-worker Generator created POST-fork: distinct streams per worker for decorrelation
    # (replaces the old getpid()-based global np.random/random seeding).
    worker_rng = np.random.default_rng()

    tokenizer = worker_config.tokenizer
    padding = worker_config.padding
    batch_size = worker_config.batch_size
    tokenizer_oov = worker_config.tokenizer_oov
    worker_preprocess = worker_config.worker_preprocess
    max_seq_len = worker_config.max_seq_len
    prompt_config = worker_config.preprocessor_prompt_config
    unconditional_prob = worker_config.unconditional_prob

    # Each worker builds its own ProblemSource (and thus its own catalog/engine) from the picklable
    # config, driven by this worker's rng. `catalog` is reused for the preprocessor + variables.
    source = ProblemSource(worker_config.source_config, rng=worker_rng)
    catalog = source.catalog
    simplipy_engine = catalog.simplipy_engine
    variables = catalog.variables

    masked_constant_id = (int(tokenizer[MASKED_CONSTANT_TOKEN])

                          if MASKED_CONSTANT_TOKEN in tokenizer else -1)

    bos_token_id = tokenizer["<bos>"]
    eos_token_id = tokenizer["<eos>"]
    has_expression_wrappers = "<expression>" in tokenizer and "</expression>" in tokenizer

    # ieee754_mixed constants representation: serialize each <constant> occurrence per-constant
    # independently as an expanded <ieee754> hex-nibble span or a compact <float> (value on the
    # numeric channel), driven by this worker's rng.
    ieee754_start_id = int(tokenizer[IEEE754_START_TOKEN])
    ieee754_end_id = int(tokenizer[IEEE754_END_TOKEN])
    # Target dialect (contract A3): targets are the engine's TAGGED CANONICAL output,
    # produced by simplify IN the tagged dialect per problem (literals fold canonically, so
    # it cannot be cached per skeleton). 'explicit' (default) keeps today's prefix targets.
    tagged_targets = worker_config.target_dialect == TARGET_DIALECT_TAGGED
    # Task blocks: <complexity> conditioning/prediction and <predict_y> auxiliary
    # blocks (validated at dataset init: mixed constants + wrapper/block tokens present).
    # The harness owns the grammar; the model owns the content: every opener / format
    # selector / <float> value position is loss-masked (task_mask True), supervision
    # lands on content nibbles and closing tags only.
    complexity_cfg = worker_config.complexity_block
    mask_cfg = worker_config.mask_block
    predict_y_cfg = worker_config.predict_y_block
    task_blocks_on = complexity_cfg is not None or predict_y_cfg is not None or mask_cfg is not None

    if "<expression>" in tokenizer and "</expression>" not in tokenizer:
        warnings.warn(
            "Tokenizer defines '<expression>' but misses '</expression>'; training batches will omit expression terminators.",
            RuntimeWarning,
            stacklevel=2,
        )
    if "</expression>" in tokenizer and "<expression>" not in tokenizer:
        warnings.warn(
            "Tokenizer defines '</expression>' but misses '<expression>'; training batches will omit expression prefixes.",
            RuntimeWarning,
            stacklevel=2,
        )
    preprocessor: FlashANSRPreprocessor | None = None
    if worker_preprocess and prompt_config is not None:
        preprocessor = FlashANSRPreprocessor(
            simplipy_engine=simplipy_engine,
            tokenizer=tokenizer,
            catalog=catalog,
            prompt_config=prompt_config,
            rng=worker_rng,
        )

    shms = {name: shared_memory.SharedMemory(name=cfg["name"]) for name, cfg in shm_configs.items()}
    pools = {name: np.ndarray(cfg["shape"], dtype=cfg["dtype"], buffer=shms[name].buf) for name, cfg in shm_configs.items()}

    # One source iterator for the worker's lifetime: the source yields ready Problems (handling
    # skeleton sampling + support sampling internally, with the per-sample support size drawn from
    # the catalog's prior). Consecutive problems share a skeleton when problems_per_expression > 1.
    problem_iter = iter(source)

    try:
        while True:
            job = work_queue.get()
            if job is None:
                break

            # The per-job n_support is IGNORED: the source config (`n_support: prior`) governs the
            # per-sample support size. The slot index is the only field we consume here.
            slot_idx, _ = job

            x_tensors_batch = pools["x_tensors"][slot_idx]
            y_tensors_batch = pools["y_tensors"][slot_idx]
            data_attn_mask_batch = pools["data_attn_mask"][slot_idx]
            outlier_mask_batch = pools["outlier_mask"][slot_idx]
            residual_batch = pools["residual"][slot_idx]
            input_ids_batch = pools["input_ids"][slot_idx]

            constants_batch = []
            metadata_batch = []
            preprocessed_batch: list[dict[str, Any]] | None = [] if preprocessor is not None else None
            n_dropped_truncation = 0
            n_dropped_nonfinite = 0
            n_skipped_task_blocks = 0
            n_collection_restructured = 0

            i = 0
            while i < batch_size:
                problem = next(problem_iter)
                if problem.is_placeholder:
                    continue

                x_support = problem.x_support
                y_support = problem.y_support
                # The encoder fits what a model would OBSERVE: the noisy targets (identical
                # to the clean ones whenever the source ran without a noise spec). The clean
                # y_support stays local for tasks that must not learn the noise.
                y_encoder = problem.y_support_noisy
                outlier_mask = problem.outlier_mask_support

                # symbolic-data >= 0.14: generative catalogs yield CONCRETE expressions
                # (literal values inside the tokens, ``problem.constants`` empty); masking
                # is downstream policy. Reconstruct the training contract here: the
                # concrete expression, and the positionally masked skeleton with the
                # extracted values aligned 1:1 to its '<constant>' placeholders (the
                # alignment the numeric head trains on). Placeholder-style problems
                # (curated pools) still substitute their carried constants first, so both
                # problem shapes take the same path.
                expression = substitute_constants(
                    list(problem.skeleton), values=list(problem.constants), inplace=False)
                if tagged_targets:
                    # The tagged canonical differs STRUCTURALLY from the prefix canonical
                    # (rational literal spellings, negative exponents absorbed into pow,
                    # negation distributed into <sub> sections) -- see
                    # symbolic_data.token_ops.tagged_canonical. np.pi/np.e stay symbolic
                    # tokens; every NUMERIC literal is extracted for ieee754 serialization.
                    try:
                        target_expression = tagged_canonical(simplipy_engine, expression)
                        skeleton, literal_values = mask_literals_positional(
                            simplipy_engine, target_expression, keep_specials=True)
                    except NonFiniteExpressionError:
                        # Minted by OUR tagged simplify call (the producer validated the
                        # prefix form finite), so this is candidate-direction semantics:
                        # drop the instance and count it, like a span-cutting truncation.
                        n_dropped_nonfinite += 1
                        continue
                else:
                    skeleton, literal_values = mask_literals_positional(
                        simplipy_engine, expression)
                literals = np.asarray(literal_values, dtype=np.float32)

                mask_unused_variable_columns(
                    arrays=(x_support,),
                    variables=variables,
                    skeleton_tokens=skeleton,
                    padding=padding,
                )

                # Drawn BEFORE the task blocks: an unconditioned (condition-dropout) instance
                # still gets a predict_y block, but pinned to the SUFFIX so the expression is in
                # scope -- see the placement comment below. The worker rng is entropy-seeded, so
                # moving this draw ahead of serialization changes no reproducibility contract.
                condition_mask_value: bool | None = None
                if unconditional_prob > 0.0:
                    condition_mask_value = draw_condition_mask(worker_rng, unconditional_prob)

                # Promptable-mask machinery (owner rulings 2026-08-24). Every decision
                # is PER SLOT over the tagged canonical's literal sites -- exactly the
                # slots the nibble serialization fills (an exact rational spells
                # structurally and contributes one slot per literal) -- so the
                # placeheld values stay recoverable for the <predict_constants> block
                # (a collect-style re-mask would lose them).
                #
                #   flag 'all'       -> every slot placeheld (deterministic).
                #   flag 'fittable'  -> simplipy's mask_fittable per slot (deterministic).
                #   unflagged partial-> per-slot three-way draw (random harness choice).
                #
                # The flag is an EMISSION-FORMAT directive: under a flag the placeholder
                # pattern is policy-determined and the placeholders are supervised; in a
                # partial instance the pattern is a random draw, so the placeholders are
                # context-only (loss-masked below). Either half without the other would
                # train noise or unlearn masked emission.
                mask_mode: str | None = None
                partial_instance = False
                n_slots = len(literals)
                placeheld: list[bool] = [False] * n_slots
                if mask_cfg is not None:
                    mask_draw = worker_rng.random()
                    if mask_draw < float(mask_cfg["p_mask_all"]):
                        mask_mode = "all"
                        placeheld = [True] * n_slots
                    elif mask_draw < float(mask_cfg["p_mask_all"]) + float(mask_cfg["p_mask_fittable"]):
                        mask_mode = "fittable"
                        placeheld = fittable_slots(
                            simplipy_engine,
                            target_expression if tagged_targets else expression)
                        assert len(placeheld) == n_slots, "slot alignment broke"
                    elif n_slots > 0 and worker_rng.random() < float(mask_cfg["p_partial"]):
                        partial_instance = True
                        placeheld = [bool(worker_rng.random() < float(mask_cfg["p_placeheld"]))
                                     for _ in range(n_slots)]
                # COLLECTION-STABILITY check (owner ruling 2026-08-24): the engine's
                # collected mask of the drawn sites is the reference; if it is
                # token-identical to plain substitution, the site values ARE the
                # placeholder ground truth (and no group has more than one member, so
                # the block's posterior cannot be degenerate). If collection
                # RESTRUCTURES (merges, absorbs signs, fires rules -- measured ~4-6%),
                # the values are engine-internal until simplipy's value-carrying mask:
                # a flagged instance then keeps the COLLECTED expression as its
                # emission target but carries no block; a partial instance skips its
                # decoration entirely (it exists only for the block). Counted, never
                # silent.
                collected_body: list[str] | None = None
                if any(placeheld):
                    source_tokens = list(target_expression if tagged_targets else expression)
                    try:
                        collected = mask_selected_sites(
                            simplipy_engine, source_tokens, placeheld, collect=True)
                        positions = nonspecial_site_positions(simplipy_engine, source_tokens)
                        assert len(positions) == n_slots, "site alignment broke"
                        expected = list(source_tokens)
                        for pos, ph in zip(positions, placeheld):
                            if ph:
                                expected[pos] = "<constant>"
                        stable = collected == expected
                    except (NonFiniteExpressionError, ValueError):
                        stable = False
                        collected = None
                    if not stable:
                        n_collection_restructured += 1
                        if mask_mode is None:
                            # partial: undecorate -- a partial instance without a
                            # sound block trains nothing.
                            partial_instance = False
                            placeheld = [False] * n_slots
                        elif collected is not None:
                            collected_body = list(collected)
                        else:
                            # the engine refused the mask outright: fall back unmasked.
                            mask_mode = None
                            placeheld = [False] * n_slots
                placeheld_values = ([float(v) for v, ph in zip(literals, placeheld) if ph]
                                    if collected_body is None else [])

                if collected_body is not None:
                    # Restructured flagged target: the engine's collected output IS
                    # the emission format. mask_literals_positional is 1:1 and
                    # leaves the engine's <constant> placeholders alone, so
                    # position p tells the two kinds apart: a value slot (was a
                    # kept literal) vs a placeholder (stays, as a None entry).
                    try:
                        skeleton_mm, kept = mask_literals_positional(
                            simplipy_engine, collected_body, keep_specials=True)
                    except (NonFiniteExpressionError, ValueError):
                        # A collected body the extractor refuses (a masked fold
                        # minting a non-finite spelling): fall back unmasked
                        # rather than kill the worker (audit 2026-08-24).
                        collected_body = None
                        mask_mode = None
                        placeheld = [False] * n_slots
                if collected_body is not None:
                    kept_iter = iter(kept)
                    collected_opt: list[float | None] = []
                    for original, slot in zip(collected_body, skeleton_mm):
                        if slot == "<constant>":
                            collected_opt.append(None if original == "<constant>"
                                                 else float(next(kept_iter)))
                    assert next(kept_iter, None) is None, "positional value alignment broke"
                    serialized_tokens, body_numeric = serialize_constant_tokens(
                        skeleton_mm, collected_opt)
                elif any(placeheld):
                    # Placeholders ARE simplipy's <constant>: the serializer's
                    # None entries keep them, value entries fill the kept slots.
                    constants_opt: list[float | None] = [
                        None if ph else float(v) for v, ph in zip(literals, placeheld)]
                    serialized_tokens, body_numeric = serialize_constant_tokens(
                        skeleton, constants_opt)
                else:
                    # Raises on non-finite constants: the generator must never emit them.
                    serialized_tokens, body_numeric = serialize_constant_tokens(
                        skeleton, literals)
                tokens_to_encode = serialized_tokens
                if has_expression_wrappers:
                    tokens_to_encode = ["<expression>", *tokens_to_encode, "</expression>"]
                    if body_numeric is not None:
                        body_numeric = [float("nan"), *body_numeric, float("nan")]

                # ---- v24 task blocks ------------------------------------------------------
                # Attached only within the sequence budget: an over-budget block is SKIPPED
                # (counted, never silent) rather than dropping the instance, so the blocks
                # add ZERO truncation pressure -- the expression prior's truncation shaping
                # stays exactly what it is without the features.
                complexity_draw: dict[str, Any] | None = None
                predict_y_draw: dict[str, Any] | None = None
                task_mask: list[bool] | None = None
                task_segments: list[int] | None = None
                block_order: dict[str, list[str]] | None = None
                predict_constants_draw: dict[str, Any] | None = None
                if task_blocks_on:
                    budget = max_seq_len - (len(tokens_to_encode) + 2)  # <bos> ... <eos>
                    # Blocks are built as ELEMENTS and assembled at the end (owner
                    # ruling 2026-08-24): truly commutative prefix elements are PERMUTED
                    # per instance so none welds to a position; the hypothesis element is
                    # pinned LAST (from the flag on, the pen is the model's until
                    # <expression>); the two suffix blocks swap 50/50. The drawn order is
                    # recorded in metadata for later splits.
                    # element = (name, tokens, numeric, masked, segments)
                    prefix_elements: list[tuple] = []
                    hypothesis_element: tuple | None = None
                    suffix_elements: list[tuple] = []

                    if mask_mode is not None and budget < 1:
                        # No room for even the flag (audit 2026-08-24): emitting the
                        # masked body flag-less would violate flag <=> format, and
                        # letting truncation cut the wrapper emitted broken rows.
                        n_dropped_truncation += 1
                        continue
                    if mask_mode is not None:
                        # The emission-format directive: harness-owned, never supervised.
                        prefix_elements.append(("mask_flag", [MASK_MODE_TOKENS[mask_mode]],
                                                [float("nan")], [True],
                                                [TASK_SEGMENT_EXPRESSION]))
                        budget -= 1

                    complexity_mode = None
                    if complexity_cfg is not None:
                        # Three-way instance draw (priors pinned in the config): hypothesis
                        # mode / prompted block / absent.
                        mode_draw = worker_rng.random()
                        if mode_draw < float(complexity_cfg["p_hypothesize"]):
                            complexity_mode = "hypothesis"
                        elif mode_draw < float(complexity_cfg["p_hypothesize"]) + float(complexity_cfg["p_present"]):
                            complexity_mode = "prompted"
                    if complexity_mode is not None:
                        assert complexity_cfg is not None  # complexity_mode implies the config
                        # mu of the MASKED target (a <constant> prices one symbol unit): the
                        # only complexity a user can state at inference without knowing the
                        # constants. complexity() measures the canonical form, so the target
                        # dialect does not matter. Exact in float32 (mu < 2**24 in practice).
                        mu = int(simplipy_engine.complexity(list(skeleton)))
                        if complexity_mode == "hypothesis":
                            # The harness-inserted flag LICENSES self-initiated property
                            # blocks: the flag itself is never supervised (only the harness
                            # may utter it), but everything after it -- opener, format
                            # selector, nibbles, closers -- is the model's own hypothesis
                            # and carries loss.
                            # The flag is NOT part of this block: it is the BOUNDARY, emitted
                            # once (below) after every given element. Everything from it to
                            # </expression> is the model's, so the opener carries loss too.
                            block_tokens = [COMPLEXITY_START_TOKEN, IEEE754_START_TOKEN,
                                            *float32_to_nibble_tokens(float(mu)),
                                            IEEE754_END_TOKEN, COMPLEXITY_END_TOKEN]
                            block_numeric = [float("nan")] * len(block_tokens)
                            block_masked = [False] * len(block_tokens)
                            variant = "hypothesis"
                        else:
                            # PROMPTED: the caller states the complexity, so it is compact and
                            # entirely harness-owned. The bytes spelling belongs to the
                            # hypothesis circumstance alone (owner ruling 2026-08-27).
                            block_tokens = [COMPLEXITY_START_TOKEN, COMPACT_CONSTANT_TOKEN, COMPLEXITY_END_TOKEN]
                            block_numeric = [float("nan"), float(mu), float("nan")]
                            block_masked = [True, True, True]
                            variant = "float"
                        # +1 for the standalone <hypothesize> boundary this variant needs.
                        need = len(block_tokens) + (1 if variant == "hypothesis" else 0)
                        if need <= budget:
                            budget -= need
                            element = ("complexity", block_tokens, block_numeric, block_masked,
                                       [TASK_SEGMENT_COMPLEXITY] * len(block_tokens))
                            if variant == "hypothesis":
                                hypothesis_element = element
                            else:
                                prefix_elements.append(element)
                            complexity_draw = {"mu": mu, "variant": variant}
                        else:
                            n_skipped_task_blocks += 1

                    if (predict_y_cfg is not None
                            and x_support.shape[0] >= int(predict_y_cfg["min_n_support"])
                            and worker_rng.random() < float(predict_y_cfg["p_present"])):
                        n_dims = x_support.shape[1]
                        if 4 + n_dims + IEEE754_SPAN_LENGTH <= budget:
                            # Prior-exactness: the held-out point is one of the ALREADY-ACCEPTED
                            # support rows (box acceptance is untouched), never an extra draw;
                            # y* is the CLEAN value -- the task supervises the function, not the
                            # noise. Full-precision nibbles (v24 ruling: no tail zeroing).
                            j = int(worker_rng.integers(x_support.shape[0]))
                            point = x_support[j].astype(NUMERIC_DTYPE_NP)
                            y_star = float(np.float32(y_support[j].reshape(-1)[0]))
                            # Placement decides WHAT the model may condition on:
                            #   suffix (after </expression>) -> the data AND the expression;
                            #   prefix (before <expression>) -> the data alone.
                            # On an UNCONDITIONED instance the encoder memory is nulled, so the
                            # suffix form becomes "evaluate this expression at x*" -- function
                            # evaluation, a well-posed task that grounds the expression tokens
                            # semantically. Only the PREFIX form would be ill-posed there (nulled
                            # memory and no expression = nothing to condition on), so the block is
                            # pinned to the suffix rather than dropped (owner ruling 2026-08-26,
                            # overturning the earlier blanket exclusion).
                            conditional = (
                                True if condition_mask_value is False
                                else bool(worker_rng.random() < float(predict_y_cfg["p_conditional"])))
                            x_support = np.delete(x_support, j, axis=0)
                            y_support = np.delete(y_support, j, axis=0)
                            y_encoder = np.delete(y_encoder, j, axis=0)
                            if outlier_mask is not None:
                                outlier_mask = np.delete(outlier_mask, j, axis=0)
                            block_tokens = [PREDICT_Y_START_TOKEN, POINT_START_TOKEN]
                            block_numeric = [float("nan"), float("nan")]
                            block_masked = [True, True]
                            for value in point:
                                block_tokens.append(COMPACT_CONSTANT_TOKEN)
                                block_numeric.append(float(value))
                                block_masked.append(True)
                            block_tokens += [POINT_END_TOKEN, IEEE754_START_TOKEN,
                                             *float32_to_nibble_tokens(y_star),
                                             IEEE754_END_TOKEN, PREDICT_Y_END_TOKEN]
                            block_numeric += [float("nan")] * (4 + IEEE754_N_NIBBLES)
                            block_masked += [True, True, *[False] * IEEE754_N_NIBBLES, False, False]
                            budget -= len(block_tokens)
                            element = ("predict_y", block_tokens, block_numeric, block_masked,
                                       [TASK_SEGMENT_PREDICT_Y] * len(block_tokens))
                            if conditional:
                                suffix_elements.append(element)
                            else:
                                prefix_elements.append(element)
                            predict_y_draw = {"x": point.tolist(), "y": y_star, "conditional": conditional}
                        else:
                            n_skipped_task_blocks += 1

                    # <predict_constants> (owner rulings 2026-08-24): conditional on >= 1
                    # placeholder, p < 1 in both circumstances so both the plain ending and
                    # the harness-opened continuation stay in-distribution. One span per
                    # placeholder, POSITIONAL order -- the binding needs no indices, and a
                    # fixed constant is simply spelled inline in the expression instead.
                    # Loss discipline as everywhere: openers force-fed, nibbles and closing
                    # tags are the model's.
                    if (placeheld_values and mask_cfg is not None
                            and condition_mask_value is not False):
                        # Same doctrine as predict_y (audit 2026-08-24): supervising
                        # constant values against a NULLED memory is a nonsense task.
                        p_block = float(mask_cfg["p_predict_constants_flagged"]
                                        if mask_mode is not None
                                        else mask_cfg["p_predict_constants_partial"])
                        if worker_rng.random() < p_block:
                            need = 2 + len(placeheld_values) * IEEE754_SPAN_LENGTH
                            if need <= budget:
                                budget -= need
                                block_tokens = [PREDICT_CONSTANTS_START_TOKEN]
                                block_numeric = [float("nan")]
                                block_masked = [True]
                                for value in placeheld_values:
                                    block_tokens += [IEEE754_START_TOKEN,
                                                     *float32_to_nibble_tokens(float(value)),
                                                     IEEE754_END_TOKEN]
                                    block_numeric += [float("nan")] * IEEE754_SPAN_LENGTH
                                    block_masked += [True, *[False] * IEEE754_N_NIBBLES, False]
                                block_tokens.append(PREDICT_CONSTANTS_END_TOKEN)
                                block_numeric.append(float("nan"))
                                block_masked.append(False)
                                suffix_elements.append((
                                    "predict_constants", block_tokens, block_numeric, block_masked,
                                    [TASK_SEGMENT_PREDICT_CONSTANTS] * len(block_tokens)))
                                predict_constants_draw = {"values": list(placeheld_values)}
                            else:
                                n_skipped_task_blocks += 1

                    if len(prefix_elements) > 1:
                        prefix_elements = [prefix_elements[int(k)]
                                           for k in worker_rng.permutation(len(prefix_elements))]
                    if hypothesis_element is not None:
                        # THE BOUNDARY (owner ruling 2026-08-27). Everything before it is
                        # given -- fixed, compact, harness-owned -- and may not recur after
                        # it; everything after it is the model's, spelled in bits. It is a
                        # marker of its own, not part of any block, so it carries no task
                        # segment: with a second hypothesizable property the flag is uttered
                        # ONCE and licenses the whole run that follows.
                        prefix_elements.append(("hypothesize", [HYPOTHESIS_TOKEN], [float("nan")],
                                                [True], [TASK_SEGMENT_EXPRESSION]))
                        prefix_elements.append(hypothesis_element)
                    if len(suffix_elements) > 1 and worker_rng.random() < 0.5:
                        suffix_elements.reverse()
                    block_order = {"prefix": [e[0] for e in prefix_elements],
                                   "suffix": [e[0] for e in suffix_elements]}

                    prefix_tokens = [t for e in prefix_elements for t in e[1]]
                    prefix_numeric = [v for e in prefix_elements for v in e[2]]
                    prefix_masked = [m for e in prefix_elements for m in e[3]]
                    prefix_segments = [g for e in prefix_elements for g in e[4]]
                    suffix_tokens = [t for e in suffix_elements for t in e[1]]
                    suffix_numeric = [v for e in suffix_elements for v in e[2]]
                    suffix_masked = [m for e in suffix_elements for m in e[3]]
                    suffix_segments = [g for e in suffix_elements for g in e[4]]

                    if prefix_tokens or suffix_tokens or partial_instance:
                        base_numeric = (body_numeric if body_numeric is not None
                                        else [float("nan")] * len(tokens_to_encode))
                        # Flag-dependent placeholder loss (owner ruling): under a flag the
                        # pattern is policy-determined and the placeholders stay
                        # supervised; in a partial instance they are a random harness
                        # draw the model cannot predict -- context-only, loss-masked.
                        body_masked = ([token == MASKED_CONSTANT_TOKEN for token in tokens_to_encode]
                                       if partial_instance else [False] * len(tokens_to_encode))
                        task_mask = [False, *prefix_masked, *body_masked,
                                     *suffix_masked, False]
                        task_segments = [0, *prefix_segments, *[0] * len(tokens_to_encode),
                                         *suffix_segments, 0]
                        tokens_to_encode = [*prefix_tokens, *tokens_to_encode, *suffix_tokens]
                        body_numeric = [*prefix_numeric, *base_numeric, *suffix_numeric]

                # Numeric channel aligned with the FINAL input_ids ([<bos>, ..., <eos>]): values
                # only at compact <float> positions; merged downstream by ensure_numeric_channel.
                input_num = None if body_numeric is None else [float("nan"), *body_numeric, float("nan")]

                body_ids = tokenizer.encode(tokens_to_encode, oov=tokenizer_oov)
                input_ids = [bos_token_id, *body_ids, eos_token_id]
                if len(input_ids) > max_seq_len:
                    if truncation_cuts_ieee754_span(
                            input_ids, max_seq_len, ieee754_start_id, ieee754_end_id):
                        # Truncation must never cut inside an <ieee754> span: drop the
                        # instance (like other rejected samples) and count it.
                        n_dropped_truncation += 1
                        continue
                    input_ids = input_ids[:max_seq_len]
                    input_ids[-1] = eos_token_id
                    if input_num is not None:
                        input_num = input_num[:max_seq_len]
                        input_num[-1] = float("nan")
                    # The task channels track input_ids (audit 2026-08-24: they were
                    # stored at pre-truncation length and silently clipped downstream).
                    if task_mask is not None:
                        task_mask = task_mask[:max_seq_len]
                        task_mask[-1] = False
                    if task_segments is not None:
                        task_segments = task_segments[:max_seq_len]
                        task_segments[-1] = 0

                metadata = {
                    "skeleton": skeleton,
                    "skeleton_hash": tuple(skeleton),
                    "expression": expression,
                    "n_support": int(x_support.shape[0]),
                }
                if isinstance(problem.noise, dict):
                    metadata["noise"] = problem.noise
                # First-class optional condition (CFG): ONLY when enabled (prob > 0), mark this
                # example conditioned (True, prob 1 - unconditional_prob) or unconditioned (False).
                # The key is emitted iff the feature is active, so condition_mask present <=> feature
                # on -> existing runs (prob 0) are byte-identical and the model never sees a mask.
                # Flows into the batch via `metadata_fields` (data.py) and routes to `null_memory`
                # in the model when False. Per-worker RNG is seeded at worker start.
                if condition_mask_value is not None:
                    metadata["condition_mask"] = condition_mask_value

                # Task-block metadata: keys present <=> feature configured (T0 contract), and
                # UNIFORM across the batch (the consumer builds columns from entry 0's keys).
                if task_blocks_on:
                    metadata["task_mask"] = (task_mask if task_mask is not None
                                             else [False] * len(input_ids))
                    # 0 = expression/other, 1 = complexity, 2 = predict_y: the per-position
                    # channel the trainer splits the CE by (per-task wandb curves).
                    metadata["task_segments"] = (task_segments if task_segments is not None
                                                 else [0] * len(input_ids))
                if complexity_cfg is not None:
                    metadata["complexity_mu"] = None if complexity_draw is None else complexity_draw["mu"]
                    metadata["complexity_variant"] = None if complexity_draw is None else complexity_draw["variant"]
                if predict_y_cfg is not None:
                    metadata["predict_y"] = predict_y_draw
                if mask_cfg is not None:
                    metadata["mask_mode"] = mask_mode
                    metadata["n_placeholders"] = int(sum(
                        1 for token_id in input_ids if token_id == masked_constant_id))
                    metadata["predict_constants"] = predict_constants_draw
                if task_blocks_on:
                    metadata["block_order"] = (block_order if block_order is not None
                                               else {"prefix": [], "suffix": []})

                # Mixed representation only (key present <=> feature on, like condition_mask):
                # the per-token numeric channel computed during serialization. AUTHORITATIVE:
                # ensure_numeric_channel passes it through verbatim (audit 2026-08-24 -- a
                # recompute would put ground-truth values at masked placeholder positions).
                if input_num is not None:
                    metadata["input_num"] = input_num

                x_tensors_batch[i, : x_support.shape[0], : x_support.shape[1]] = x_support
                x_tensors_batch[i, x_support.shape[0]:, :] = 0

                y_tensors_batch[i, : y_encoder.shape[0], : y_encoder.shape[1]] = y_encoder
                y_tensors_batch[i, y_encoder.shape[0]:, :] = 0

                outlier_mask_batch[i, : y_encoder.shape[0]] = (
                    outlier_mask.reshape(-1).astype(np.float32) if outlier_mask is not None else 0)
                outlier_mask_batch[i, y_encoder.shape[0]:] = 0

                # The residual the model would have to explain: observed minus truth, in the
                # dtype the encoder actually reads (v25: binary64). Differencing in the ENCODER's
                # dtype -- not a wider one -- is the point: the target is exactly the displacement
                # present in the data the model sees, cancellation and all. See y_encoder above
                # for why the clean array stays local.
                residual_batch[i, : y_encoder.shape[0]] = (
                    y_encoder.reshape(-1).astype(NUMERIC_DTYPE_NP)
                    - y_support.reshape(-1).astype(NUMERIC_DTYPE_NP))
                residual_batch[i, y_encoder.shape[0]:] = 0

                data_attn_mask_batch[i, : x_support.shape[0]] = 1
                data_attn_mask_batch[i, x_support.shape[0]:] = 0

                input_ids_batch[i, :] = tokenizer["<pad>"]
                input_ids_batch[i, : len(input_ids)] = input_ids

                constants_batch.append(literals)
                metadata_batch.append(metadata)
                if preprocessed_batch is not None and preprocessor is not None:
                    instance = {
                        "input_ids": list(input_ids),
                        "skeletons": list(metadata.get("skeleton", [])),
                    }
                    preprocessed_batch.append(preprocessor._format_single(instance))

                i += 1
            payload: dict[str, Any] = {"metadata": metadata_batch, "constants": constants_batch}
            if task_blocks_on:
                payload["n_skipped_task_blocks"] = n_skipped_task_blocks
                payload["n_collection_restructured"] = n_collection_restructured
            if preprocessed_batch is not None:
                payload["preprocessed"] = preprocessed_batch
            # Instances dropped because truncation would have cut inside an <ieee754> span
            # while filling THIS batch.
            payload["n_dropped_truncation"] = n_dropped_truncation
            if tagged_targets:
                # Instances whose tagged canonicalization folded a degenerate sub-expression
                # to a non-finite spelling while filling THIS batch (tagged targets only).
                payload["n_dropped_nonfinite"] = n_dropped_nonfinite
            metadata_list[slot_idx] = payload
            result_queue.put(slot_idx)
    except Exception as exc:  # noqa: BLE001 - a dead worker must not become a silent hang
        # Without this the loop was `try: while True: ... finally: shm.close()` with NO except.
        # Any raise from serialization killed the worker, and the pool then blocked forever in
        # FlashANSRDataset.get_completed_slot() with no error message anywhere -- indistinguishable
        # from slow generation. Report the slot as failed so the consumer sees a cause, then let
        # the worker die; the pool's own supervision decides what to do about a missing producer.
        import traceback
        try:
            metadata_list[slot_idx] = {
                "worker_error": f"{type(exc).__name__}: {exc}",
                "worker_traceback": traceback.format_exc(),
            }
            result_queue.put(slot_idx)
        except Exception:  # noqa: BLE001 - the slot index may not exist yet; nothing left to do
            pass
        raise
    finally:
        for shm in shms.values():
            shm.close()

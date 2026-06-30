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
from simplipy.utils import substitude_constants as substitute_constants
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing import FlashANSRPreprocessor
from flash_ansr.utils.tensor_ops import mask_unused_variable_columns


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


class SharedMemoryWorkerPool:
    """Manage worker processes that stream samples into shared memory."""

    def __init__(
        self,
        *,
        source: ProblemSource,
        tokenizer: Tokenizer,
        padding: Literal["random", "zero"],
    ) -> None:
        self.source = source
        self.tokenizer = tokenizer
        self.padding = padding

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
                "dtype": np.float32,
            },
            "y_tensors": {
                "shape": (self.pool_size, batch_size, max_n_support, 1),
                "dtype": np.float32,
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
    variables = catalog.variables

    bos_token_id = tokenizer["<bos>"]
    eos_token_id = tokenizer["<eos>"]
    has_expression_wrappers = "<expression>" in tokenizer and "</expression>" in tokenizer

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
            simplipy_engine=catalog.simplipy_engine,
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
            input_ids_batch = pools["input_ids"][slot_idx]

            constants_batch = []
            metadata_batch = []
            preprocessed_batch: list[dict[str, Any]] | None = [] if preprocessor is not None else None

            i = 0
            while i < batch_size:
                problem = next(problem_iter)
                if problem.is_placeholder:
                    continue

                x_support = problem.x_support
                y_support = problem.y_support
                skeleton = list(problem.skeleton)
                literals = np.asarray(problem.constants, dtype=np.float32)

                mask_unused_variable_columns(
                    arrays=(x_support,),
                    variables=variables,
                    skeleton_tokens=skeleton,
                    padding=padding,
                )

                tokens_to_encode = list(skeleton)
                if has_expression_wrappers:
                    tokens_to_encode = ["<expression>", *tokens_to_encode, "</expression>"]

                body_ids = tokenizer.encode(tokens_to_encode, oov=tokenizer_oov)
                input_ids = [bos_token_id, *body_ids, eos_token_id]
                if len(input_ids) > max_seq_len:
                    input_ids = input_ids[:max_seq_len]
                    input_ids[-1] = eos_token_id

                metadata = {
                    "skeleton": skeleton,
                    "skeleton_hash": tuple(skeleton),
                    "expression": substitute_constants(skeleton, values=literals, inplace=False),
                    "n_support": int(x_support.shape[0]),
                }
                # First-class optional condition (CFG): ONLY when enabled (prob > 0), mark this
                # example conditioned (True, prob 1 - unconditional_prob) or unconditioned (False).
                # The key is emitted iff the feature is active, so condition_mask present <=> feature
                # on -> existing runs (prob 0) are byte-identical and the model never sees a mask.
                # Flows into the batch via `metadata_fields` (data.py) and routes to `null_memory`
                # in the model when False. Per-worker RNG is seeded at worker start.
                if unconditional_prob > 0.0:
                    metadata["condition_mask"] = bool(worker_rng.random() >= unconditional_prob)

                x_tensors_batch[i, : x_support.shape[0], : x_support.shape[1]] = x_support
                x_tensors_batch[i, x_support.shape[0]:, :] = 0

                y_tensors_batch[i, : y_support.shape[0], : y_support.shape[1]] = y_support
                y_tensors_batch[i, y_support.shape[0]:, :] = 0

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
            if preprocessed_batch is not None:
                payload["preprocessed"] = preprocessed_batch
            metadata_list[slot_idx] = payload
            result_queue.put(slot_idx)
    finally:
        for shm in shms.values():
            shm.close()

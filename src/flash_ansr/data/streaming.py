"""Shared-memory streaming of procedurally generated training samples."""

import multiprocessing as mp
import os
import random
import signal
import warnings
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.managers import ListProxy, SyncManager
from typing import Any, Literal

import numpy as np

from flash_ansr.expressions import SkeletonPool, NoValidSampleFoundError
from flash_ansr.expressions.token_ops import substitute_constants
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing import FlashANSRPreprocessor
from flash_ansr.utils.tensor_ops import mask_unused_variable_columns


@dataclass
class WorkerConfig:
    """Configuration passed to worker processes generating samples."""
    skeleton_pool: SkeletonPool
    tokenizer: Tokenizer
    padding: Literal["random", "zero"]
    n_per_equation: int
    batch_size: int
    tokenizer_oov: Literal["unk", "raise"]
    worker_preprocess: bool
    max_seq_len: int
    preprocessor_prompt_config: dict[str, Any] | None


class SharedMemoryWorkerPool:
    """Manage worker processes that stream samples into shared memory."""

    def __init__(
        self,
        *,
        skeleton_pool: SkeletonPool,
        tokenizer: Tokenizer,
        padding: Literal["random", "zero"],
    ) -> None:
        self.skeleton_pool = skeleton_pool
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
    ) -> None:
        """Allocate shared buffers and spin up producer workers."""
        if self._is_initialized:
            return

        self.worker_preprocess_enabled = worker_preprocess
        self._num_workers = os.cpu_count() or 1 if num_workers is None else num_workers
        self.pool_size = self._num_workers * prefetch_factor

        if max_n_support is None:
            max_n_support = self.skeleton_pool.support_sampler.configured_max_n_support
            if max_n_support is None:
                raise ValueError(
                    "Support sampler configuration must define a maximum support size via "
                    "'n_support_prior.kwargs.max_value' or an equivalent field."
                )

        shm_configs: dict[str, dict[str, Any]] = {
            "x_tensors": {
                "shape": (self.pool_size, batch_size, max_n_support, len(self.skeleton_pool.variables)),
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

        worker_config = WorkerConfig(
            skeleton_pool=self.skeleton_pool,
            tokenizer=self.tokenizer,
            padding=self.padding,
            n_per_equation=n_per_equation,
            batch_size=batch_size,
            tokenizer_oov=tokenizer_oov,
            worker_preprocess=worker_preprocess,
            max_seq_len=max_seq_len,
            preprocessor_prompt_config=preprocessor_prompt_config,
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
    np.random.seed(os.getpid())
    random.seed(os.getpid())

    skeleton_pool = worker_config.skeleton_pool
    tokenizer = worker_config.tokenizer
    padding = worker_config.padding
    n_per_equation = worker_config.n_per_equation
    batch_size = worker_config.batch_size
    tokenizer_oov = worker_config.tokenizer_oov
    worker_preprocess = worker_config.worker_preprocess
    max_seq_len = worker_config.max_seq_len
    prompt_config = worker_config.preprocessor_prompt_config

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
            simplipy_engine=skeleton_pool.simplipy_engine,
            tokenizer=tokenizer,
            skeleton_pool=skeleton_pool,
            prompt_config=prompt_config,
        )

    shms = {name: shared_memory.SharedMemory(name=cfg["name"]) for name, cfg in shm_configs.items()}
    pools = {name: np.ndarray(cfg["shape"], dtype=cfg["dtype"], buffer=shms[name].buf) for name, cfg in shm_configs.items()}

    try:
        while True:
            job = work_queue.get()
            if job is None:
                break

            slot_idx, n_support = job

            x_tensors_batch = pools["x_tensors"][slot_idx]
            y_tensors_batch = pools["y_tensors"][slot_idx]
            data_attn_mask_batch = pools["data_attn_mask"][slot_idx]
            input_ids_batch = pools["input_ids"][slot_idx]

            constants_batch = []
            metadata_batch = []
            preprocessed_batch: list[dict[str, Any]] | None = [] if preprocessor is not None else None

            i = 0
            while i < batch_size:
                try:
                    skeleton_hash, skeleton_code, skeleton_constants = skeleton_pool.sample_skeleton()
                    skeleton = list(skeleton_hash)
                except NoValidSampleFoundError:
                    continue

                temp_samples = []
                attempts = 0
                max_total_attempts = n_per_equation * 20

                succeeded = True
                n_to_generate = min(n_per_equation, batch_size - i)

                for _ in range(n_to_generate):
                    sample_found = False
                    while not sample_found:
                        if attempts >= max_total_attempts:
                            succeeded = False
                            break

                        attempts += 1
                        try:
                            x_support, y_support, literals = skeleton_pool.sample_data(
                                skeleton_code, len(skeleton_constants), n_support=n_support
                            )

                            mask_unused_variable_columns(
                                arrays=(x_support,),
                                variables=skeleton_pool.variables,
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

                            temp_samples.append(
                                {
                                    "x": x_support,
                                    "y": y_support,
                                    "input_ids": input_ids,
                                    "constants": literals,
                                    "metadata": {
                                        "skeleton": skeleton,
                                        "skeleton_hash": skeleton_hash,
                                        "expression": substitute_constants(skeleton, values=literals, inplace=False),
                                        "n_support": int(x_support.shape[0]),
                                    },
                                }
                            )
                            sample_found = True
                        except NoValidSampleFoundError:
                            continue

                    if not succeeded:
                        break

                if succeeded:
                    for sample in temp_samples:
                        x_tensors_batch[i, : sample["x"].shape[0], : sample["x"].shape[1]] = sample["x"]
                        x_tensors_batch[i, sample["x"].shape[0]:, :] = 0

                        y_tensors_batch[i, : sample["y"].shape[0], : sample["y"].shape[1]] = sample["y"]
                        y_tensors_batch[i, sample["y"].shape[0]:, :] = 0

                        data_attn_mask_batch[i, : sample["x"].shape[0]] = 1
                        data_attn_mask_batch[i, sample["x"].shape[0]:] = 0

                        input_ids_batch[i, :] = tokenizer["<pad>"]
                        input_ids_batch[i, : len(sample["input_ids"])] = sample["input_ids"]

                        constants_batch.append(sample["constants"])
                        metadata_batch.append(sample["metadata"])
                        if preprocessed_batch is not None and preprocessor is not None:
                            instance = {
                                "input_ids": list(sample["input_ids"]),
                                "skeletons": list(sample["metadata"].get("skeleton", [])),
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

import copy
import os
import time
import warnings
import types
from typing import Any, Generator, Literal

import numpy as np
import torch
from datasets import Dataset, disable_progress_bars, load_from_disk
from simplipy import SimpliPyEngine
from tqdm import tqdm

from flash_ansr.data.collate import BatchFormatter
from flash_ansr.data.streaming import SharedMemoryWorkerPool
from flash_ansr.expressions import SkeletonPool
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing import FlashANSRPreprocessor
from flash_ansr.utils.config_io import load_config, save_config
from flash_ansr.utils.paths import substitute_root_path


class FlashANSRDataset:
    """Dataset wrapper for amortized neural symbolic regression training."""

    def __init__(
        self,
        skeleton_pool: SkeletonPool,
        tokenizer: Tokenizer,
        padding: Literal["random", "zero"],
        preprocessor: FlashANSRPreprocessor | None = None,
    ) -> None:
        self.skeleton_pool = skeleton_pool
        self.tokenizer = tokenizer
        self.padding = padding
        self.preprocessor = preprocessor
        self.data = None

        self._collator = BatchFormatter(tokenizer=tokenizer)
        self._stream = SharedMemoryWorkerPool(
            skeleton_pool=skeleton_pool,
            tokenizer=tokenizer,
            padding=padding,
        )
        self._preprocessor_prompt_config = (
            copy.deepcopy(preprocessor.prompt_config) if preprocessor is not None else None
        )

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        if self._stream.is_initialized:
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
        return self.skeleton_pool.simplipy_engine

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "FlashANSRDataset":
        config_ = load_config(config)

        if "dataset" in config_.keys():
            config_ = config_["dataset"]

        if isinstance(config, str) and isinstance(config_["skeleton_pool"], str):
            if config_["skeleton_pool"].startswith('.'):  # pragma: no cover - config guard
                config_["skeleton_pool"] = os.path.join(os.path.dirname(config), config_["skeleton_pool"])
            config_["skeleton_pool"] = substitute_root_path(config_["skeleton_pool"])

        if os.path.isfile(config_["skeleton_pool"]) or isinstance(config_["skeleton_pool"], dict):
            skeleton_pool = SkeletonPool.from_config(config_["skeleton_pool"])
        elif os.path.isdir(config_["skeleton_pool"]):
            skeleton_pool = SkeletonPool.load(config_["skeleton_pool"])[1]
        else:
            raise ValueError(f"Invalid skeleton pool configuration: {config_['skeleton_pool']}")

        tokenizer = Tokenizer.from_config(config_["tokenizer"])

        preprocessor_cfg = config_.get("preprocessor") if isinstance(config_, dict) else None
        preprocessor: FlashANSRPreprocessor | None = None
        if preprocessor_cfg is not None:
            preprocessor = FlashANSRPreprocessor.from_config(
                preprocessor_cfg,
                simplipy_engine=skeleton_pool.simplipy_engine,
                tokenizer=tokenizer,
                skeleton_pool=skeleton_pool,
            )

        return cls(
            skeleton_pool=skeleton_pool,
            tokenizer=tokenizer,
            padding=config_["padding"],
            preprocessor=preprocessor,
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
        if self.data is None:
            raise ValueError("No dataset to save. Please generate or load a dataset first.")

        directory = substitute_root_path(directory)
        os.makedirs(directory, exist_ok=True)

        self.data.save_to_disk(dataset_path=os.path.join(directory, "dataset"), *args, **kwargs)

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
        config_path = os.path.join(directory, "dataset.yaml")
        resolved_directory = substitute_root_path(directory)

        dataset = cls.from_config(config_path)
        dataset.data = load_from_disk(os.path.join(resolved_directory, "dataset"))

        return load_config(config_path), dataset

    def collate(self, batch: dict[str, Any], device: str | torch.device | int = "cpu") -> dict[str, Any]:
        return self._collator.collate(batch, device=device)

    def compile(
        self,
        size: int | None = None,
        steps: int | None = None,
        batch_size: int | None = None,
        n_support: int | None = None,
        verbose: bool = False,
    ) -> None:
        disable_progress_bars()
        if size is None and steps is None:
            size = len(self.skeleton_pool)

        self.data = Dataset.from_list(
            list(
                self.iterate(
                    size=size,
                    steps=steps,
                    batch_size=batch_size,
                    n_support=n_support,
                    verbose=verbose,
                )
            )
        )

    @staticmethod
    def _inject_preprocessed_fields(batch: dict[str, Any], samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        for key in samples[0].keys():
            batch[key] = [sample[key] for sample in samples]

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
        )

    def shutdown(self) -> None:
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
        tokenizer_oov: Literal["unk", "raise"] = "raise",
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        persistent: bool = False,
        tqdm_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        if batch_size is None:
            batch_size = 1

        tqdm_kwargs = dict(tqdm_kwargs) if tqdm_kwargs else {}

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

        self._initialize_stream(
            prefetch_factor=prefetch_factor,
            batch_size=batch_size,
            n_per_equation=n_per_equation,
            max_seq_len=max_seq_len,
            max_n_support=max_n_support,
            num_workers=num_workers,
            tokenizer_oov=tokenizer_oov,
            worker_preprocess=use_worker_preprocess,
        )

        if self._stream.metadata_pool is None or not self._stream.buffers:
            raise RuntimeError("Multiprocessing resources are not properly initialized.")

        pool_size = self._stream.pool_size

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
                completed_slot_idx = self._stream.get_completed_slot()
                metadata_and_constants = self._stream.metadata_pool[completed_slot_idx]
                if metadata_and_constants is None:
                    raise RuntimeError("Worker returned empty payload.")

                metadata_batch = metadata_and_constants["metadata"]
                metadata_fields: dict[str, list[Any]] = {}
                if metadata_batch:
                    for key in metadata_batch[0]:
                        metadata_fields[key] = [entry[key] for entry in metadata_batch]

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
        finally:
            pbar.close()
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

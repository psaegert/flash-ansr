import os
import warnings
import time
from typing import Any, Generator, Literal
import signal
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.managers import SyncManager, ListProxy

import torch
import numpy as np

from tqdm import tqdm
from datasets import Dataset, load_from_disk, disable_progress_bars

from simplipy import SimpliPyEngine

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils import load_config, save_config, substitute_root_path
from flash_ansr.expressions import SkeletonPool, NoValidSampleFoundError
from flash_ansr.expressions.utils import substitude_constants
from flash_ansr.preprocess import FlashASNRPreprocessor


class FlashANSRDataset:
    '''
    Dataset for amortized neural symbolic regression training.
    '''
    def __init__(self, skeleton_pool: SkeletonPool, tokenizer: Tokenizer, padding: Literal['random', 'zero'], preprocessor: FlashASNRPreprocessor | None = None) -> None:
        self.skeleton_pool = skeleton_pool
        self.tokenizer = tokenizer
        self.padding = padding
        self.preprocessor = preprocessor
        self.data = None

        # --- ADDED: Attributes for persistent resources ---
        self._is_initialized = False
        self._manager: SyncManager | None = None
        self._shms: dict[str, shared_memory.SharedMemory] = {}
        self._pools: dict[str, Any] = {}
        self._metadata_pool: ListProxy | None = None
        self._work_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None
        self._available_slots_queue: mp.Queue | None = None
        self._workers: list[mp.Process] = []
        self._num_workers = 0

    # --- ADDED: Destructor for robust cleanup ---
    def __del__(self) -> None:
        """Safety net to ensure resources are cleaned up when the object is garbage collected."""
        if self._is_initialized:
            warnings.warn(
                "FlashANSRDataset was not explicitly shut down. "
                "Call `dataset.shutdown()` for cleaner resource management."
            )
            self.shutdown()

    @property
    def simplipy_engine(self) -> SimpliPyEngine:
        return self.skeleton_pool.simplipy_engine

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "FlashANSRDataset":
        '''
        Initialize a dataset from a configuration file.

        Parameters
        ----------
        config : dict or str
            The configuration file or dictionary.

        Returns
        -------
        FlashANSRDataset
            The dataset.
        '''
        config_ = load_config(config)

        if "dataset" in config_.keys():
            config_ = config_["dataset"]

        # If the config is a string, convert relative paths within the config to absolute paths
        if isinstance(config, str) and isinstance(config_["skeleton_pool"], str):
            if config_["skeleton_pool"].startswith('.'):
                config_["skeleton_pool"] = os.path.join(os.path.dirname(config), config_["skeleton_pool"])

        if os.path.isfile(config_["skeleton_pool"]) or isinstance(config_["skeleton_pool"], dict):
            skeleton_pool = SkeletonPool.from_config(config_["skeleton_pool"])
        elif os.path.isdir(config_["skeleton_pool"]):
            skeleton_pool = SkeletonPool.load(config_["skeleton_pool"])[1]
        else:
            raise ValueError(f"Invalid skeleton pool configuration: {config_['skeleton_pool']}")

        tokenizer = Tokenizer.from_config(config_["tokenizer"])

        if 'preprocessor' in config_.keys() and config_['preprocessor'] is not None:
            preprocessor = FlashASNRPreprocessor.from_config(config_['preprocessor'])
        else:
            preprocessor = None

        return cls(
            skeleton_pool=skeleton_pool,
            tokenizer=tokenizer,
            padding=config_["padding"],
            preprocessor=preprocessor
        )

    def save(self, directory: str, *args: Any, config: dict[str, Any] | str | None = None, reference: str = 'relative', recursive: bool = True, **kwargs: Any) -> None:
        '''
        Save the dataset to disk.

        Parameters
        ----------
        directory : str
            The directory to save the dataset to.
        config : dict or str, optional
            The configuration file or dictionary, by default None.
        reference : str, optional
            Determines the reference base path. One of
            - 'relative': relative to the specified directory
            - 'project': relative to the project root
            - 'absolute': absolute paths
        recursive : bool, optional
            Save any referenced configs too
        **kwargs
            Additional arguments to pass to the dataset's save_to_disk method.
        '''
        if self.data is None:
            raise ValueError("No dataset to save. Please generate or load a dataset first.")

        directory = substitute_root_path(directory)

        os.makedirs(directory, exist_ok=True)

        self.data.save_to_disk(dataset_path=os.path.join(directory, 'dataset'), *args, **kwargs)

        # Copy the config to the directory for best portability
        if config is None:
            warnings.warn("No config specified, saving the model without a config file. Loading the model will require manual configuration.")
        else:
            save_config(load_config(config, resolve_paths=True), directory=directory, filename='dataset.yaml', reference=reference, recursive=recursive, resolve_paths=True)

    @classmethod
    def load(cls, directory: str) -> tuple[dict[str, Any], "FlashANSRDataset"]:
        '''
        Load a dataset from disk.

        Parameters
        ----------
        directory : str
            The directory to load the dataset from.

        Returns
        -------
        dict
            The configuration dictionary.
        FlashANSRDataset
            The FlashANSRDataset object.
        '''
        config_path = os.path.join(directory, 'dataset.yaml')
        resolved_directory = substitute_root_path(directory)

        dataset = cls.from_config(config_path)
        dataset.data = load_from_disk(os.path.join(resolved_directory, 'dataset'))

        return load_config(config_path), dataset

    def _pad_sequence(self, sequence: list[int], max_length: int, pad_value: Any, device: str | torch.device | int = 'cpu', dtype: torch.dtype = torch.long) -> torch.Tensor:
        if not isinstance(sequence, torch.Tensor):
            sequence_tensor = torch.tensor(sequence, device=device, dtype=dtype)
        else:
            # If it's already a tensor, just ensure it's on the correct device.
            sequence_tensor = sequence.to(device=device, dtype=dtype)

        return torch.nn.functional.pad(
            sequence_tensor,
            (0, max_length - len(sequence)),
            value=pad_value
        )

    def collate(self, batch: dict[str, Any], device: str | torch.device | int = 'cpu') -> dict[str, Any]:
        '''
        Collate a batch of data inplace.
        '''
        # --- Streamlined and Corrected Batch Processing Logic ---

        # 1. Pad and stack 'input_ids' (variable-length sequences)
        # We handle this first because it's a list of lists.
        if isinstance(batch['input_ids'][0], list):
            max_length_input_ids = max(len(seq) for seq in batch['input_ids'])
            padded_input_ids = [
                self._pad_sequence(seq, max_length_input_ids, self.tokenizer['<pad>'], device=device, dtype=torch.long)
                for seq in batch['input_ids']
            ]
            batch['input_ids'] = torch.stack(padded_input_ids)
        else:  # It's likely already a padded tensor
            batch['input_ids'] = batch['input_ids'].to(device=device, dtype=torch.long)

        # 2. Stack dense tensors ('x_tensors', 'y_tensors')
        for k, dtype in [
            ('x_tensors', torch.float32),
            ('y_tensors', torch.float32),
        ]:
            # The input from the dataloader might be a list of tensors; stack them.
            if isinstance(batch[k], list):
                batch[k] = torch.stack(batch[k])
            batch[k] = batch[k].to(device=device, dtype=dtype)

        # 3. Handle 'constants' (ragged data) -> THE CRUCIAL FIX
        # Since constants have variable lengths, they CANNOT be stacked.
        # We process them into a LIST of tensors on the correct device.
        constants_list = []
        for const_item in batch['constants']:
            # Ensure each item is a tensor before moving to the device
            if not isinstance(const_item, torch.Tensor):
                const_item = torch.tensor(const_item, dtype=torch.float32)
            constants_list.append(const_item.to(device))
        batch['constants'] = constants_list  # The result is a list[torch.Tensor]

        # 4. Handle other optional fields
        if 'input_num' in batch:
            if isinstance(batch['input_num'][0], list):
                max_length_input_num = max(len(seq) for seq in batch['input_num'])
                padded_input_num = [
                    self._pad_sequence(seq, max_length_input_num, torch.nan, device=device, dtype=torch.float32)
                    for seq in batch['input_num']
                ]
                batch['input_num'] = torch.stack(padded_input_num).unsqueeze(-1)
            else:  # Already a tensor
                batch['input_num'] = batch['input_num'].to(device=device, dtype=torch.float32)

        if 'complexities' in batch:
            batch['complexities'] = [
                torch.tensor(c, device=device, dtype=torch.float32) if c is not None else None
                for c in batch['complexities']
            ]

        # 5. Create labels
        batch['labels'] = batch['input_ids'].clone()[..., 1:]

        # 6. Create ids for equal expressions
        batch['expression_ids'] = []
        expression_to_id: dict[tuple, int] = {}

        for i, expr in enumerate(batch['input_ids']):
            expr_key = tuple(expr.flatten().tolist())
            if expr_key not in expression_to_id:
                expression_to_id[expr_key] = len(expression_to_id)
            batch['expression_ids'].append(expression_to_id[expr_key])
        batch['expression_ids'] = torch.tensor(batch['expression_ids'], device=device, dtype=torch.long)

        return batch

    def compile(self, size: int | None = None, steps: int | None = None, batch_size: int | None = None, n_support: int | None = None, verbose: bool = False) -> None:
        disable_progress_bars()
        if size is None and steps is None:
            size = len(self.skeleton_pool)

        self.data = Dataset.from_list(
            list(self.iterate(size=size, steps=steps, batch_size=batch_size, n_support=n_support, verbose=verbose))
        )

    @staticmethod
    def _producer_worker(
        work_queue: mp.Queue,
        result_queue: mp.Queue,
        shm_configs: dict,
        metadata_list: list,
        worker_init_args: dict
    ) -> None:
        """
        Worker that generates data and writes it directly into shared memory slots.
        If generating samples for a specific equation fails (hits max attempts),
        it discards that equation and tries a new one, without aborting the whole batch.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        np.random.seed(os.getpid())

        # Unpack initialization arguments
        skeleton_pool: SkeletonPool = worker_init_args['skeleton_pool']
        tokenizer: Tokenizer = worker_init_args['tokenizer']
        padding: str = worker_init_args['padding']
        n_per_equation: int = worker_init_args['n_per_equation']
        batch_size: int = worker_init_args['batch_size']

        # --- Connect to shared memory and create numpy views ---
        shms = {name: shared_memory.SharedMemory(name=cfg['name']) for name, cfg in shm_configs.items()}
        pools = {name: np.ndarray(cfg['shape'], dtype=cfg['dtype'], buffer=shms[name].buf) for name, cfg in shm_configs.items()}

        try:
            while True:
                job = work_queue.get()
                if job is None:  # Sentinel
                    break

                slot_idx, n_support_frag = job

                # --- Buffers for a single batch ---
                # These are views into the shared memory for the assigned slot
                x_tensors_batch = pools['x_tensors'][slot_idx]
                y_tensors_batch = pools['y_tensors'][slot_idx]
                input_ids_batch = pools['input_ids'][slot_idx]

                # Local buffers for data that can't be pre-allocated
                constants_batch = []
                metadata_batch = []

                # --- Generate one full batch ---
                i = 0
                while i < batch_size:
                    # Step 1: Find a usable skeleton
                    try:
                        skeleton_hash, skeleton_code, skeleton_constants = skeleton_pool.sample_skeleton()
                        skeleton = list(skeleton_hash)
                    except NoValidSampleFoundError:
                        continue  # Try finding another skeleton, batch progress `i` is unaffected

                    # Step 2: Attempt to generate n_per_equation samples for this skeleton
                    # Use temporary lists to hold samples. Only commit them if all are successful.
                    temp_samples = []
                    attempts = 0
                    max_total_attempts = n_per_equation * 20  # A reasonable limit

                    succeeded = True
                    # The number of samples to generate for this equation, ensuring we don't exceed batch_size
                    n_to_generate = min(n_per_equation, batch_size - i)

                    for _ in range(n_to_generate):
                        sample_found = False
                        while not sample_found:
                            if attempts >= max_total_attempts:
                                succeeded = False
                                break  # Abort this skeleton

                            attempts += 1
                            try:
                                # Generate one sample
                                x_support, y_support, literals = skeleton_pool.sample_data(
                                    skeleton_code, len(skeleton_constants), n_support_frag
                                )

                                if padding == 'zero':
                                    for var_idx, variable in enumerate(skeleton_pool.variables):
                                        if variable not in skeleton:
                                            x_support[:, var_idx] = 0

                                input_ids = tokenizer.encode(skeleton, add_bos=True, add_eos=True)

                                # Store the successful sample
                                temp_samples.append({
                                    'x': x_support, 'y': y_support, 'input_ids': input_ids,
                                    'constants': literals,
                                    'metadata': {
                                        'skeletons': skeleton,
                                        'skeleton_hashes': skeleton_hash,
                                        'expressions': substitude_constants(skeleton, values=literals, inplace=False),
                                    }
                                })
                                sample_found = True  # Success, move to next sample for this skeleton

                            except NoValidSampleFoundError:
                                continue  # Retry generating this specific sample

                        if not succeeded:
                            break  # Exit the for loop if max attempts were reached

                    # Step 3: If generation for this skeleton was successful, commit samples to the batch
                    if succeeded:
                        for sample in temp_samples:
                            # Populate batch arrays
                            x_tensors_batch[i] = sample['x']
                            y_tensors_batch[i] = sample['y']

                            # Pad and insert input_ids
                            input_ids_batch[i, :] = tokenizer['<pad>']  # Reset padding
                            input_ids_batch[i, :len(sample['input_ids'])] = sample['input_ids']

                            constants_batch.append(sample['constants'])
                            metadata_batch.append(sample['metadata'])

                            i += 1  # Increment main batch counter
                    # If not succeeded, we do nothing. The main `while i < batch_size` loop continues,
                    # effectively discarding the failed skeleton and its partial samples.

                # --- Batch is complete, finalize and notify main process ---
                metadata_list[slot_idx] = {'metadata': metadata_batch, 'constants': constants_batch}
                result_queue.put(slot_idx)

        finally:
            # Clean up worker's connection to shared memory
            for shm in shms.values():
                shm.close()

    def _initialize_workers(self, prefetch_factor: int, batch_size: int, n_per_equation: int, max_seq_len: int, num_workers: int | None = None) -> None:
        """Initializes all multiprocessing resources. Idempotent."""
        if self._is_initialized:
            return

        self._num_workers = os.cpu_count() or 1 if num_workers is None else num_workers
        pool_size = self._num_workers * prefetch_factor

        # 1. Define Shapes and Create Shared Memory Pools
        shm_configs: dict[str, dict[str, Any]] = {
            'x_tensors': {'shape': (pool_size, batch_size, self.skeleton_pool.n_support_prior_config['kwargs']['max_value'], len(self.skeleton_pool.variables)), 'dtype': np.float32},
            'y_tensors': {'shape': (pool_size, batch_size, self.skeleton_pool.n_support_prior_config['kwargs']['max_value'], 1), 'dtype': np.float32},
            'input_ids': {'shape': (pool_size, batch_size, max_seq_len), 'dtype': np.int64},
        }

        # Create SHM segments and store them on the instance
        self._shms = {
            name: shared_memory.SharedMemory(create=True, size=int(np.prod(cfg['shape']) * np.dtype(cfg['dtype']).itemsize))
            for name, cfg in shm_configs.items()
        }
        for name, shm in self._shms.items():
            shm_configs[name]['name'] = shm.name

        self._pools = {
            name: np.ndarray(cfg['shape'], dtype=cfg['dtype'], buffer=self._shms[name].buf)
            for name, cfg in shm_configs.items()
        }

        # Create Manager and Queues
        self._manager = mp.Manager()
        self._metadata_pool = self._manager.list([None] * pool_size)
        self._work_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._available_slots_queue = mp.Queue()
        for i in range(pool_size):
            self._available_slots_queue.put(i)

        # 3. Start Worker Processes
        worker_init_args = {
            'skeleton_pool': self.skeleton_pool, 'tokenizer': self.tokenizer,
            'padding': self.padding, 'n_per_equation': n_per_equation, 'batch_size': batch_size
        }
        self._workers = []
        for _ in range(self._num_workers):
            p = mp.Process(target=self._producer_worker, args=(self._work_queue, self._result_queue, shm_configs, self._metadata_pool, worker_init_args), daemon=True)
            p.start()
            self._workers.append(p)

        self._is_initialized = True

    def shutdown(self) -> None:
        """Gracefully shuts down all multiprocessing resources."""
        if not self._is_initialized:
            return

        if self._work_queue is None or self._result_queue is None or self._available_slots_queue is None:
            raise RuntimeError("Multiprocessing resources are not properly initialized.")

        try:
            # Tell workers to stop
            for _ in range(self._num_workers):
                self._work_queue.put(None)

            # Wait for them to finish
            for p in self._workers:
                p.join(timeout=5)  # Add a timeout
                if p.is_alive():
                    p.terminate()  # Force terminate if stuck

            # Shutdown manager
            if self._manager:
                self._manager.shutdown()

            # Close and unlink shared memory
            for shm in self._shms.values():
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass  # Already unlinked

        finally:
            # Reset state regardless of success
            self._is_initialized = False
            self._manager = None
            self._shms.clear()
            self._pools.clear()
            self._metadata_pool = None
            self._work_queue = None
            self._result_queue = None
            self._available_slots_queue = None
            self._workers.clear()
            self._num_workers = 0

    def iterate(
        self,
        size: int | None = None,
        steps: int | None = None,
        batch_size: int | None = None,
        n_support: int | None = None,
        max_seq_len: int = 128,
        n_per_equation: int = 1,
        preprocess: bool = False,
        verbose: bool = False,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        persistent: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        if batch_size is None:
            batch_size = 1

        if self.data is not None:
            yield from tqdm(self.data, desc="Iterating over pre-compiled dataset", disable=not verbose)
            return

        if size is None and steps is None:
            raise ValueError("Must specify 'size' or 'steps'.")
        if steps is None:
            steps = (size + batch_size - 1) // batch_size  # type: ignore

        self._initialize_workers(
            prefetch_factor=prefetch_factor,
            batch_size=batch_size,
            n_per_equation=n_per_equation,
            max_seq_len=max_seq_len,
            num_workers=num_workers,
        )
        pool_size = self._num_workers * prefetch_factor

        if self._work_queue is None or self._result_queue is None or self._available_slots_queue is None or self._metadata_pool is None or self._pools is None:
            raise RuntimeError("Multiprocessing resources are not properly initialized.")

        pbar = tqdm(total=steps, desc="Generating Batches", disable=not verbose)
        try:
            # Prefill the work queue
            for _ in range(min(steps, pool_size)):
                slot_idx = self._available_slots_queue.get()
                n_support_frag = self.skeleton_pool.n_support_prior_config["kwargs"]["max_value"] if n_support is None else n_support
                self._work_queue.put((slot_idx, n_support_frag))

            # Main producer-consumer loop
            for _ in range(steps):
                completed_slot_idx = self._result_queue.get()

                # Construct batch
                metadata_and_constants = self._metadata_pool[completed_slot_idx]
                batch_dict = {
                    'x_tensors': torch.from_numpy(self._pools['x_tensors'][completed_slot_idx]),
                    'y_tensors': torch.from_numpy(self._pools['y_tensors'][completed_slot_idx]),
                    'input_ids': torch.from_numpy(self._pools['input_ids'][completed_slot_idx]),
                    'constants': [torch.from_numpy(c) for c in metadata_and_constants['constants']],
                    **{k: [d[k] for d in metadata_and_constants['metadata']] for k in metadata_and_constants['metadata'][0]}
                }

                if persistent:
                    batch_dict = {k: v.clone() if isinstance(v, torch.Tensor) else [t.clone() for t in v] if k == 'constants' else v for k, v in batch_dict.items()}  # type: ignore

                if preprocess and self.preprocessor:
                    yield self.preprocessor.format(batch_dict)
                else:
                    yield batch_dict

                pbar.update(1)

                # Recycle the slot and request new work if there are more steps to go
                self._available_slots_queue.put(completed_slot_idx)
                if _ + pool_size < steps:
                    slot_to_refill = self._available_slots_queue.get()
                    n_support_frag = self.skeleton_pool.n_support_prior_config["kwargs"]["max_value"] if n_support is None else n_support
                    self._work_queue.put((slot_to_refill, n_support_frag))

        finally:
            pbar.close()

    def _benchmark(self, n_samples: int, batch_size: int, verbose: bool = False) -> dict[str, Any]:
        '''
        Benchmark the speed of the dataset generation.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        batch_size : int
            The batch size.
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Returns
        -------
        dict
            The benchmark results.
        '''
        iteration_times = []
        time_1 = time.time()
        for _ in self.iterate(size=n_samples, steps=None, batch_size=batch_size, n_support=None, verbose=verbose):
            iteration_times.append(time.time() - time_1)
            time_1 = time.time()

        iteration_times_array = np.array(iteration_times)

        return {
            'mean_iteration_time': iteration_times_array.mean(),
            'std_iteration_time': iteration_times_array.std(),
            'min_iteration_time': iteration_times_array.min(),
            'max_iteration_time': iteration_times_array.max(),
        }

    def __len__(self) -> int:
        '''

        '''
        if self.data is None:
            raise ValueError("No dataset to get the length of. Please generate or load a dataset first.")

        return len(self.data)

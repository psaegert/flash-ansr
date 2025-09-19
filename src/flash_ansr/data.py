import os
import warnings
import time
from typing import Any, Generator, Literal
import signal
import atexit
import multiprocessing as mp
from multiprocessing import shared_memory

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


# --- Globals for robust cleanup ---
_managed_resources: list = []


def _cleanup_resources() -> None:
    """Ensure shared memory is unlinked and managers are shut down on exit."""
    for resource in _managed_resources:
        if isinstance(resource, shared_memory.SharedMemory):
            try:
                resource.close()
                resource.unlink()  # This is the crucial step
            except FileNotFoundError:
                pass  # Already cleaned up
        elif hasattr(resource, 'shutdown'):
            resource.shutdown()
        elif isinstance(resource, mp.Process):
            if resource.is_alive():
                resource.terminate()
            resource.join()


atexit.register(_cleanup_resources)


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
        For each skeleton, it generates n_per_equation positive samples.
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
                constants_batch = []
                x_tensors_batch = np.zeros(pools['x_tensors'].shape[1:], dtype=pools['x_tensors'].dtype)
                y_tensors_batch = np.zeros(pools['y_tensors'].shape[1:], dtype=pools['y_tensors'].dtype)
                input_ids_batch = np.full(pools['input_ids'].shape[1:], tokenizer['<pad>'], dtype=pools['input_ids'].dtype)
                metadata_batch = []

                # --- Generate one full batch ---
                i = 0
                while i < batch_size:
                    # Step 1: Find a usable skeleton by sampling until success
                    skeleton_hash, skeleton_code, skeleton_constants, skeleton = None, None, None, None
                    while True:
                        try:
                            skeleton_hash, skeleton_code, skeleton_constants = skeleton_pool.sample_skeleton()
                            skeleton = list(skeleton_hash)
                            break  # Found a valid skeleton
                        except NoValidSampleFoundError:
                            # This is rare but possible if the pool is heavily filtered. Keep trying.
                            continue

                    # Step 2: Generate n_per_equation samples (positive pairs) for this skeleton
                    generated_count = 0
                    # A safety break to prevent infinite loops on pathologically difficult skeletons
                    max_attempts_per_skeleton = 10

                    for _ in range(max_attempts_per_skeleton):
                        # Stop if we have enough samples for this skeleton OR the entire batch is full
                        if generated_count >= n_per_equation or i >= batch_size:
                            break

                        try:
                            # Resample data (constants and points) for the SAME skeleton
                            x_support, y_support, literals = skeleton_pool.sample_data(
                                skeleton_code, len(skeleton_constants), n_support_frag
                            )

                            if padding == 'zero':
                                for var_idx, variable in enumerate(skeleton_pool.variables):
                                    if variable not in skeleton:
                                        x_support[:, var_idx] = 0

                            input_ids = tokenizer.encode(skeleton, add_bos=True, add_eos=True)

                            # Populate the batch arrays at the current index 'i'
                            x_tensors_batch[i] = x_support
                            y_tensors_batch[i] = y_support
                            input_ids_batch[i, :len(input_ids)] = input_ids
                            constants_batch.append(literals)
                            metadata_batch.append({
                                'skeletons': skeleton,
                                'skeleton_hashes': skeleton_hash,
                                'expressions': substitude_constants(skeleton, values=literals, inplace=False),
                            })

                            # Increment counters for the next sample
                            generated_count += 1
                            i += 1

                        except NoValidSampleFoundError:
                            # If data sampling for this skeleton fails, just try again
                            continue

                # --- Write the completed batch directly into the shared memory slot ---
                pools['x_tensors'][slot_idx] = x_tensors_batch
                pools['y_tensors'][slot_idx] = y_tensors_batch
                pools['input_ids'][slot_idx] = input_ids_batch

                # Metadata and constants are handled by the manager/queue
                metadata_list[slot_idx] = {'metadata': metadata_batch, 'constants': constants_batch}

                # --- Signal completion ---
                result_queue.put(slot_idx)

        finally:
            # Clean up worker's connection to shared memory
            for shm in shms.values():
                shm.close()

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

        if num_workers is None:
            num_workers = os.cpu_count() or 1

        if steps is None:
            steps = (size + batch_size - 1) // batch_size  # type: ignore

        pool_size = num_workers * prefetch_factor

        # --- 1. Define Shapes and Create Shared Memory Pools ---
        shm_configs: dict[str, dict[str, Any]] = {
            'x_tensors': {'shape': (pool_size, batch_size, self.skeleton_pool.n_support_prior_config['kwargs']['max_value'], len(self.skeleton_pool.variables)), 'dtype': np.float32},
            'y_tensors': {'shape': (pool_size, batch_size, self.skeleton_pool.n_support_prior_config['kwargs']['max_value'], 1), 'dtype': np.float32},
            'input_ids': {'shape': (pool_size, batch_size, max_seq_len), 'dtype': np.int64},
        }

        shms = {name: shared_memory.SharedMemory(create=True, size=int(np.prod(cfg['shape']) * np.dtype(cfg['dtype']).itemsize)) for name, cfg in shm_configs.items()}
        for shm in shms.values():
            _managed_resources.append(shm)

        # Add shm names to configs to pass to workers
        for name, shm in shms.items():
            shm_configs[name]['name'] = shm.name

        pools = {name: np.ndarray(cfg['shape'], dtype=cfg['dtype'], buffer=shms[name].buf) for name, cfg in shm_configs.items()}

        manager = mp.Manager()
        _managed_resources.append(manager)
        metadata_pool = manager.list([None] * pool_size)

        # --- 2. Setup Coordination Queues ---
        work_queue: mp.Queue = mp.Queue()
        result_queue: mp.Queue = mp.Queue()
        available_slots_queue: mp.Queue = mp.Queue()
        for i in range(pool_size):
            available_slots_queue.put(i)  # Prime with all available slots

        # --- 3. Start Worker Processes ---
        worker_init_args = {
            'skeleton_pool': self.skeleton_pool, 'tokenizer': self.tokenizer,
            'padding': self.padding, 'n_per_equation': n_per_equation, 'batch_size': batch_size
        }

        workers = []
        for _ in range(num_workers):
            p = mp.Process(target=self._producer_worker, args=(work_queue, result_queue, shm_configs, metadata_pool, worker_init_args), daemon=True)
            p.start()
            workers.append(p)
            _managed_resources.append(p)

        pbar = tqdm(total=steps, desc="Generating Batches", disable=not verbose)

        try:
            # --- 4. Main Producer-Consumer Loop ---
            # Prefill the work queue to get workers started
            for _ in range(pool_size):
                slot_idx = available_slots_queue.get()
                n_support_frag = self.skeleton_pool.n_support_prior_config["kwargs"]["max_value"] if n_support is None else n_support
                work_queue.put((slot_idx, n_support_frag))

            for i in range(steps):
                # Wait for a worker to finish a slot
                completed_slot_idx = result_queue.get()

                # --- Construct the batch by reading from the shared slot (Only copy if "persistent") ---
                metadata_and_constants = metadata_pool[completed_slot_idx]
                batch_dict = {
                    'x_tensors': torch.from_numpy(pools['x_tensors'][completed_slot_idx]).clone() if persistent else torch.from_numpy(pools['x_tensors'][completed_slot_idx]),
                    'y_tensors': torch.from_numpy(pools['y_tensors'][completed_slot_idx]).clone() if persistent else torch.from_numpy(pools['y_tensors'][completed_slot_idx]),
                    'input_ids': torch.from_numpy(pools['input_ids'][completed_slot_idx]).clone() if persistent else torch.from_numpy(pools['input_ids'][completed_slot_idx]),
                    'constants': [torch.from_numpy(c).clone() if persistent else torch.from_numpy(c) for c in metadata_and_constants['constants']],
                    **{k: [d[k] for d in metadata_and_constants['metadata']] for k in metadata_and_constants['metadata'][0]}
                }

                if preprocess and self.preprocessor:
                    yield self.preprocessor.format(batch_dict)
                else:
                    yield batch_dict

                pbar.update(1)

                # Recycle the slot by putting it back in the available queue and request new work
                available_slots_queue.put(completed_slot_idx)
                slot_to_refill = available_slots_queue.get()
                n_support_frag = self.skeleton_pool.n_support_prior_config["kwargs"]["max_value"] if n_support is None else n_support
                work_queue.put((slot_to_refill, n_support_frag))

        finally:
            pbar.close()

            # --- 5. Graceful Shutdown (Corrected Order) ---
            if verbose:
                print("Shutting down workers and cleaning up resources...")

            # Step 1: Tell all workers to stop accepting new jobs
            for _ in range(num_workers):
                work_queue.put(None)

            # Step 2: Wait for all workers to finish their current job and exit
            for p in workers:
                p.join()

            # Step 3: Now that workers are stopped, it's safe to clean up shared resources
            atexit.unregister(_cleanup_resources)
            _cleanup_resources()   # This will shut down the manager and unlink memory

            if verbose:
                print("All resources cleaned up.")

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

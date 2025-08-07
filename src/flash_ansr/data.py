import os
import warnings
import time
from typing import Any, Generator, Literal

import torch
import numpy as np

from tqdm import tqdm
from datasets import Dataset, load_from_disk, disable_progress_bars

from simplipy import SimpliPyEngine

from flash_ansr.models.transformer_utils import Tokenizer
from flash_ansr.utils import load_config, save_config, substitute_root_path
from flash_ansr.expressions import SkeletonPool, NoValidSampleFoundError
from flash_ansr.expressions.utils import substitude_constants
from flash_ansr.preprocess import FlashASNRPreprocessor


class FlashANSRDataset:
    '''
    Dataset for amortized neural symbolic regression training.

    Parameters
    ----------
    skeleton_pool : SkeletonPool
        The skeleton pool to sample from.
    padding : {'random', 'zero'}
        The padding strategy for the input_ids, by default 'random'.
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
        return torch.nn.functional.pad(
            torch.tensor(sequence, device=device, dtype=dtype),
            (0, max_length - len(sequence)),
            value=pad_value
        )

    def collate(self, batch: dict[str, Any], device: str | torch.device | int = 'cpu') -> dict[str, Any]:
        '''
        Collate a batch of data inplace.

        Parameters
        ----------
        batch : dict
            The batch of data.
        device : str or torch.device or int, optional
            The device to move the data to. If a string, it should be one of 'cpu', 'cuda', 'cuda:0', etc., by default 'cpu'.

        Returns
        -------
        tuple
            The collated batch.
        '''
        # Determine the maximum length of the input_ids
        if isinstance(batch['input_ids'][0], list):
            # Batch of instances
            max_length_input_ids = max(len(input_id) for input_id in batch['input_ids'])

            for i in range(len(batch['input_ids'])):
                batch['input_ids'][i] = self._pad_sequence(batch['input_ids'][i], max_length_input_ids, self.tokenizer['<pad>'], device=device, dtype=torch.long)

            for k, dtype in [
                ('input_ids', torch.long),
                ('x_tensors', torch.float32),
                ('y_tensors', torch.float32),
            ]:
                batch[k] = torch.stack(batch[k]).to(device=device, dtype=dtype)  # type: ignore

            for i, constant in enumerate(batch['constants']):
                if isinstance(constant, torch.Tensor):
                    batch['constants'][i] = constant.to(device)
                else:
                    batch['constants'][i] = torch.tensor(constant, device=device, dtype=torch.float32)

            if 'input_num' in batch:
                max_length_input_num = max(len(input_id) for input_id in batch['input_num'])
                for i in range(len(batch['input_num'])):
                    batch['input_num'][i] = self._pad_sequence(batch['input_num'][i], max_length_input_num, torch.nan, device=device, dtype=torch.float32)

                batch['input_num'] = torch.stack(batch['input_num']).to(device=device, dtype=torch.float32).unsqueeze(-1)

            if 'complexities' in batch:
                batch['complexities'] = [torch.tensor(c, device=device, dtype=torch.float32) if c is not None else None for c in batch['complexities']]

        else:
            # Single instance
            max_length_input_ids = len(batch['input_ids'])

            batch['input_ids'] = self._pad_sequence(batch['input_ids'], max_length_input_ids, self.tokenizer['<pad>'], device=device, dtype=torch.long)

            for k, dtype in [
                ('input_ids', torch.long),
                ('x_tensors', torch.float32),
                ('y_tensors', torch.float32),
            ]:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device=device, dtype=dtype)
                else:
                    batch[k] = torch.tensor(batch[k], device=device, dtype=dtype)

            if isinstance(batch['constants'], torch.Tensor):
                batch['constants'] = batch['constants'].to(device)
            else:
                batch['constants'] = torch.tensor(batch['constants'], device=device, dtype=torch.float32)

            if 'input_num' in batch:
                max_length_input_num = len(batch['input_num'])
                batch['input_num'] = self._pad_sequence(batch['input_num'], max_length_input_num, torch.nan, device=device, dtype=torch.float32).unsqueeze(-1)

            if 'complexities' in batch:
                if batch['complexities'] is not None:
                    batch['complexities'] = torch.tensor(batch['complexities'], device=device, dtype=torch.float32)

        # Create the labels for the next token prediction task (i.e. shift the input_ids by one position to the right)
        batch['labels'] = batch['input_ids'].clone()[..., 1:]

        return batch

    def compile(self, size: int | None = None, steps: int | None = None, batch_size: int | None = None, n_support: int | None = None, verbose: bool = False) -> None:
        disable_progress_bars()
        if size is None and steps is None:
            size = len(self.skeleton_pool)

        self.data = Dataset.from_list(
            list(self.iterate(size=size, steps=steps, batch_size=batch_size, n_support=n_support, verbose=verbose))
        )

    def iterate(
            self,
            size: int | None = None,
            steps: int | None = None,
            batch_size: int | None = None,
            n_support: int | None = None,
            n_per_equation: int = 1,
            tqdm_total: int | None = None,
            avoid_fragmentation: bool = True,
            preprocess: bool = False,
            verbose: bool = False) -> Generator[dict[str, list | torch.Tensor], None, None]:
        '''
        Iterate over the dataset.

        Parameters
        ----------
        size : int, optional
            The total number of data to generate, by default None.
        steps : int, optional
            The number of batches to generate, by default None.
        batch_size : int or BatchSizeScheduler, optional
            The batch size or scheduler, by default None.
        n_support : int, optional
            The number of support points to sample, by default None.
        n_per_equation : int, optional
            The number of instances with distinct constants and support points to generate per equation, by default 1.
        tqdm_total : int, optional
            The total number of iterations for the tqdm progress bar, by default None.
        avoid_fragmentation : bool, optional
            Whether to avoid memory fragmentation by allocating the maximum size tensor in the first batch, by default True.
        preprocess : bool, optional
            Whether to preprocess the data, by default False.
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Yields
        -------
        dict
            The next batch of data.
        '''
        if self.data is None:
            # Procedurally generate the dataset

            if batch_size is None:
                # Generate single instances

                if steps is not None:
                    # Trying to parameterize the number of batches for non-batched data generation
                    raise ValueError(f'Speficfied {steps=} which is not used for non-batched data generation')

                if preprocess and self.preprocessor is not None:
                    # Preprocess with available preprocessor
                    for instance in self.generate(size=size, n_support=n_support, n_per_equation=n_per_equation, tqdm_total=tqdm_total, verbose=verbose, avoid_fragmentation=avoid_fragmentation):
                        yield self.preprocessor.format(instance)
                else:
                    if preprocess:
                        # Trying to preprocess without a preprocessor
                        warnings.warn("No preprocessor specified, skipping preprocessing.")

                    # Generate the data without preprocessing
                    yield from self.generate(size=size, n_support=n_support, n_per_equation=n_per_equation, tqdm_total=tqdm_total, verbose=verbose, avoid_fragmentation=avoid_fragmentation)

            else:
                # Generate data in batches

                if preprocess and self.preprocessor is not None:
                    # Preprocess with available preprocessor
                    for batch in self.generate_batch(batch_size=batch_size, size=size, steps=steps, n_support=n_support, n_per_equation=n_per_equation, tqdm_total=tqdm_total, verbose=verbose, avoid_fragmentation=avoid_fragmentation):
                        yield self.preprocessor.format(batch)
                else:
                    if preprocess:
                        # Trying to preprocess without a preprocessor
                        warnings.warn("No preprocessor specified, skipping preprocessing.")

                    # Generate the data without preprocessing
                    yield from self.generate_batch(batch_size=batch_size, size=size, steps=steps, n_support=n_support, n_per_equation=n_per_equation, tqdm_total=tqdm_total, verbose=verbose, avoid_fragmentation=avoid_fragmentation)

        else:
            yield from tqdm(self.data, desc="Iterating over dataset", disable=not verbose, smoothing=0.01)

    def generate_batch(
            self,
            batch_size: int,
            size: int | None = None,
            steps: int | None = None,
            n_support: int | None = None,
            n_per_equation: int = 1,
            tqdm_total: int | None = None,
            avoid_fragmentation: bool = True,
            verbose: bool = False) -> Generator[dict[str, list | torch.Tensor], None, None]:
        '''
        Generate a batch of data.

        Parameters
        ----------
        batch_size : int
            The batch size.
        size : int, optional
            The total number of data to generate, by default None.
        steps : int, optional
            The number of batches to generate, by default None.
        n_support : int, optional
            The number of support points to sample, by default None.
        n_per_equation : int, optional
            The number of instances with distinct constants and support points to generate per equation, by default 1.
        tqdm_total : int, optional
            The total number of iterations for the tqdm progress bar, by default None
        avoid_fragmentation : bool, optional
            Whether to avoid memory fragmentation by allocating the maximum size tensor in the first batch, by default True.
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Yields
        -------
        dict
            The next batch of data.
        '''
        if size is not None and steps is not None:
            raise ValueError(f'Must either specify the total number of data (size) or batches (steps) to generate, got {size=}, {steps=}')

        if size is not None:
            if isinstance(batch_size, int):
                steps = int(np.ceil(size / batch_size))

        pbar = tqdm(desc="Batch generating data", unit="b", total=steps or tqdm_total, disable=not verbose, smoothing=0.01)

        batch_id = 0
        n_rejected = 0
        n_generated = 0

        while (size is None and steps is None) or (steps is not None and batch_id < steps) or (size is not None and n_generated < size):
            batch: dict[str, list | torch.Tensor] = {
                'n_rejected': [],
                'skeletons': [],
                'skeleton_hashes': [],
                'expressions': [],
                'constants': [],
                'input_ids': [],
                'x_tensors': [],
                'y_tensors': [],
            }

            if n_support is None:
                if batch_id == 0 and avoid_fragmentation:
                    # Allocate the maximum size tensor to avoid memory fragmentation
                    n_support_frag = int(self.skeleton_pool.n_support_prior_kwargs['max_value'])
                elif n_support is None:
                    n_support_frag = int(np.round(self.skeleton_pool.n_support_prior(size=1))[0])
                else:
                    n_support_frag = n_support
            else:
                n_support_frag = n_support

            for instance in self.generate(
                    size=min(batch_size, size - batch_id * batch_size) if size is not None else batch_size,
                    n_support=n_support_frag,
                    n_per_equation=n_per_equation,
                    avoid_fragmentation=False,
                    verbose=False):

                for key in instance.keys():
                    batch[key].append(instance[key])  # type: ignore

            n_rejected += batch['n_rejected'][-1][0]  # type: ignore
            n_generated += len(batch['input_ids'])

            yield batch

            batch = {k: [] for k in batch}
            batch_id += 1

            pbar.update(1)
            pbar.set_postfix(reject_rate=f'{n_rejected / (n_rejected + n_generated):.2%}')

        pbar.close()

    def generate(
            self,
            size: int | None = None,
            n_support: int | None = None,
            n_per_equation: int = 1,
            avoid_fragmentation: bool = True,
            tqdm_total: int | None = None,
            verbose: bool = False) -> Generator[dict[str, list[str | int] | torch.Tensor], None, None]:
        '''
        Generate data.

        Parameters
        ----------
        size : int, optional
            The total number of data to generate, by default None.
        n_support : int, optional
            The number of support points to sample, by default None.
        n_per_equation : int, optional
            The number of instances with distinct constants and support points to generate per equation, by default 1.
        avoid_fragmentation : bool, optional
            Whether to avoid memory fragmentation by allocating the maximum size tensor in the first batch, by default True.
        tqdm_total : int, optional
            The total number of iterations for the tqdm progress bar, by default None.
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Yields
        -------
        dict
            The next batch of data.
        '''
        if verbose:
            pbar = tqdm(desc="Generating data", total=size or tqdm_total, smoothing=0.01)

        n_generated = 0
        n_rejected = 0
        while size is None or n_generated < size:
            # Allocate the maximum size tensor to avoid memory fragmentation
            n_support_frag = int(self.skeleton_pool.n_support_prior_kwargs['max_value']) if n_generated == 0 and avoid_fragmentation else n_support

            try:
                # sample = self.skeleton_pool.sample(n_support=n_support_frag)
                skeleton_hash, skeleton_code, skeleton_constants = self.skeleton_pool.sample_skeleton()
                skeleton = list(skeleton_hash)

                buffer = []

                n_generated_per_equation = 0
                while n_generated_per_equation < n_per_equation:
                    try:
                        x_support, y_support, literals = self.skeleton_pool.sample_data(skeleton_code, len(skeleton_constants), n_support_frag)

                        if self.padding == 'zero':
                            # Set all x that do not appear in the expression to 0
                            for i, variable in enumerate(self.skeleton_pool.variables):
                                if variable not in skeleton:
                                    x_support[:, i] = 0

                        # Tokenize the expression to get the input_ids
                        input_ids = self.tokenizer.encode(skeleton, add_bos=True, add_eos=True)

                        # Yield the sample
                        buffer.append({
                            'n_rejected': [n_rejected],
                            'skeletons': skeleton,
                            'skeleton_hashes': skeleton_hash,
                            'expressions': substitude_constants(skeleton, values=literals, inplace=False),
                            'constants': torch.tensor(literals, dtype=torch.float32),
                            'input_ids': input_ids,
                            'x_tensors': torch.tensor(x_support, dtype=torch.float32),
                            'y_tensors': torch.tensor(y_support, dtype=torch.float32),
                        })

                    except NoValidSampleFoundError:
                        buffer = []
                        n_generated_per_equation = 0
                        break

                    n_generated_per_equation += 1

                if len(buffer) == 0:
                    n_rejected += 1
                    continue

                yield from buffer

            except NoValidSampleFoundError:
                n_rejected += 1
                continue

            n_generated += n_per_equation

            if verbose:
                pbar.update(1)
                pbar.set_postfix(reject_rate=f'{n_rejected / (n_generated + n_rejected):.2%}')

        if verbose:
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

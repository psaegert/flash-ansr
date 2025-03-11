import unittest
import tempfile
import shutil

import torch
from datasets import Dataset

from flash_ansr import FlashANSRDataset, get_path, SkeletonPool


class TestFlashANSRDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = get_path('data', 'test', 'skeleton_pool', 'val')

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_save_load(self):
        pool = SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_val.yaml'))

        pool.create(size=10)

        pool.save(
            self.save_dir,
            config=get_path('configs', 'test', 'skeleton_pool_val.yaml'))

        test_config = get_path('configs', 'test', 'dataset_val.yaml')
        dataset = FlashANSRDataset.from_config(test_config)

        dataset.data = Dataset.from_dict({
            'input_ids': [[1, 2, 3, 4], [5, 6, 7, 8]],
            'labels': [[1, 2, 3, 4], [5, 6, 7, 8]],
            'x_tensor': [[1, 2, 3, 4], [5, 6, 7, 8]]
        })

        dataset.save(self.temp_dir, config=test_config)

        loaded_config, loaded_dataset = FlashANSRDataset.load(self.temp_dir)

        for data, data_loaded in zip(dataset.data, loaded_dataset.data):
            assert data == data_loaded

    def test_iterate_step(self):
        dataset = FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml'))

        for batch in dataset.iterate(steps=2, batch_size=13):
            assert len(batch['input_ids']) == 13

    def test_iterate_size(self):
        dataset = FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml'))

        for batch in dataset.iterate(size=20, batch_size=13):
            assert len(batch['input_ids']) in [13, 7]

    def test_collate_batch(self):
        dataset = FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml'))

        for batch in dataset.iterate(steps=2, batch_size=13):
            batch = dataset.collate(batch)
            assert isinstance(batch['input_ids'], torch.Tensor)
            assert batch['x_tensors'].shape[0] == 13

    def test_collate_single(self):
        dataset = FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml'))

        for batch in dataset.iterate(size=7, batch_size=None, n_support=17):
            batch = dataset.collate(batch)
            assert isinstance(batch['input_ids'], torch.Tensor)
            assert batch['x_tensors'].shape[0] in [17, 512]  # First instance has maximum number of points to reserve maximum memory

    def test_iterate_avoid_fragmentation(self):
        dataset = FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml'))

        expected_sizes = [512, 17, 17]

        i = 0
        for batch in dataset.iterate(size=3, batch_size=None, n_support=17, avoid_fragmentation=True):
            batch = dataset.collate(batch)
            assert batch['x_tensors'].shape[0] == expected_sizes[i]
            i += 1

    def test_iterate_no_avoid_fragmentation(self):
        dataset = FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml'))

        for batch in dataset.iterate(size=3, batch_size=None, n_support=17, avoid_fragmentation=False):
            batch = dataset.collate(batch)
            assert batch['x_tensors'].shape[0] == 17

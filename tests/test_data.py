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
        with FlashANSRDataset.from_config(test_config) as dataset:
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
        with FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml')) as dataset:
            for batch in dataset.iterate(steps=2, batch_size=13):
                assert len(batch['input_ids']) == 13

    def test_iterate_size(self):
        with FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml')) as dataset:
            for batch in dataset.iterate(size=20, batch_size=13):
                assert len(batch['input_ids']) in [13, 7]

    def test_collate_batch(self):
        with FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml')) as dataset:
            for batch in dataset.iterate(steps=2, batch_size=13):
                batch = dataset.collate(batch)
                assert isinstance(batch['input_ids'], torch.Tensor)
                assert batch['x_tensors'].shape[0] == 13

    def test_collate_single(self):
        with FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml')) as dataset:
            for batch in dataset.iterate(size=7, batch_size=None, n_support=3):
                batch = dataset.collate(batch)
                assert isinstance(batch['input_ids'], torch.Tensor)
                print(batch['x_tensors'][0, :10, :10])
                for i in range(batch['x_tensors'].shape[-1]):
                    assert (batch['x_tensors'][0, :, i] != 0).sum() in [0, 3]  # 7 non-zero rows

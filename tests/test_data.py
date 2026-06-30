import unittest
import tempfile
import shutil

import torch
from datasets import Dataset

from symbolic_data import ProblemSource, load_config

from flash_ansr import FlashANSRDataset, get_path
from flash_ansr.model.tokenizer import Tokenizer


class TestFlashANSRDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_load(self):
        # The dataset now resolves a generative `source.catalog` directly from the config; the old
        # pool.create()/save() directory-load path is gone. This exercises the compiled-Dataset
        # save -> load -> equality roundtrip (catalog-independent) plus that `load` re-resolves the
        # NESTED `source.catalog` reference in the written `dataset.yaml`.
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
        # The training config uses `n_support: prior` (variable support size). To pin a fixed support
        # count for this column-occupancy assertion, build an inline fixed-count source instead.
        catalog = load_config(get_path('configs', 'test', 'catalog_val.yaml'))
        source = ProblemSource({
            'catalog': catalog,
            'sampling': {'n_support': 3, 'n_validation': 0, 'noise': 0.0},
        })
        tokenizer = Tokenizer.from_config(get_path('configs', 'test', 'tokenizer.yaml'))
        with FlashANSRDataset(source=source, tokenizer=tokenizer, padding='zero') as dataset:
            for batch in dataset.iterate(size=7, batch_size=None):
                batch = dataset.collate(batch)
                assert isinstance(batch['input_ids'], torch.Tensor)
                print(batch['x_tensors'][0, :10, :10])
                for i in range(batch['x_tensors'].shape[-1]):
                    assert (batch['x_tensors'][0, :, i] != 0).sum() in [0, 3]  # 3 support rows

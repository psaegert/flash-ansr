import unittest
import tempfile
import shutil

from datasets import Dataset

from flash_ansr import FlashANSRDataset, get_path, SkeletonPool


class TestFlashANSRDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = get_path('data', 'test', 'skeleton_pool', 'val')

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.save_dir)

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

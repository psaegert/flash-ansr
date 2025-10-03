import unittest
import datetime
import shutil

from unittest import mock

from flash_ansr import SkeletonPool, get_path
from flash_ansr.train import Trainer


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.val_skeleton_save_dir = get_path('data', 'test', 'skeleton_pool_val')

        # Create a skeleton pool
        pool = SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_val.yaml'))
        pool.create(size=10)
        pool.save(
            self.val_skeleton_save_dir,
            config=get_path('configs', 'test', 'skeleton_pool_val.yaml'))

    def tearDown(self) -> None:
        shutil.rmtree(self.val_skeleton_save_dir)

    @mock.patch('wandb.init')
    @mock.patch('wandb.log')
    def test_train(self, mock_log, mock_init):
        trainer = Trainer.from_config(get_path('configs', 'test', 'train.yaml'))

        steps = 2
        device = 'cpu'

        trainer.run(
            project_name='neural-symbolic-regression-test',
            entity='psaegert',
            name=f'pytest-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}',
            verbose=True,
            steps=steps,
            device=device,
            preprocess=False,
            checkpoint_interval=None,
            checkpoint_directory=None,
            wandb_mode="disabled",
            validate_size=10,
            validate_interval=1)

        trainer.train_dataset.shutdown()
        trainer.val_dataset.shutdown()

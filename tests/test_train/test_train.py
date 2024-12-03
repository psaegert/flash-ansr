import unittest
import datetime
import shutil

from unittest import mock

from flash_ansr import Trainer, SkeletonPool, get_path


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

        trainer.run_from_config(
            project_name='neural-symbolic-regression-test',
            entity='psaegert',
            name=f'pytest-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}',
            verbose=True,
            checkpoint_interval=None,
            checkpoint_directory=None,
            validate_interval=1)

import unittest
import shutil

from flash_ansr.eval.evaluation import Evaluation
from flash_ansr import get_path
from flash_ansr import FlashANSRTransformer
from flash_ansr.data import FlashANSRDataset
from flash_ansr.expressions import SkeletonPool


class TestEvaluation(unittest.TestCase):
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

    def test_from_config(self):
        evaluation = Evaluation.from_config(get_path('configs', 'test', 'evaluation.yaml'))

        assert evaluation is not None
        assert isinstance(evaluation, Evaluation)
        assert evaluation.n_support == 512

    def test_evaluate(self):
        evaluation = Evaluation.from_config(get_path('configs', 'test', 'evaluation.yaml'))
        nsr_transformer = FlashANSRTransformer.from_config(get_path('configs', 'test', 'nsr.yaml'))
        val_dataset = FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml'))

        evaluation.evaluate(
            model=nsr_transformer,
            dataset=val_dataset,
            size=2)

import unittest
import shutil

from flash_ansr.eval.evaluation import Evaluation
from flash_ansr import get_path, FlashANSR, FlashANSRTransformer, GenerationConfig
from flash_ansr.data import FlashANSRDataset
from flash_ansr.expressions import SkeletonPool, ExpressionSpace


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
        ansr = FlashANSR(
            expression_space=ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml')),
            flash_ansr_transformer=FlashANSRTransformer.from_config(get_path('configs', 'test', 'nsr.yaml')),
            generation_config=GenerationConfig(method='beam_search', beam_width=2, ),
            numeric_head=False,
            n_restarts=3,
            refiner_p0_noise='uniform',
            refiner_p0_noise_kwargs={'low': -5, 'high': 5},
        )

        val_dataset = FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml'))

        evaluation.evaluate(
            model=ansr,
            dataset=val_dataset,
            size=2)

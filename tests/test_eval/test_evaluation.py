import unittest
import shutil

import torch

from flash_ansr.eval.evaluation import Evaluation
from flash_ansr import (
    get_path,
    FlashANSR,
    SoftmaxSamplingConfig,
    install_model,
)
from flash_ansr.data import FlashANSRDataset
from flash_ansr.expressions import SkeletonPool


MODEL = "psaegert/flash-ansr-v19.0-6M"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        install_model(MODEL)
        evaluation = Evaluation.from_config(get_path('configs', 'test', 'evaluation.yaml'))
        ansr = FlashANSR.load(
            directory=get_path('models', MODEL),
            generation_config=SoftmaxSamplingConfig(choices=5),
            n_restarts=2,
        ).to(DEVICE)

        val_dataset = FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml'))

        results = evaluation.evaluate(
            model=ansr,
            dataset=val_dataset,
            size=2)

        for k, v in results.items():
            print(f"{k}: {len(v)}")

        assert len(set(len(v) for v in results.values())) == 1  # All results have the same length
        assert 'y_pred' in results
        assert 'predicted_expression' in results
        assert len(results['y_pred']) == 2

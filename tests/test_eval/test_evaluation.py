import os
import shutil
import unittest
import warnings

import torch

from flash_ansr.eval.evaluation import Evaluation
from flash_ansr.eval.result_store import ResultStore
from flash_ansr import (
    get_path,
    FlashANSR,
    SoftmaxSamplingConfig,
    install_model,
)
from flash_ansr.data import FlashANSRDataset
from flash_ansr.expressions import SkeletonPool


MODEL = "psaegert/flash-ansr-v23.0-3M"
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
        assert evaluation.refiner_workers is None

    def test_evaluate(self):
        install_model(MODEL)
        evaluation = Evaluation.from_config(get_path('configs', 'test', 'evaluation.yaml'))
        ansr = FlashANSR.load(
            directory=get_path('models', MODEL),
            generation_config=SoftmaxSamplingConfig(choices=5),
            n_restarts=2,
        ).to(DEVICE)

        assert ansr.refiner_workers == max(1, os.cpu_count() or 1)

        ansr_serial = FlashANSR.load(
            directory=get_path('models', MODEL),
            generation_config=SoftmaxSamplingConfig(choices=5),
            n_restarts=2,
            refiner_workers=0,
        )
        assert ansr_serial.refiner_workers == 0

        with FlashANSRDataset.from_config(get_path('configs', 'test', 'dataset_val.yaml')) as val_dataset:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in power",
                    category=RuntimeWarning,
                )
                results = evaluation.evaluate(
                    model=ansr,
                    dataset=val_dataset,
                    size=2,
                )

            for k, v in results.items():
                print(f"{k}: {len(v)}")

            assert len(set(len(v) for v in results.values())) == 1  # All results have the same length
            assert 'y_pred' in results
            assert 'predicted_expression' in results
            assert len(results['y_pred']) == 2

    def test_result_store_backfills_placeholder_defaults(self):
        store = ResultStore({"foo": [1, 2]})
        store.append({"foo": 3, "placeholder": True, "placeholder_reason": "failed"})
        snapshot = store.snapshot()
        self.assertEqual(snapshot["foo"], [1, 2, 3])
        self.assertEqual(snapshot["placeholder"], [False, False, True])
        self.assertEqual(snapshot["placeholder_reason"], [None, None, "failed"])

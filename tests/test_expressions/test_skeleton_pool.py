import unittest
from unittest import mock

from tqdm import tqdm

from flash_ansr import SkeletonPool, get_path, NoValidSampleFoundError


class TestSkeletonPool(unittest.TestCase):
    def test_contamination(self):
        pool_1 = SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_train.yaml'))
        pool_2 = SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_train.yaml'))

        pool_1.create(size=100, verbose=True)

        pool_2.register_holdout_pool(pool_1)

        for _ in tqdm(range(10000)):
            try:
                skeleton, code, constants = pool_2.sample_skeleton(new=True)
            except NoValidSampleFoundError:
                continue
            assert skeleton not in pool_1.skeletons
            pool_2.is_held_out(skeleton, constants)

    def test_register_holdout_pool_populates_holdout_manager(self):
        pool_1 = SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_train.yaml'))
        pool_2 = SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_train.yaml'))

        pool_1.create(size=15, verbose=False)

        pool_2.register_holdout_pool(pool_1)

        self.assertGreater(len(pool_2.holdout_skeletons), 0)
        sampled_skeletons = list(pool_1.skeletons)
        self.assertGreater(len(sampled_skeletons), 0)

        for skeleton in sampled_skeletons[:10]:
            code, constants = pool_1.skeleton_codes[skeleton]
            self.assertTrue(pool_2.is_held_out(list(skeleton), constants, code))

    def test_is_held_out_returns_true_on_overflow(self):
        pool = SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_train.yaml'))

        for _ in range(20):
            try:
                skeleton, code, constants = pool.sample_skeleton(new=True, decontaminate=False)
                break
            except NoValidSampleFoundError:
                continue
        else:
            self.fail("Failed to sample skeleton for overflow test")

        with mock.patch.object(pool.holdout_manager, 'is_held_out', side_effect=OverflowError):
            self.assertTrue(pool.is_held_out(skeleton, constants, code))

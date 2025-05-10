import unittest

from flash_ansr import SkeletonPool, get_path, NoValidSampleFoundError


class TestSkeletonPool(unittest.TestCase):
    def test_contamination(self):
        pool_1 = SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_train.yaml'))
        pool_2 = SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_train.yaml'))

        pool_1.create(size=100, verbose=True)

        pool_2.register_holdout_pool(pool_1)

        for _ in range(10000):
            try:
                skeleton, code, constants = pool_2.sample_skeleton(new=True)
            except NoValidSampleFoundError:
                continue
            assert skeleton not in pool_1.skeletons
            pool_2.is_held_out(skeleton, constants)

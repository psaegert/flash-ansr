import unittest
import pandas as pd

from simplipy import SimpliPyEngine

from flash_ansr.compat.convert_data import SOOSEParser, FeynmanParser, NguyenParser
from flash_ansr import SkeletonPool, get_path


class TestConvertData(unittest.TestCase):
    def test_import_test_data_soose(self):
        df = pd.DataFrame({
            'eq': [
                '-x_2 + log(x_1 - x_2**4)',
                'cos(x_2*tan(x_1**2 - x_2))']
        })

        parser = SOOSEParser()

        skeleton_pool = parser.parse_data(
            test_set_df=df,
            simplipy_engine=SimpliPyEngine.from_config(get_path('configs', 'test', 'simplipy_engine.yaml')),
            base_skeleton_pool=SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_test.yaml')))

        self.assertIsInstance(skeleton_pool, SkeletonPool)

    def test_import_test_data_feynman(self):
        df = pd.DataFrame({
            'Formula': [
                '-x_2 + log(x_1 - x_2**4)',
                'cos(x_2*tan(x_1**2 - x_2))'],
            '# variables': [2, 2],
        })

        parser = FeynmanParser()

        skeleton_pool = parser.parse_data(
            test_set_df=df,
            simplipy_engine=SimpliPyEngine.from_config(get_path('configs', 'test', 'simplipy_engine.yaml')),
            base_skeleton_pool=SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_test.yaml')))

        self.assertIsInstance(skeleton_pool, SkeletonPool)

    def test_import_test_data_nguyen(self):
        df = pd.DataFrame({
            'Equation': [
                '-x_2 + log(x_1 - x_2**4)',
                'cos(x_2*tan(x_1**2 - x_2))']
        })

        parser = NguyenParser()

        skeleton_pool = parser.parse_data(
            test_set_df=df,
            simplipy_engine=SimpliPyEngine.from_config(get_path('configs', 'test', 'simplipy_engine.yaml')),
            base_skeleton_pool=SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_test.yaml')))

        self.assertIsInstance(skeleton_pool, SkeletonPool)

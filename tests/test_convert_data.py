import unittest
import pandas as pd

from simplipy import SimpliPyEngine

from flash_ansr.convert_data import SOOSEParser, FeynmanParser, NguyenParser, FastSRBParser
from flash_ansr import LampleChartonCatalog, get_path


class TestConvertData(unittest.TestCase):
    def test_import_test_data_soose(self):
        df = pd.DataFrame({
            'eq': [
                '-x_2 + log(x_1 - x_2**4)',
                'cos(x_2*tan(x_1**2 - x_2))']
        })

        parser = SOOSEParser()

        catalog = parser.parse_data(
            test_set_df=df,
            simplipy_engine=SimpliPyEngine.load('dev_7-3', install=True),
            base_catalog=LampleChartonCatalog.from_config(get_path('configs', 'test', 'catalog_test.yaml')))

        self.assertIsInstance(catalog, LampleChartonCatalog)

    def test_import_test_data_feynman(self):
        df = pd.DataFrame({
            'Formula': [
                '-x_2 + log(x_1 - x_2**4)',
                'cos(x_2*tan(x_1**2 - x_2))'],
            '# variables': [2, 2],
        })

        parser = FeynmanParser()

        catalog = parser.parse_data(
            test_set_df=df,
            simplipy_engine=SimpliPyEngine.load('dev_7-3', install=True),
            base_catalog=LampleChartonCatalog.from_config(get_path('configs', 'test', 'catalog_test.yaml')))

        self.assertIsInstance(catalog, LampleChartonCatalog)

    def test_import_test_data_nguyen(self):
        df = pd.DataFrame({
            'Equation': [
                '-x_2 + log(x_1 - x_2**4)',
                'cos(x_2*tan(x_1**2 - x_2))']
        })

        parser = NguyenParser()

        catalog = parser.parse_data(
            test_set_df=df,
            simplipy_engine=SimpliPyEngine.load('dev_7-3', install=True),
            base_catalog=LampleChartonCatalog.from_config(get_path('configs', 'test', 'catalog_test.yaml')))

        self.assertIsInstance(catalog, LampleChartonCatalog)

    def test_import_test_data_fastsrb(self):
        df = pd.DataFrame({
            'prepared': [
                'v1 * v2 * v3',
                'sin(v1) / (v2 + v3)'
            ]
        })

        parser = FastSRBParser()

        catalog = parser.parse_data(
            test_set_df=df,
            simplipy_engine=SimpliPyEngine.load('dev_7-3', install=True),
            base_catalog=LampleChartonCatalog.from_config(get_path('configs', 'test', 'catalog_test.yaml')))

        self.assertIsInstance(catalog, LampleChartonCatalog)

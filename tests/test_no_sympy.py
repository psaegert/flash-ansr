"""The SymPy simplification path is REMOVED from flash-ansr (owner ruling, 2026-08-18).

SymPy simplification was an ablation of the product simplifier (SimpliPy), and the standing
rule is that production code stays clean and minimal: experiments and ablations branch off it
or patch it, they never live inside it. This module is the executable statement of that
removal. It pins three things:

1. **The code is gone**: no ``flash_ansr.utils.sympy_timeout`` module, no ``sympy`` import
   anywhere in the package, no ``[sympy]`` extra in ``pyproject.toml``.
2. **A config that asks for it FAILS LOUDLY** -- both on flash-ansr's OWN ``simplify``
   selector (the generation config / model post-processing) and on the ``simplify`` value
   flash-ansr passes THROUGH to symbolic-data's catalog. Never a silent fallback to a
   different simplifier: a fallback would swap the canonicalizer under a config that
   explicitly named one, which is a silent behaviour change, not a removal.
3. **No shipped config still requests it.**

symbolic-data keeps its own ``simplify='sympy'`` path; the ruling covers flash-ansr, so the
flash-ansr side is a refusal at the boundary, not an edit to symbolic-data.
"""
import importlib
import os
import re
import unittest

import yaml

from flash_ansr import get_path
from flash_ansr.utils.generation import SoftmaxSamplingConfig, create_generation_config


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _package_py_files() -> list[str]:
    src = os.path.join(_repo_root(), 'src', 'flash_ansr')
    return [
        os.path.join(dirpath, name)
        for dirpath, _, filenames in os.walk(src)
        for name in filenames
        if name.endswith('.py')
    ]


class TestSympyCodeIsGone(unittest.TestCase):
    def test_sympy_timeout_module_removed(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module('flash_ansr.utils.sympy_timeout')

    def test_no_sympy_import_in_package(self) -> None:
        # An `import sympy` / `from sympy import ...` anywhere in the package would mean the
        # path came back. Matches the import statement only, so prose in a docstring or an
        # error message that NAMES the removal does not false-positive.
        pattern = re.compile(r'^\s*(?:import\s+sympy|from\s+sympy\b)', re.MULTILINE)
        offenders = []
        for path in _package_py_files():
            with open(path, encoding='utf-8') as handle:
                if pattern.search(handle.read()):
                    offenders.append(os.path.relpath(path, _repo_root()))
        self.assertEqual(offenders, [], f"sympy is imported in {offenders}")

    def test_pyproject_has_no_sympy_extra(self) -> None:
        with open(os.path.join(_repo_root(), 'pyproject.toml'), encoding='utf-8') as handle:
            pyproject = handle.read()
        self.assertNotRegex(pyproject, r'(?m)^\s*sympy\s*=\s*\[', "the [sympy] optional-dependency extra is back")
        self.assertNotRegex(pyproject, r'(?m)^\s*"sympy[><=~]', "sympy is declared as a dependency")


class TestOwnSimplifySelectorRefusesSympy(unittest.TestCase):
    """flash-ansr's OWN `simplify` selector: the generation config and the model's
    post-processing. These were flash-ansr code, and they are the sites the ruling removes."""

    def _assert_names_the_removal(self, message: str) -> None:
        lowered = message.lower()
        self.assertIn('sympy', lowered)
        self.assertIn('removed', lowered)
        # No silent fallback: the error must say the request is refused, not quietly re-served.
        self.assertIn('refused', lowered)

    def test_softmax_sampling_config_refuses_sympy(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            SoftmaxSamplingConfig(simplify='sympy')
        self._assert_names_the_removal(str(ctx.exception))

    def test_create_generation_config_refuses_sympy(self) -> None:
        # The config-file route (`generation_config: {method: softmax_sampling, kwargs: {...}}`).
        with self.assertRaises(ValueError) as ctx:
            create_generation_config(method='softmax_sampling', simplify='sympy')
        self._assert_names_the_removal(str(ctx.exception))

    def test_softmax_sampling_config_still_accepts_bools(self) -> None:
        self.assertIs(SoftmaxSamplingConfig(simplify=True).simplify, True)
        self.assertIs(SoftmaxSamplingConfig(simplify=False).simplify, False)

    def test_postprocess_sampled_refuses_sympy(self) -> None:
        from flash_ansr import FlashANSRModel

        model = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))
        with self.assertRaises(ValueError) as ctx:
            model._postprocess_sampled([], [], simplify='sympy')
        self._assert_names_the_removal(str(ctx.exception))

    def test_sample_top_kp_refuses_sympy_before_decoding(self) -> None:
        # `return_raw=True` skips post-processing entirely, so the guard cannot live only in
        # `_postprocess_sampled`: the request must be refused at the entry point too.
        import torch

        from flash_ansr import FlashANSRModel

        model = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))
        with self.assertRaises(ValueError) as ctx:
            model.sample_top_kp(torch.rand(13, 11), choices=1, max_len=4, simplify='sympy', return_raw=True)
        self._assert_names_the_removal(str(ctx.exception))


class TestCatalogPassthroughRefusesSympy(unittest.TestCase):
    """The `simplify` key in a CATALOG config is symbolic-data's parameter, which flash-ansr
    only passes through. symbolic-data is not edited; flash-ansr refuses to build a data
    source that would route into it."""

    def _dataset_config(self, simplify) -> dict:
        catalog = yaml.safe_load(open(get_path('configs', 'test', 'catalog_train.yaml'), encoding='utf-8'))
        catalog['simplify'] = simplify
        return {
            "source": {"catalog": catalog,
                       "sampling": {"n_support": "prior", "n_validation": 0, "noise": 0.0}},
            "tokenizer": get_path('configs', 'test', 'tokenizer.yaml'),
            "padding": "zero",
        }

    def test_dataset_from_config_refuses_sympy_catalog(self) -> None:
        from flash_ansr import FlashANSRDataset

        with self.assertRaises(ValueError) as ctx:
            FlashANSRDataset.from_config(self._dataset_config('sympy'))
        message = str(ctx.exception).lower()
        self.assertIn('sympy', message)
        self.assertIn('removed', message)
        self.assertIn('refused', message)

    def test_dataset_from_config_still_accepts_bool_catalogs(self) -> None:
        from flash_ansr import FlashANSRDataset

        with FlashANSRDataset.from_config(self._dataset_config(True)) as dataset:
            self.assertTrue(dataset.source.catalog.simplify)


class TestShippedConfigs(unittest.TestCase):
    def test_no_shipped_config_requests_sympy(self) -> None:
        configs_dir = get_path('configs')
        offenders = []
        for dirpath, _, filenames in os.walk(configs_dir):
            for name in filenames:
                if not name.endswith(('.yaml', '.yml')):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding='utf-8') as handle:
                    if re.search(r'(?m)^\s*simplify:\s*[\'"]?sympy', handle.read()):
                        offenders.append(os.path.relpath(path, _repo_root()))
        self.assertEqual(sorted(offenders), [], f"shipped configs still request the removed sympy path: {offenders}")


if __name__ == '__main__':
    unittest.main()

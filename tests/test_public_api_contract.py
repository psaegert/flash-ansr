"""flash-ansr <-> srbf public API contract test.

After the v0.6 repo split, ``srbf`` (the carved-out eval framework) imports a fixed set of
symbols from ``flash-ansr`` and depends on their signatures staying stable. Today srbf lives in the
same tree (``src/flash_ansr/eval``, ``/baselines``, ``/compat``); once carved out it becomes a
downstream package that pins ``flash-ansr`` as a dependency. This test freezes the surface so
flash-ansr cannot merge a contract break unknowingly *before* the carve makes such a break a
cross-repo regression.

Design choices (deliberate):

* **Import via the path srbf actually uses.** Several contract symbols are NOT re-exported from the
  ``flash_ansr`` package root: the scoring primitives live in ``flash_ansr.scoring``,
  ``mask_unused_variable_columns`` in ``flash_ansr.utils.tensor_ops``, and
  ``normalize_skeleton`` / ``normalize_expression`` in ``flash_ansr.expressions.normalization``.
  srbf imports them via those submodule paths (see ``eval/run_config.py``, ``eval/model_adapters.py``,
  ``eval/data_sources.py``, ``baselines/*``), so the contract is pinned at those paths. Adding root
  re-exports later is optional API polish, not required for the contract to hold.
* **Signatures are frozen by required-parameter NAMES, not by exact string equality.** A contract
  test must catch the breaking direction (a parameter srbf passes by keyword is removed or renamed)
  without false-positiving on the safe direction (a new optional parameter is added). So each check
  asserts the expected parameter names are a subset of the live signature.
* **The private ``_fit_*`` coupling is a KNOWN GAP, recorded not frozen-as-public.** srbf's
  ``FlashANSRAdapter`` drives the model through the *private* ``_fit_generate`` / ``_fit_refine`` /
  ``_apply_fit_result`` methods (``eval/model_adapters.py``), not the public ``.fit`` /
  ``compile_results``. That is the seam srbf really relies on, so we assert those methods exist to
  catch a silent rename, but mark them explicitly as not-yet-public (§5 should decide whether to
  promote them or test them directly across the carve).
"""
from __future__ import annotations

import inspect

import pytest


def _params(obj) -> set[str]:
    """Return the parameter names of a callable's signature."""
    return set(inspect.signature(obj).parameters)


def _assert_has_params(obj, expected: set[str]) -> None:
    """Assert ``expected`` parameter names are all present in ``obj``'s signature.

    Subset (not equality) so that adding a new optional parameter to ``obj`` -- a non-breaking
    change for srbf -- does not fail the contract, while removing/renaming one that srbf passes does.
    """
    actual = _params(obj)
    missing = expected - actual
    assert not missing, f"{obj!r} lost contract parameter(s) {sorted(missing)}; live params={sorted(actual)}"


class TestScoringPrimitives:
    """The unified candidate-scoring primitives. srbf's baselines import
    these from ``flash_ansr.scoring`` (``baselines/skeleton_pool_model.py``, ``baselines/brute_force_model.py``)."""

    def test_importable(self):
        from flash_ansr.scoring import (  # noqa: F401
            FLOAT64_EPS,
            compute_fvu,
            normalize_variance,
            score_from_fvu,
        )

    def test_float64_eps_is_float(self):
        from flash_ansr.scoring import FLOAT64_EPS

        assert isinstance(FLOAT64_EPS, float)

    def test_signatures(self):
        from flash_ansr.scoring import compute_fvu, normalize_variance, score_from_fvu

        _assert_has_params(compute_fvu, {"loss", "sample_count", "variance"})
        _assert_has_params(normalize_variance, {"variance"})
        _assert_has_params(
            score_from_fvu,
            {
                "fvu",
                "complexity",
                "constant_count",
                "log_prob",
                "length_penalty",
                "constants_penalty",
                "likelihood_penalty",
            },
        )

    def test_constant_helpers(self):
        # count_constants / is_constant_token were promoted to the scoring surface by the
        # constant-helper consolidation (§2); both srbf baselines now import them.
        from flash_ansr.scoring import count_constants, is_constant_token

        _assert_has_params(is_constant_token, {"token"})
        _assert_has_params(count_constants, {"expression"})


class TestExpressionNormalization:
    """``normalize_skeleton`` / ``normalize_expression`` -- srbf's adapters + data sources import
    these from ``simplipy`` (where expression-token normalization lives post-carve)."""

    def test_signatures(self):
        from simplipy import normalize_expression, normalize_skeleton

        _assert_has_params(normalize_skeleton, {"tokens"})
        _assert_has_params(normalize_expression, {"tokens"})


class TestTensorOps:
    """``mask_unused_variable_columns`` -- srbf imports it from ``flash_ansr.utils.tensor_ops``
    (``eval/data_sources.py``); the keyword-only args are part of the contract."""

    def test_signature(self):
        from flash_ansr.utils.tensor_ops import mask_unused_variable_columns

        _assert_has_params(
            mask_unused_variable_columns,
            {"arrays", "variables", "skeleton_tokens", "padding"},
        )


class TestConfigIO:
    """``load_config`` / ``save_config`` -- srbf imports from ``flash_ansr.utils.config_io``
    (``eval/run_config.py``)."""

    def test_signatures(self):
        from flash_ansr.utils.config_io import load_config, save_config

        _assert_has_params(load_config, {"config", "resolve_paths"})
        _assert_has_params(save_config, {"config", "directory", "filename"})


class TestPaths:
    """``get_path`` / ``get_root`` -- the project-root resolution the research tier + (post-split)
    cross-repo asset story depend on (``flash_ansr.utils.paths``; ``get_root`` added in Step-1 prep)."""

    def test_signatures(self):
        from flash_ansr.utils.paths import get_path, get_root, substitute_root_path

        _assert_has_params(get_path, {"filename", "create"})
        # get_root() takes no parameters; importable + callable is the contract.
        assert callable(get_root)
        # substitute_root_path resolves {{ROOT}} placeholders; imported by 5 srbf-bound
        # modules (both baselines + eval engine/provenance/result_store/run_config).
        _assert_has_params(substitute_root_path, {"path"})


class TestGenerationConfigs:
    """Generation configs -- srbf builds them via the ``create_generation_config`` factory
    (``eval/run_config.py``); the concrete classes + base + union alias are the public surface."""

    def test_factory_signature(self):
        from flash_ansr.utils.generation import create_generation_config

        _assert_has_params(create_generation_config, {"method"})

    def test_config_classes_importable(self):
        from flash_ansr.utils.generation import (  # noqa: F401
            BeamSearchConfig,
            GenerationConfig,
            GenerationConfigBase,
            MCTSGenerationConfig,
            SoftmaxSamplingConfig,
        )

        assert GenerationConfig is not None  # union alias

    @pytest.mark.parametrize("name", ["BeamSearchConfig", "SoftmaxSamplingConfig", "MCTSGenerationConfig"])
    def test_concrete_configs_are_classes(self, name):
        import flash_ansr.utils.generation as gen

        assert inspect.isclass(getattr(gen, name))


class TestDataset:
    """``FlashANSRDataset`` / ``FlashANSRPreprocessor`` -- srbf imports both from ``flash_ansr.data``
    (``eval/run_config.py``, ``eval/data_sources.py``); the dataset is also required by the
    PySR/NeSymReS adapters for eval-data generation (§2)."""

    def test_importable_and_class(self):
        from flash_ansr.data import FlashANSRDataset, FlashANSRPreprocessor

        assert inspect.isclass(FlashANSRDataset)
        assert inspect.isclass(FlashANSRPreprocessor)


class TestResultsPayload:
    """The result-serialisation surface srbf's SkeletonPoolModel baseline imports from
    ``flash_ansr.results`` (``baselines/skeleton_pool_model.py``)."""

    def test_importable(self):
        from flash_ansr.results import (  # noqa: F401
            RESULTS_FORMAT_VERSION,
            deserialize_results_payload,
            load_results_payload,
            save_results_payload,
            serialize_results_payload,
        )

    def test_format_version_is_int(self):
        from flash_ansr.results import RESULTS_FORMAT_VERSION

        assert isinstance(RESULTS_FORMAT_VERSION, int)


class TestRefiner:
    """``Refiner`` / ``ConvergenceError`` -- srbf's baselines instantiate the refiner and catch the
    error (``flash_ansr.refine``)."""

    def test_importable(self):
        from flash_ansr.refine import ConvergenceError, Refiner

        assert inspect.isclass(Refiner)
        assert issubclass(ConvergenceError, Exception)

    def test_refiner_signature(self):
        from flash_ansr.refine import Refiner

        _assert_has_params(Refiner.__init__, {"simplipy_engine", "n_variables"})


class TestSkeletonPool:
    """``SkeletonPool`` / ``NoValidSampleFoundError`` -- srbf's SkeletonPoolModel baseline imports
    these from ``symbolic_data`` (the data layer owner post-carve)."""

    def test_importable(self):
        from symbolic_data import NoValidSampleFoundError, SkeletonPool

        assert inspect.isclass(SkeletonPool)
        assert issubclass(NoValidSampleFoundError, Exception)


class TestFlashANSR:
    """The product class. srbf constructs it via the ``load`` classmethod (``eval/run_config.py``)
    and reads predictions via ``predict`` (``eval/model_adapters.py``). ``fit`` / ``compile_results``
    / ``results`` are public-contract surface used by the product + research tiers."""

    def test_importable(self):
        from flash_ansr.flash_ansr import FlashANSR

        assert inspect.isclass(FlashANSR)

    def test_load_is_the_construction_entrypoint(self):
        # srbf builds every FlashANSR model via FlashANSR.load(...) (run_config.py); the plan's §5
        # method list omits .load -- it is the actual entrypoint and belongs in the freeze.
        from flash_ansr.flash_ansr import FlashANSR

        assert isinstance(inspect.getattr_static(FlashANSR, "load"), classmethod)
        _assert_has_params(
            FlashANSR.load,
            {"directory", "generation_config", "n_restarts", "refiner_method", "device"},
        )

    def test_predict_signature(self):
        from flash_ansr.flash_ansr import FlashANSR

        _assert_has_params(FlashANSR.predict, {"X", "nth_best_beam", "nth_best_constants"})

    def test_fit_signature(self):
        from flash_ansr.flash_ansr import FlashANSR

        _assert_has_params(FlashANSR.fit, {"X", "y", "variable_names"})

    def test_compile_results_signature(self):
        from flash_ansr.flash_ansr import FlashANSR

        _assert_has_params(
            FlashANSR.compile_results,
            {"length_penalty", "constants_penalty", "likelihood_penalty"},
        )

    def test_results_attribute_present(self):
        # Plan §5 names FlashANSR.results; it is an instance attribute set in __init__ (the live
        # srbf coupling is the FitResult.results dataclass field, reached via the private _fit_*
        # seam above). Guard the attribute assignment without instantiating the heavy model.
        from flash_ansr.flash_ansr import FlashANSR

        assert "results" in FlashANSR.__init__.__code__.co_names

    def test_private_fit_coupling_is_present(self):
        """KNOWN §5 GAP -- not a public contract, recorded here so a silent rename is caught.

        srbf's FlashANSRAdapter drives the model through these PRIVATE methods rather than the public
        ``.fit`` (``eval/model_adapters.py``: ``_fit_generate`` -> ``_fit_refine`` ->
        ``_apply_fit_result``). Until §5 decides whether to promote this seam to public or test it
        directly across the carve, this asserts the methods exist so an accidental rename breaks here.
        """
        from flash_ansr.flash_ansr import FlashANSR

        for private in ("_fit_generate", "_fit_refine", "_apply_fit_result"):
            assert callable(getattr(FlashANSR, private, None)), (
                f"FlashANSR.{private} is the private seam srbf's adapter relies on (known §5 gap)"
            )


class TestPackageRootReExports:
    """Symbols the ``flash_ansr`` package root currently re-exports (``__init__.py``). Pinning these
    guards the convenience surface the research/product tiers use directly."""

    @pytest.mark.parametrize(
        "name",
        [
            "FlashANSR",
            "FlashANSRDataset",
            "Refiner",
            "ConvergenceError",
            "SkeletonPool",
            "NoValidSampleFoundError",
            "GenerationConfig",
            "GenerationConfigBase",
            "BeamSearchConfig",
            "SoftmaxSamplingConfig",
            "MCTSGenerationConfig",
            "create_generation_config",
            "get_path",
            "get_root",
            "load_config",
            "save_config",
        ],
    )
    def test_root_reexport_present(self, name):
        import flash_ansr

        assert hasattr(flash_ansr, name), f"flash_ansr.{name} regressed from the package root surface"

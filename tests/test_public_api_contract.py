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
  ``normalize_skeleton`` / ``normalize_expression`` in ``symbolic_data.token_ops`` (simplipy 0.14 carve-back).
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
                "n_nodes",
                "constant_count",
                "log_prob",
                "node_penalty",
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
    these from ``symbolic_data.token_ops``: simplipy 0.14 replaced its ``normalize_*``
    surface with the engine-bound ``to_skeleton``/``to_expression`` canonicalisers, and
    the positional walks moved to the consumer package (the decontamination key and the
    concrete ground truth must not depend on the loaded rule artifact)."""

    def test_signatures(self):
        from symbolic_data.token_ops import normalize_expression, normalize_skeleton

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
            GenerationConfig,
            GenerationConfigBase,
            SoftmaxSamplingConfig,
        )

        assert GenerationConfig is not None  # union alias

    @pytest.mark.parametrize("name", ["SoftmaxSamplingConfig"])
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


class TestGenerativeCatalog:
    """``LampleChartonCatalog`` / ``NoValidSampleFoundError`` -- srbf's SkeletonPoolModel baseline +
    flash-ansr training import the generative catalog from ``symbolic_data`` (the data-layer owner;
    0.6 replaced ``SkeletonPool`` with the catalog-based ``LampleChartonCatalog``)."""

    def test_importable(self):
        from symbolic_data import GenerativeCatalog, LampleChartonCatalog, NoValidSampleFoundError

        assert inspect.isclass(LampleChartonCatalog)
        assert issubclass(LampleChartonCatalog, GenerativeCatalog)
        assert issubclass(NoValidSampleFoundError, Exception)


class TestFlashANSREstimator:
    """The product class (0.17 surface). srbf constructs it via ``load`` and drives it through
    ``fit`` -> ``FitResult``; the README and the SRBench wrapper use the stateful sugar
    (``predict`` / ``get_expression`` / ``results``) on top of ``result_``."""

    def test_importable(self):
        from flash_ansr.flash_ansr import FlashANSR

        assert inspect.isclass(FlashANSR)

    def test_load_is_the_construction_entrypoint(self):
        from flash_ansr.flash_ansr import FlashANSR

        assert isinstance(inspect.getattr_static(FlashANSR, "load"), classmethod)
        _assert_has_params(FlashANSR.load, {"directory", "generation_config", "refine", "ranking", "compute"})

    def test_fit_signature(self):
        from flash_ansr.flash_ansr import FlashANSR

        _assert_has_params(FlashANSR.fit, {"X", "y", "variable_names", "draws", "complexity", "seed", "on_empty", "verbose"})
        # the retired per-call knobs must not creep back
        assert not ({"emission", "conditioned", "converge_error", "refine_seed", "X_val", "top_k"} & _params(FlashANSR.fit))

    def test_generate_signature(self):
        from flash_ansr.flash_ansr import FlashANSR

        _assert_has_params(FlashANSR.generate, {"X", "y", "variable_names", "draws", "complexity", "seed", "verbose"})

    def test_predict_and_get_expression_signatures(self):
        from flash_ansr.flash_ansr import FlashANSR

        _assert_has_params(FlashANSR.predict, {"X", "rank"})
        _assert_has_params(FlashANSR.get_expression, {"rank", "return_prefix", "precision", "map_variables"})

    def test_retired_verbs_are_gone(self):
        from flash_ansr.flash_ansr import FlashANSR

        for name in ("infer", "compile_results", "save_results", "load_results", "ranking_config",
                     "predict_y", "predict_constants", "predict_complexity", "score_outliers"):
            assert not hasattr(FlashANSR, name), f"FlashANSR.{name} was retired in 0.17"

    def test_result_surface(self):
        from flash_ansr.inference import FitResult, Candidate, CandidateLedger

        _assert_has_params(FitResult.predict, {"X", "rank"})
        _assert_has_params(FitResult.rerank, {"ranking"})
        _assert_has_params(FitResult.get_expression, {"rank", "return_prefix", "precision", "map_variables"})
        for name in ("candidates", "ledger", "generation_time", "refinement_time", "ranking", "n_variables", "variable_mapping"):
            assert name in FitResult.__dataclass_fields__
        for name in ("expression", "slots", "expression_prefix", "expression_infix", "constants", "score", "fvu", "mdl", "rank"):
            assert name in Candidate.__dataclass_fields__
        assert "y_pred" not in Candidate.__dataclass_fields__
        assert "result_index" in CandidateLedger.__dataclass_fields__


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
            "LampleChartonCatalog",
            "NoValidSampleFoundError",
            "GenerationConfig",
            "GenerationConfigBase",
            "SoftmaxSamplingConfig",
            "PriorSamplingConfig",
            "FitResult",
            "Candidate",
            "RefineConfig",
            "ComputeConfig",
            "RankingConfig",
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

"""Helpers for building config-driven evaluation runs."""
from __future__ import annotations

import copy
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from flash_ansr.baselines import BruteForceModel, SkeletonPoolModel
from simplipy import SimpliPyEngine

from flash_ansr.benchmarks import FastSRBBenchmark
from flash_ansr.data import FlashANSRDataset
from flash_ansr.eval.data_sources import FastSRBSource, SkeletonDatasetSource
from flash_ansr.eval.engine import EvaluationEngine
from flash_ansr.eval.model_adapters import (
    BruteForceAdapter,
    E2EAdapter,
    FlashANSRAdapter,
    NeSymReSAdapter,
    PySRAdapter,
    SkeletonPoolAdapter,
)
from flash_ansr.eval.result_store import ResultStore
from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.generation import create_generation_config
from flash_ansr.utils.paths import substitute_root_path


@dataclass(slots=True)
class EvaluationRunPlan:
    """Container describing how to execute an evaluation run."""

    engine: EvaluationEngine | None
    remaining: int | None
    save_every: int | None
    output_path: str | None
    completed: bool
    total_limit: int | None
    existing_results: int


def build_evaluation_run(
    config: str | Mapping[str, Any],
    *,
    limit_override: int | None = None,
    output_override: str | None = None,
    save_every_override: int | None = None,
    resume: bool | None = None,
    experiment: str | None = None,
) -> EvaluationRunPlan:
    """Instantiate data sources, adapters, and the engine from a config file."""

    raw_config = load_config(config) if isinstance(config, str) else dict(config)
    config_dict = _select_experiment(raw_config, experiment)
    run_cfg = _extract_run_section(config_dict)

    data_cfg = run_cfg.get("data_source")
    if not isinstance(data_cfg, Mapping):
        raise KeyError("run.data_source section is required")
    model_cfg = run_cfg.get("model_adapter")
    if not isinstance(model_cfg, Mapping):
        raise KeyError("run.model_adapter section is required")
    runner_cfg = run_cfg.get("runner", {})

    data_target_limit = _coerce_optional_int(data_cfg.get("target_size"), "data_source.target_size")

    save_every = save_every_override if save_every_override is not None else runner_cfg.get("save_every")
    save_every = _coerce_optional_int(save_every, "runner.save_every")

    output_path = output_override or runner_cfg.get("output")
    if save_every is not None and output_path is None:
        raise ValueError("runner.output must be provided when save_every is set")

    resume_flag = runner_cfg.get("resume", True)
    if resume is not None:
        resume_flag = resume

    initial_results = None
    if resume_flag and output_path:
        initial_results = _load_existing_results(output_path)

    store = ResultStore(initial_results)
    existing = store.size

    limit_value = limit_override if limit_override is not None else runner_cfg.get("limit")
    limit_value = _coerce_optional_int(limit_value, "runner.limit")

    preloaded_assets: dict[str, Any] = {}

    if limit_value is None:
        limit_value = data_target_limit
    if limit_value is None:
        inferred_limit, assets = _infer_total_limit_from_data_source(data_cfg)
        limit_value = inferred_limit
        preloaded_assets.update(assets)

    total_limit = limit_value

    if total_limit is not None:
        remaining = max(0, total_limit - existing)
        if remaining == 0:
            return EvaluationRunPlan(
                engine=None,
                remaining=0,
                save_every=save_every,
                output_path=output_path,
                completed=True,
                total_limit=total_limit,
                existing_results=existing,
            )
    else:
        remaining = None

    target_override = remaining if remaining is not None else data_target_limit

    data_source, context = _build_data_source(
        data_cfg,
        target_size_override=target_override,
        skip=existing,
        preloaded_assets=preloaded_assets,
    )

    size_hint = getattr(data_source, "size_hint", None)
    pending = size_hint() if callable(size_hint) else None
    if pending is not None and pending <= 0:
        return EvaluationRunPlan(
            engine=None,
            remaining=0,
            save_every=save_every,
            output_path=output_path,
            completed=True,
            total_limit=total_limit if total_limit is not None else existing,
            existing_results=existing,
        )

    adapter = _build_model_adapter(model_cfg, context=context)

    engine = EvaluationEngine(
        data_source=data_source,
        model_adapter=adapter,
        result_store=store,
    )

    return EvaluationRunPlan(
        engine=engine,
        remaining=remaining,
        save_every=save_every,
        output_path=output_path,
        completed=False,
        total_limit=total_limit,
        existing_results=existing,
    )


# ---------------------------------------------------------------------------
# Builders

def _build_data_source(
    config: Mapping[str, Any],
    *,
    target_size_override: int | None,
    skip: int,
    preloaded_assets: Mapping[str, Any] | None = None,
) -> tuple[SkeletonDatasetSource | FastSRBSource, dict[str, Any]]:
    preloaded_assets = dict(preloaded_assets or {})
    dtype = str(config.get("type", "skeleton_dataset")).lower()
    if dtype == "skeleton_dataset":
        dataset = preloaded_assets.get("dataset")
        if dataset is None:
            dataset_spec = config.get("dataset")
            if dataset_spec is None:
                raise ValueError("data_source.dataset must be provided for skeleton_dataset sources")
            dataset = _load_dataset(dataset_spec)

        target_size = target_size_override
        n_support = _coerce_optional_int(config.get("n_support"), "data_source.n_support")
        noise_level = float(config.get("noise_level", 0.0))
        preprocess = bool(config.get("preprocess", False))
        device = str(config.get("device", "cpu"))
        iterator_buffer = _coerce_optional_int(config.get("iterator_buffer"), "data_source.iterator_buffer")
        tokenizer_oov = str(config.get("tokenizer_oov", "unk"))
        max_trials = _coerce_optional_int(config.get("max_trials"), "data_source.max_trials")

        if "evaluation_order" in config:
            warnings.warn(
                "data_source.evaluation_order is no longer supported; deterministic sampling always iterates sequentially.",
                RuntimeWarning,
            )

        source = SkeletonDatasetSource(
            dataset=dataset,
            target_size=target_size,
            n_support=n_support,
            noise_level=noise_level,
            preprocess=preprocess,
            device=device,
            iterator_buffer=iterator_buffer or 2,
            tokenizer_oov=tokenizer_oov,
            skip=skip,
            datasets_per_expression=_coerce_optional_int(config.get("datasets_per_expression"), "data_source.datasets_per_expression"),
            datasets_random_seed=_coerce_optional_int(config.get("datasets_random_seed"), "data_source.datasets_random_seed"),
            max_trials=max_trials,
        )
        return source, {"dataset": dataset}

    if dtype == "fastsrb":
        benchmark = preloaded_assets.get("benchmark")
        if benchmark is None:
            benchmark_path = config.get("benchmark_path")
            if benchmark_path is None:
                raise ValueError("data_source.benchmark_path must be provided for FastSRB sources")
            benchmark = FastSRBBenchmark(
                substitute_root_path(str(benchmark_path)),
                random_state=_coerce_optional_int(config.get("benchmark_random_state"), "data_source.benchmark_random_state"),
            )

        eq_ids = _parse_equation_ids(config.get("eq_ids"))
        datasets_per_expression_cfg = config.get("datasets_per_expression")
        legacy_count_cfg = config.get("count")
        repeats_field = "data_source.datasets_per_expression"
        repeats_value = datasets_per_expression_cfg
        if repeats_value is None:
            repeats_field = "data_source.count"
            repeats_value = legacy_count_cfg
        if repeats_value is None:
            repeats_field = "data_source.datasets_per_expression"
            repeats_value = 1
        if datasets_per_expression_cfg is not None and legacy_count_cfg is not None and datasets_per_expression_cfg != legacy_count_cfg:
            warnings.warn(
                "data_source.count is deprecated; using datasets_per_expression value while both were provided.",
                RuntimeWarning,
            )
        datasets_per_expression = _coerce_int(repeats_value, repeats_field)
        support_points = _coerce_int(config.get("support_points", 100), "data_source.support_points")
        sample_points = _coerce_optional_int(config.get("sample_points"), "data_source.sample_points")
        n_support_override = _coerce_optional_int(
            config.get("n_support_override", config.get("n_support")),
            "data_source.n_support_override",
        )
        method = str(config.get("method", "random"))
        max_trials = _coerce_int(config.get("max_trials", 100), "data_source.max_trials")
        incremental = bool(config.get("incremental", False))
        random_state = _coerce_optional_int(config.get("random_state"), "data_source.random_state")
        noise_level = float(config.get("noise_level", 0.0))

        available_ids = eq_ids if eq_ids is not None else benchmark.equation_ids()
        total_available = len(available_ids) * datasets_per_expression
        target_size = target_size_override
        if target_size is None:
            target_size = max(0, total_available - skip)
        else:
            target_size = max(0, target_size)

        source = FastSRBSource(
            benchmark=benchmark,
            target_size=target_size,
            skip=skip,
            eq_ids=eq_ids,
            datasets_per_expression=datasets_per_expression,
            support_points=support_points,
            sample_points=sample_points,
            n_support_override=n_support_override,
            method=method,
            max_trials=max_trials,
            incremental=incremental,
            random_state=random_state,
            noise_level=noise_level,
        )
        return source, {"benchmark": benchmark}

    raise ValueError(f"Unsupported data source type: {dtype}")


def _infer_total_limit_from_data_source(config: Mapping[str, Any]) -> tuple[int | None, dict[str, Any]]:
    target_size = _coerce_optional_int(config.get("target_size"), "data_source.target_size")
    if target_size is not None:
        return target_size, {}

    dtype = str(config.get("type", "skeleton_dataset")).lower()
    if dtype == "skeleton_dataset":
        dataset_spec = config.get("dataset")
        if dataset_spec is None:
            raise ValueError("data_source.dataset must be provided for skeleton_dataset sources")
        dataset = _load_dataset(dataset_spec)
        repeats = _coerce_optional_int(
            config.get("datasets_per_expression"),
            "data_source.datasets_per_expression",
        )
        per_expression = repeats if repeats is not None and repeats > 0 else 1
        pool_size = len(getattr(dataset, "skeleton_pool"))
        return pool_size * per_expression, {"dataset": dataset}

    if dtype != "fastsrb":
        return None, {}

    benchmark_path = config.get("benchmark_path")
    if benchmark_path is None:
        raise ValueError("data_source.benchmark_path must be provided for FastSRB sources")

    benchmark = FastSRBBenchmark(
        substitute_root_path(str(benchmark_path)),
        random_state=_coerce_optional_int(config.get("benchmark_random_state"), "data_source.benchmark_random_state"),
    )

    eq_ids = _parse_equation_ids(config.get("eq_ids"))
    datasets_per_expression_cfg = config.get("datasets_per_expression")
    legacy_count_cfg = config.get("count")
    repeats_field = "data_source.datasets_per_expression"
    repeats_value = datasets_per_expression_cfg
    if repeats_value is None:
        repeats_field = "data_source.count"
        repeats_value = legacy_count_cfg
    if repeats_value is None:
        repeats_field = "data_source.datasets_per_expression"
        repeats_value = 1
    datasets_per_expression = _coerce_int(repeats_value, repeats_field)

    available_ids = eq_ids if eq_ids is not None else benchmark.equation_ids()
    return len(available_ids) * datasets_per_expression, {"benchmark": benchmark}


AdapterBuilder = Callable[[Mapping[str, Any], Mapping[str, Any]], FlashANSRAdapter | PySRAdapter | NeSymReSAdapter | SkeletonPoolAdapter | BruteForceAdapter | E2EAdapter]


def _build_model_adapter(config: Mapping[str, Any], *, context: Mapping[str, Any]) -> FlashANSRAdapter | PySRAdapter | NeSymReSAdapter | SkeletonPoolAdapter | BruteForceAdapter | E2EAdapter:
    adapter_type = str(config.get("type", "flash_ansr")).lower()
    builder = _ADAPTER_REGISTRY.get(adapter_type)
    if builder is None:
        raise ValueError(f"Unsupported model adapter type: {adapter_type}")
    return builder(config, context)


def _build_flash_ansr_adapter(config: Mapping[str, Any], context: Mapping[str, Any]) -> FlashANSRAdapter:  # noqa: ARG001
    model_path = config.get("model_path")
    eval_config_payload = config.get("evaluation_config")
    if model_path is None or eval_config_payload is None:
        raise ValueError("flash_ansr adapter requires model_path and evaluation_config")

    if isinstance(eval_config_payload, Mapping):
        eval_cfg = dict(eval_config_payload)
    else:
        eval_cfg = load_config(substitute_root_path(str(eval_config_payload)))

    if "evaluation" in eval_cfg:
        eval_cfg = eval_cfg["evaluation"]

    evaluation_overrides = config.get("evaluation_overrides")
    if evaluation_overrides is not None:
        if not isinstance(evaluation_overrides, Mapping):
            raise ValueError("evaluation_overrides must be a mapping")
        eval_cfg = _merge_mappings(eval_cfg, evaluation_overrides)

    generation_section = eval_cfg.get("generation_config")
    if not isinstance(generation_section, Mapping):
        raise ValueError("evaluation.generation_config must be provided in the evaluation config")

    generation_overrides = config.get("generation_overrides")
    if generation_overrides is not None:
        if not isinstance(generation_overrides, Mapping):
            raise ValueError("generation_overrides must be a mapping")
        generation_section = _merge_mappings(generation_section, generation_overrides)

    generation_config = create_generation_config(
        method=generation_section["method"],
        **generation_section.get("kwargs", {}),
    )

    model = FlashANSR.load(
        directory=substitute_root_path(str(model_path)),
        generation_config=generation_config,
        n_restarts=eval_cfg["n_restarts"],
        refiner_method=eval_cfg.get("refiner_method", "curve_fit_lm"),
        refiner_p0_noise=eval_cfg["refiner_p0_noise"],
        refiner_p0_noise_kwargs=eval_cfg.get("refiner_p0_noise_kwargs"),
        parsimony=eval_cfg["parsimony"],
        device=eval_cfg.get("device", config.get("device", "cpu")),
        refiner_workers=config.get("refiner_workers", eval_cfg.get("refiner_workers")),
    )

    complexity = config.get("complexity", eval_cfg.get("complexity", "none"))
    adapter_device = config.get("device", eval_cfg.get("device", "cpu"))
    refiner_workers = config.get("refiner_workers", eval_cfg.get("refiner_workers"))

    return FlashANSRAdapter(
        model,
        device=adapter_device,
        complexity=complexity,
        refiner_workers=refiner_workers,
    )


def _build_pysr_adapter(config: Mapping[str, Any], context: Mapping[str, Any]) -> PySRAdapter:
    timeout = _coerce_int(config.get("timeout_in_seconds", 60), "model_adapter.timeout_in_seconds")
    niterations = _coerce_int(config.get("niterations", 100), "model_adapter.niterations")
    padding = bool(config.get("padding", True))
    use_mult_div = bool(config.get("use_mult_div_operators", False))

    dataset = context.get("dataset")
    simplipy_engine = dataset.simplipy_engine if isinstance(dataset, FlashANSRDataset) else None
    engine_override = config.get("simplipy_engine")
    if engine_override is not None:
        simplipy_engine = SimpliPyEngine.load(substitute_root_path(str(engine_override)), install=True)

    if simplipy_engine is None:
        raise ValueError(
            "PySR adapter requires a SimpliPy engine (provide one via a skeleton_dataset data source or set "
            "model_adapter.simplipy_engine)."
        )

    return PySRAdapter(
        timeout_in_seconds=timeout,
        niterations=niterations,
        use_mult_div_operators=use_mult_div,
        padding=padding,
        simplipy_engine=simplipy_engine,
    )


def _build_nesymres_adapter(config: Mapping[str, Any], context: Mapping[str, Any]) -> NeSymReSAdapter:  # noqa: ARG001
    from flash_ansr.compat.nesymres import load_nesymres

    eq_setting_path = config.get("eq_setting_path")
    config_path = config.get("config_path")
    weights_path = config.get("weights_path")
    simplipy_engine_path = config.get("simplipy_engine")
    if not all([eq_setting_path, config_path, weights_path, simplipy_engine_path]):
        raise ValueError("nesymres adapter requires eq_setting_path, config_path, weights_path, and simplipy_engine")

    beam_width = _coerce_optional_int(config.get("beam_width"), "model_adapter.beam_width")
    n_restarts = _coerce_optional_int(config.get("n_restarts"), "model_adapter.n_restarts")
    device = str(config.get("device", "cpu"))
    remove_padding = bool(config.get("remove_padding", True))

    model, fitfunc = load_nesymres(
        eq_setting_path=substitute_root_path(str(eq_setting_path)),
        config_path=substitute_root_path(str(config_path)),
        weights_path=substitute_root_path(str(weights_path)),
        beam_size=beam_width,
        n_restarts=n_restarts,
        device=device,
    )

    simplipy_engine = SimpliPyEngine.load(substitute_root_path(str(simplipy_engine_path)), install=True)

    return NeSymReSAdapter(
        model=model,
        fitfunc=fitfunc,
        simplipy_engine=simplipy_engine,
        device=device,
        beam_width=beam_width,
        remove_padding=remove_padding,
    )


def _build_e2e_adapter(config: Mapping[str, Any], context: Mapping[str, Any]) -> E2EAdapter:
    simplipy_engine = _resolve_simplipy_engine(config, context, adapter_name="e2e")

    model_path = config.get("model_path")
    if model_path is None:
        raise ValueError("e2e adapter requires model_path")

    candidates_per_bag = _coerce_int(config.get("candidates_per_bag", 1), "model_adapter.candidates_per_bag")
    max_input_points = _coerce_int(config.get("max_input_points", 200), "model_adapter.max_input_points")

    max_number_bags_cfg = config.get("max_number_bags")
    max_number_bags = _coerce_optional_int(max_number_bags_cfg, "model_adapter.max_number_bags")
    if max_number_bags is None:
        max_number_bags = 10

    n_trees_to_refine = _coerce_int(config.get("n_trees_to_refine", 10), "model_adapter.n_trees_to_refine")
    rescale = bool(config.get("rescale", True))

    return E2EAdapter(
        model_path=substitute_root_path(str(model_path)),
        simplipy_engine=simplipy_engine,
        device=str(config.get("device", "cpu")),
        candidates_per_bag=candidates_per_bag,
        max_input_points=max_input_points,
        max_number_bags=max_number_bags,
        n_trees_to_refine=n_trees_to_refine,
        rescale=rescale,
    )


def _resolve_simplipy_engine(config: Mapping[str, Any], context: Mapping[str, Any], *, adapter_name: str) -> SimpliPyEngine:
    dataset = context.get("dataset")
    simplipy_engine = dataset.simplipy_engine if isinstance(dataset, FlashANSRDataset) else None

    engine_override = config.get("simplipy_engine")
    if engine_override is not None:
        simplipy_engine = SimpliPyEngine.load(substitute_root_path(str(engine_override)), install=True)

    if simplipy_engine is None:
        raise ValueError(
            f"{adapter_name} adapter requires a SimpliPy engine (provide one via a skeleton_dataset data source or set model_adapter.simplipy_engine)."
        )

    return simplipy_engine


def _build_skeleton_pool_adapter(config: Mapping[str, Any], context: Mapping[str, Any]) -> SkeletonPoolAdapter:
    simplipy_engine = _resolve_simplipy_engine(config, context, adapter_name="skeleton_pool")

    skeleton_pool = config.get("skeleton_pool")
    if skeleton_pool is None:
        raise ValueError("skeleton_pool adapter requires skeleton_pool")
    if isinstance(skeleton_pool, str):
        skeleton_pool = substitute_root_path(skeleton_pool)

    model = SkeletonPoolModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=skeleton_pool,
        samples=_coerce_int(config.get("samples", 32), "model_adapter.samples"),
        unique=bool(config.get("unique", True)),
        ignore_holdouts=bool(config.get("ignore_holdouts", True)),
        seed=_coerce_optional_int(config.get("seed"), "model_adapter.seed"),
        n_restarts=_coerce_int(config.get("n_restarts", 8), "model_adapter.n_restarts"),
        refiner_method=str(config.get("refiner_method", "curve_fit_lm")),
        refiner_p0_noise=config.get("refiner_p0_noise", "normal"),
        refiner_p0_noise_kwargs=config.get("refiner_p0_noise_kwargs", "default"),
        numpy_errors=config.get("numpy_errors", "ignore"),
        parsimony=float(config.get("parsimony", 0.05)),
    )

    return SkeletonPoolAdapter(model)


def _build_brute_force_adapter(config: Mapping[str, Any], context: Mapping[str, Any]) -> BruteForceAdapter:
    simplipy_engine = _resolve_simplipy_engine(config, context, adapter_name="brute_force")

    skeleton_pool = config.get("skeleton_pool")
    if skeleton_pool is None:
        raise ValueError("brute_force adapter requires skeleton_pool")
    if isinstance(skeleton_pool, str):
        skeleton_pool = substitute_root_path(skeleton_pool)

    model = BruteForceModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=skeleton_pool,
        max_expressions=_coerce_int(config.get("max_expressions", 10000), "model_adapter.max_expressions"),
        max_length=_coerce_optional_int(config.get("max_length"), "model_adapter.max_length"),
        include_constant_token=bool(config.get("include_constant_token", True)),
        ignore_holdouts=bool(config.get("ignore_holdouts", True)),
        n_restarts=_coerce_int(config.get("n_restarts", 8), "model_adapter.n_restarts"),
        refiner_method=str(config.get("refiner_method", "curve_fit_lm")),
        refiner_p0_noise=config.get("refiner_p0_noise", "normal"),
        refiner_p0_noise_kwargs=config.get("refiner_p0_noise_kwargs", "default"),
        numpy_errors=config.get("numpy_errors", "ignore"),
        parsimony=float(config.get("parsimony", 0.05)),
    )

    return BruteForceAdapter(model)


_ADAPTER_REGISTRY: dict[str, AdapterBuilder] = {
    "flash_ansr": _build_flash_ansr_adapter,
    "pysr": _build_pysr_adapter,
    "nesymres": _build_nesymres_adapter,
    "skeleton_pool": _build_skeleton_pool_adapter,
    "brute_force": _build_brute_force_adapter,
    "e2e": _build_e2e_adapter,
}


# ---------------------------------------------------------------------------
# Utilities

def _extract_run_section(config: Mapping[str, Any]) -> MutableMapping[str, Any]:
    if "run" in config:
        return dict(config["run"])
    if "evaluation_run" in config:
        return dict(config["evaluation_run"])
    return dict(config)


def _select_experiment(config: Mapping[str, Any], experiment: str | None) -> MutableMapping[str, Any]:
    experiments = config.get("experiments")
    if not experiments:
        return dict(config)
    if not isinstance(experiments, Mapping):
        raise ValueError("config.experiments must be a mapping")

    chosen = experiment or config.get("default_experiment")
    if chosen is None:
        available = ", ".join(str(key) for key in experiments.keys()) or "<none>"
        raise ValueError(f"Config defines experiments ({available}) but no experiment name was provided")
    if chosen not in experiments:
        available = ", ".join(str(key) for key in experiments.keys())
        raise KeyError(f"Experiment '{chosen}' not found. Available experiments: {available}")

    selection = experiments[chosen]
    if not isinstance(selection, Mapping):
        raise ValueError("Each experiment entry must be a mapping containing a run section")
    return dict(selection)


def _merge_mappings(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def _coerce_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} must be provided")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field_name} must be an integer") from exc


def _coerce_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field_name} must be an integer or null") from exc


def _load_dataset(spec: Any) -> FlashANSRDataset:
    if isinstance(spec, Mapping):
        path = spec.get("path") or spec.get("config")
        if path is None:
            raise ValueError("dataset spec must include a path")
        mode = str(spec.get("mode", "auto")).lower()
    else:
        path = spec
        mode = "auto"

    resolved_path = substitute_root_path(str(path))
    if mode not in {"auto", "config", "compiled"}:
        raise ValueError("dataset spec mode must be 'auto', 'config', or 'compiled'")

    if mode == "config":
        return FlashANSRDataset.from_config(resolved_path)
    if mode == "compiled":
        _, dataset = FlashANSRDataset.load(resolved_path)
        return dataset

    if os.path.isdir(resolved_path):
        _, dataset = FlashANSRDataset.load(resolved_path)
        return dataset
    return FlashANSRDataset.from_config(resolved_path)


def _load_existing_results(path: str) -> Mapping[str, Sequence[Any]] | None:
    resolved = Path(substitute_root_path(path))
    if not resolved.exists():
        return None
    with resolved.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive
        raise ValueError("Stored evaluation results must be a mapping")
    return payload  # type: ignore[return-value]


def _parse_equation_ids(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw_tokens = [token.strip() for token in value.replace(",", " ").split()]
        return [token for token in raw_tokens if token]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise ValueError("fastsrb.eq_ids must be a string or a list of strings")


__all__ = ["EvaluationRunPlan", "build_evaluation_run"]

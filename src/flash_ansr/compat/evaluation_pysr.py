from typing import Any

from flash_ansr import FlashANSRDataset
from flash_ansr.eval.data_sources import SkeletonDatasetSource
from flash_ansr.eval.engine import EvaluationEngine
from flash_ansr.eval.model_adapters import PySRAdapter
from flash_ansr.eval.result_store import ResultStore
from flash_ansr.utils.config_io import load_config


class PySREvaluation():
    def __init__(
            self,
            n_support: int | None = None,
            noise_level: float = 0.0,
            timeout_in_seconds: int = 60,
            niterations: int = 100,
            padding: bool = True,
            use_mult_div_operators: bool = False) -> None:

        self.n_support = n_support
        self.noise_level = noise_level
        self.timeout_in_seconds = timeout_in_seconds
        self.niterations = niterations
        self.padding = padding
        self.use_mult_div_operators = use_mult_div_operators

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "PySREvaluation":
        config_ = load_config(config)

        if "evaluation" in config_.keys():
            config_ = config_["evaluation"]

        return cls(
            n_support=config_["n_support"],
            noise_level=config_.get("noise_level", 0.0),
            timeout_in_seconds=config_["timeout_in_seconds"],
            niterations=config_["niterations"],
            padding=config_['padding'],
            use_mult_div_operators=config_['use_mult_div_operators'],
        )

    def evaluate(
            self,
            dataset: FlashANSRDataset,
            results_dict: dict[str, Any] | None = None,
            size: int | None = None,
            save_every: int | None = None,
            output_file: str | None = None,
            verbose: bool = True) -> dict[str, Any]:
        if save_every is not None and output_file is None:
            raise ValueError('output_file must be provided when save_every is set.')

        store = ResultStore(results_dict)
        existing = store.size

        target_total = len(dataset.skeleton_pool) if size is None else size
        if target_total <= existing:
            return dict(sorted(store.snapshot().items()))

        remaining = target_total - existing

        data_source = SkeletonDatasetSource(
            dataset,
            target_size=remaining,
            n_support=self.n_support,
            noise_level=self.noise_level,
            preprocess=False,
            device='cpu',
        )

        adapter = PySRAdapter(
            timeout_in_seconds=self.timeout_in_seconds,
            niterations=self.niterations,
            use_mult_div_operators=self.use_mult_div_operators,
            padding=self.padding,
            simplipy_engine=dataset.simplipy_engine,
        )

        engine = EvaluationEngine(
            data_source=data_source,
            model_adapter=adapter,
            result_store=store,
        )

        results = engine.run(
            limit=remaining,
            save_every=save_every,
            output_path=output_file,
            verbose=verbose,
            progress=verbose,
        )

        return dict(sorted(results.items()))

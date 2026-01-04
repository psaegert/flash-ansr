import warnings
from typing import Any

from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.data import FlashANSRDataset
from flash_ansr.eval.data_sources import SkeletonDatasetSource
from flash_ansr.eval.engine import EvaluationEngine
from flash_ansr.eval.model_adapters import FlashANSRAdapter
from flash_ansr.eval.result_store import ResultStore
from flash_ansr.utils.config_io import load_config


class Evaluation():
    '''
    Evaluate a Flash-ANSR model on a dataset.

    Parameters
    ----------
    n_support : int, optional
        Number of input points for each equation. Default is None (sampled from the dataset).
    noise_level : float, optional
        Noise level for the constant fitting in units of standard deviations of the target variable. Default is 0.0.
    complexity : str, optional
        Complexity constraint for the generated equations. Can be 'none' or 'ground_truth'. Default is 'none'.
    preprocess : bool, optional
        Whether to preprocess the data using FlashASNRPreprocessor. Default is False.
    device : str, optional
        Device to run the evaluation on. Default is 'cpu'.
    refiner_workers : int or None, optional
        Number of worker processes to use during constant refinement. ``None``
        relies on the model default (all available CPU cores).
    '''
    def __init__(
            self,
            n_support: int | None = None,
            noise_level: float = 0.0,
            complexity: str | list[int | float] = 'none',
            preprocess: bool = False,
            device: str = 'cpu',
            refiner_workers: int | None = None) -> None:

        self.n_support = n_support
        self.noise_level = noise_level
        self.complexity = complexity
        self.preprocess = preprocess

        self.device = device
        self.refiner_workers = refiner_workers

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "Evaluation":
        '''
        Create an Evaluation object from a configuration dictionary or a configuration file.

        Parameters
        ----------
        config : dict or str
            Configuration dictionary or path to the configuration file.

        Returns
        -------
        Evaluation
            The Evaluation object.
        '''
        config_ = load_config(config)

        if "evaluation" in config_.keys():
            config_ = config_["evaluation"]

        return cls(
            n_support=config_["n_support"],
            noise_level=config_.get("noise_level", 0.0),
            complexity=config_.get("complexity", 'none'),
            preprocess=config_.get("preprocess", False),
            device=config_["device"],
            refiner_workers=config_.get("refiner_workers", None)
        )

    def evaluate(
            self,
            model: FlashANSR,
            dataset: FlashANSRDataset,
            results_dict: dict[str, Any] | None = None,
            size: int | None = None,
            save_every: int | None = None,
            output_file: str | None = None,
            verbose: bool = True) -> dict[str, Any]:
        '''
        Evaluate the model on the dataset.

        Parameters
        ----------
        model : FlashANSR
            The model to evaluate.
        size : int, optional
            Number of samples to evaluate. Default is None.
        verbose : bool, optional
            Whether to print the progress. Default is True.

        Returns
        -------
        dict
            Dictionary with the evaluation results.
        '''
        if verbose:
            print(
                "Evaluating model with configuration: "
                f"model.parsimony={model.parsimony}, noise_level={self.noise_level}, "
                f"n_support={self.n_support}, complexity={self.complexity}"
            )

        store = ResultStore(results_dict)
        existing = store.size

        target_total = len(dataset.skeleton_pool) if size is None else size
        if target_total <= existing:
            if target_total < existing:
                warnings.warn(
                    "Requested evaluation size is smaller than the number of existing results. "
                    "Returning without additional evaluation."
                )
            return dict(sorted(store.snapshot().items()))

        remaining = target_total - existing

        data_source = SkeletonDatasetSource(
            dataset,
            target_size=remaining,
            n_support=self.n_support,
            noise_level=self.noise_level,
            preprocess=self.preprocess,
            device=self.device,
        )

        adapter = FlashANSRAdapter(
            model,
            device=self.device,
            complexity=self.complexity,
            refiner_workers=self.refiner_workers,
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

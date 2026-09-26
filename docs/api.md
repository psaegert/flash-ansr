# API Reference

## FlashANSR
::: flash_ansr.FlashANSR
    options:
      heading_level: 3
      members:
        - load
        - fit
        - generate
        - predict
        - get_expression
        - results
        - to
        - close
      members_order: source
      filters:
        - "!^_"
      show_root_toc_entry: false

## The result
The object [`FlashANSR.fit`](#flashansr) returns and keeps as `model.result_`: the score-sorted refined candidates, the full classified candidate ledger, the ranking that ordered them and the phase times. Plain data: it pickles, re-ranks offline (`rerank`) and evaluates through the engine it was bound to.

### FitResult
::: flash_ansr.inference.FitResult
    options:
      heading_level: 4
      show_root_toc_entry: false

### Candidate
::: flash_ansr.inference.Candidate
    options:
      heading_level: 4
      show_root_toc_entry: false

### CandidateLedger
::: flash_ansr.inference.CandidateLedger
    options:
      heading_level: 4
      show_root_toc_entry: false

### Generation
::: flash_ansr.flash_ansr.Generation
    options:
      heading_level: 4
      show_root_toc_entry: false

## The estimator's policy
The four config objects `FlashANSR.load` takes: the sampler, the refiner, the ranking and the compute.

### SoftmaxSamplingConfig
::: flash_ansr.utils.generation.SoftmaxSamplingConfig
    options:
      heading_level: 4
      show_root_toc_entry: false

### PriorSamplingConfig
::: flash_ansr.utils.generation.PriorSamplingConfig
    options:
      heading_level: 4
      show_root_toc_entry: false

### OracleConfig
::: flash_ansr.utils.generation.OracleConfig
    options:
      heading_level: 4
      show_root_toc_entry: false

### RefineConfig
::: flash_ansr.estimator_config.RefineConfig
    options:
      heading_level: 4
      show_root_toc_entry: false

### RankingConfig
::: flash_ansr.scoring.RankingConfig
    options:
      heading_level: 4
      show_root_toc_entry: false

### ComputeConfig
::: flash_ansr.estimator_config.ComputeConfig
    options:
      heading_level: 4
      show_root_toc_entry: false

## FlashANSRDataset
::: flash_ansr.data.FlashANSRDataset
    options:
      heading_level: 3
      members:
        - from_config
        - iterate
        - compile
        - save
        - shutdown
      members_order: source
      filters:
        - "!^_"
      show_root_toc_entry: false

## FlashANSRPreprocessor
::: flash_ansr.preprocessing.FlashANSRPreprocessor
    options:
      heading_level: 3
      show_root_toc_entry: false

## Generation configurations
### SoftmaxSamplingConfig
::: flash_ansr.utils.generation.SoftmaxSamplingConfig
    options:
      heading_level: 3
      show_root_toc_entry: false

### PriorSamplingConfig

::: flash_ansr.PriorSamplingConfig

### PriorSampler

::: flash_ansr.prior.PriorSampler

## Utilities
::: flash_ansr.get_path
    options:
      heading_level: 3
      show_root_toc_entry: false

::: flash_ansr.load_config
    options:
      heading_level: 3
      show_root_toc_entry: false

## HybridRegressor
Flash-ANSR seeding PySR ([Flash-ANSR + PySR](hybrid.md)); install with `pip install flash-ansr[pysr]`.

::: flash_ansr.hybrid.HybridRegressor
    options:
      heading_level: 3
      members:
        - fit
        - predict
        - get_expression
        - solve
      members_order: source
      show_root_toc_entry: false

### HybridConfig
::: flash_ansr.hybrid.HybridConfig
    options:
      heading_level: 4
      show_root_toc_entry: false

### HybridFitResult
::: flash_ansr.hybrid.HybridFitResult
    options:
      heading_level: 4
      show_root_toc_entry: false

### HybridCandidate
::: flash_ansr.hybrid.HybridCandidate
    options:
      heading_level: 4
      show_root_toc_entry: false

# API Reference

## FlashANSR
::: flash_ansr.FlashANSR
    options:
      heading_level: 3
      members:
        - load
        - fit
        - infer
        - predict
        - get_expression
        - save_results
        - load_results
        - compile_results
      members_order: source
      filters:
        - "!^_"
      show_root_toc_entry: false

## Inference results
The objects returned by [`FlashANSR.infer`](#flashansr): the score-sorted refined candidates plus the full classified candidate ledger.

`InferenceResult.to_dataframe()` returns the refined survivors (the `FIT_OK` candidates in `result.candidates`) as a pandas DataFrame, one row per candidate; it does not include the full ledger.

### InferenceResult
::: flash_ansr.inference.InferenceResult
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
### BeamSearchConfig
::: flash_ansr.utils.generation.BeamSearchConfig
    options:
      heading_level: 3
      show_root_toc_entry: false

### SoftmaxSamplingConfig
::: flash_ansr.utils.generation.SoftmaxSamplingConfig
    options:
      heading_level: 3
      show_root_toc_entry: false

### MCTSGenerationConfig
::: flash_ansr.utils.generation.MCTSGenerationConfig
    options:
      heading_level: 3
      show_root_toc_entry: false

## Utilities
::: flash_ansr.get_path
    options:
      heading_level: 3
      show_root_toc_entry: false

::: flash_ansr.load_config
    options:
      heading_level: 3
      show_root_toc_entry: false
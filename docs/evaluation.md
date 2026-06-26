# Evaluation

In flash-ansr 0.6, evaluation, comparison baselines, and systematic benchmarking moved out of this repository into the standalone companion package **srbf** (Symbolic Regression Benchmark Framework).

## Get srbf

```sh
pip install srbf
```

Repository: [https://github.com/psaegert/srbf](https://github.com/psaegert/srbf). See srbf's README for usage (running evaluations, configuring baselines, and external-model setup).

## Evaluation configs

flash-ansr still ships the evaluation configs under `configs/evaluation/` as data. These are consumed by srbf; they are no longer driven by an in-repo command.

## What stays in flash-ansr

- Model loading plus `fit` / `predict` via the `FlashANSR` API, see [Getting Started](getting_started.md).
- The `flash_ansr benchmark` CLI for dataset iteration timing.

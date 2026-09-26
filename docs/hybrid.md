# Flash-ANSR + PySR

The hybrid combines Flash-ANSR with [PySR](https://github.com/MilesCranmer/PySR), an evolutionary search. Flash-ANSR
draws candidate formulas from the data and fits their constants; its best candidates become the starting population
of PySR's search; PySR's hall of fame (its best formula of every length) is then priced the way Flash-ANSR prices its
own candidates, and Flash-ANSR's ranking picks the answer from the combined pool.

## Install

```sh
pip install flash-ansr[pysr]
```

PySR runs its search in Julia, which it downloads and compiles on first use (a few minutes, once).

## Use

```python
from flash_ansr import FlashANSR
from flash_ansr.hybrid import HybridRegressor

model = FlashANSR.load("psaegert/flash-ansr-v25.0-T8-20M")   # any Flash-ANSR model
hybrid = HybridRegressor(model)

result = hybrid.fit(X, y)          # 1024 draws, then 64 PySR iterations
result.get_expression()            # the answer, infix
result.best.source                 # "flash-ansr" or "pysr": where the answer came from
hybrid.predict(X_new)              # evaluate the answer
result.to_dataframe()              # the whole ranked pool
```

`fit` returns a `HybridFitResult`: a `FitResult` whose candidates are the combined pool in Flash-ANSR's ranking order,
each tagged with its `source`, plus what the PySR stage did (`niterations`, `n_seeds`, `pysr_time`,
`pysr_equations`, `pysr_error`). `result.rerank(...)`, `save` / `HybridFitResult.load` work as for `FitResult`.

## Budget: draws and iterations

The hybrid is defined by its work: Flash-ANSR draws `draws` candidates, PySR runs `niterations` iterations. The
default pairs them so that the two stages take the same time (r* = 0.5, the split at which the combination kept most
of both parts' recoveries), measured with Flash-ANSR T8-20M on one workstation:

| draws | PySR iterations |
|------:|----------------:|
| 512 | 16 |
| 1024 | 64 |
| 2048 | 256 |
| 4096 | 512 |
| 8192 | 1024 |

`hybrid.fit(X, y, draws=4096)` takes the paired 512 iterations; `fit(..., niterations=...)` sets them yourself, and
`niterations=0` is Flash-ANSR alone. A draw count outside the table takes the pair of the nearest row.

## Configuration

`HybridRegressor(model, HybridConfig(...))` sets the method's knobs: `k_seeds` (how many Flash-ANSR candidates seed
PySR, 100), `pysr` (a `PySRSettings`: `maxsize`, `parsimony`, `warmup`, further `PySRRegressor` arguments) and
`pysr_timeout_cap_s` (a safety cap on PySR's wall time). PySR uses Flash-ANSR's 23-operator vocabulary.

A second mode runs by the clock instead of by work: `HybridConfig(budget_s=T, ratio=r)` gives Flash-ANSR the wall time
(1 - r) T and PySR the rest. It serves evaluation sweeps through `HybridRegressor.solve`, the path a benchmark harness
drives (it returns a record rather than a `FitResult`, and can cache one generation pass per problem across every
budget of a ladder or every cell of a sweep).

## Reproducibility

Flash-ANSR's stage is reproducible with `fit(..., seed=)`. PySR's search runs on several threads and is not
deterministic run to run, so the hybrid's answer can differ between runs at the same seed.

# Inference API for the v24 task family — design proposal

**Status: DRAFT 2026-08-26, for owner review.** Nothing here is implemented beyond
what the capability probes prototyped; the probes live in the session scratch and
double as working reference implementations for every verb below.

## Why

T16 is trained on a task FAMILY — expression emission (with or without spelled
constants, under emission-format flags), constant infilling, complexity
conditioning and hypothesis, held-out-point prediction, per-point outlier
classification — but the deployed surface exposes exactly one verb, `fit()`, and
every other capability is reachable only by monkeypatching prompt internals or
driving `FlashANSRModel` by hand. The 2026-08-26 capability probes needed:

* an instance-level `_prepare_prompt_prefix` monkeypatch to set `<mask_all>`;
* hand-built training-grammar prompts + a hand-rolled forced-opener greedy loop
  for `<predict_constants>`;
* `Refiner._initialize_expression` + `expression_lambda` privates to evaluate an
  expression at given constants;
* `model(...)` + `model.outlier_head(model.point_representations)` internals for
  outlier scores;
* manual `v1..vn -> x1..xN` variable renaming at the prompt boundary.

Each bullet is a missing public verb. The same probes also surfaced a harness
bug (fixed, 514810f): post-processing judged validity on raw tokens and silently
discarded every constant-bearing sample — the lesson is that v23 assumptions
live in several layers, and a designed surface is how they stop leaking.

## Principles

1. **One public verb per trained capability.** If the training data contains a
   task, the estimator exposes it; if it does not, the estimator refuses loudly.
2. **The harness owns the grammar at inference exactly as in training.** Openers
   and flags are force-fed and never sampled; the model owns content nibbles and
   closing tags. One shared grammar-aware decoder (the `constrain_ieee754`
   machinery generalized to the task blocks) replaces per-script loops.
3. **Dialects and variable names resolve at the boundary.** Every verb accepts
   explicit or tagged prefix tokens and user variable names; internally
   everything is the tagged canonical over `x1..xN`, and outputs map back.
4. **v23 checkpoints degrade gracefully.** Verbs whose tokens are absent from
   the vocabulary raise `CapabilityUnavailable` at call time, not mid-decode.

## Proposed surface

```python
ansr = FlashANSR.load("v24.0-T16")

# 1  Regression (existing, semantics unchanged)
ansr.fit(X, y, variable_names=..., complexity=...)   # complexity= already exists

# 2  Emission control (replaces the mask_all monkeypatch)
ansr.fit(X, y, emission="constants")   # default: constants spelled (ieee754 spans)
ansr.fit(X, y, emission="skeleton")    # <mask_all>: placeholders, refiner fits all
ansr.fit(X, y, emission="fittable")    # <mask_fittable>
ansr.fit(X, y, refine=False)           # as-emitted: verbatim constants, no optimizer;
                                       # candidate selection by support FVU
# Candidate grows: .constants_emitted (verbatim span values, None for v23 beams)
# next to the refined .constants.

# 3  Constant infilling (the <predict_constants> block)
res = ansr.predict_constants(X, y, expression)
# expression: prefix tokens (explicit or tagged) or infix str, any variable names,
# '<constant>' marks the slots. Returns values in slot order + per-slot nibble
# logits so callers can read confidence; grammar-forced decode inside.

# 4  Per-point outlier scores (the outlier head)
p = ansr.score_outliers(X, y)                      # raw sigmoid per point
p = ansr.score_outliers(X, y, operating_point="P@R50")   # thresholded via the
# measured calibration table stored with the checkpoint (AUROC 0.9888 eval).
# Documented caveat: the head reads the data-set encoder only -- scores condition
# on (X, y), never on any expression.

# 5  Held-out-point prediction (the <predict_y> block)
y_star = ansr.predict_y(X, y, x_star)

# 6  Complexity hypothesis (the <hypothesize> licence)
mu = ansr.hypothesize_complexity(X, y)

# 7  Expression evaluation (public form of the Refiner internals)
y_hat = ansr.evaluate(expression, constants, X)
```

## Defaults that change

* `constrain_ieee754=True` becomes the sampling default for vocabularies that
  carry the span tokens: the grammar guarantees well-formed spans, so the
  malformed-carrier disposal path becomes dead code instead of a silent filter.
* Dedup stays "by constantified skeleton" (the documented semantics); the
  carrier-preservation rule from 514810f is the implementation of it.

## Cost

`predict_constants`, `score_outliers`, `emission=`, `refine=False` are thin:
the machinery exists and the probes are working prototypes. `predict_y` /
`hypothesize_complexity` are the same shape as `predict_constants`. `evaluate`
is a refactor of `Refiner` internals. The boundary work (dialect + variable
normalization shared by all verbs) is the only genuinely new module.

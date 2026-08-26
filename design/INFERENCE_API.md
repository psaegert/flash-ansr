# Inference API for the v24 task family

**Status: owner-ratified surface, 2026-08-26.** Supersedes the 2026-08-26 draft, which
assumed `complexity=` already worked and keyed `score_outliers` on a calibration table
that does not exist. Ruled by the owner in the same sitting as the srbf/flash-ansr
division of labour; implementation follows this document verb by verb.

## Division of labour

**srbf owns evaluation** — config, running, storing, analysing, reporting — for the
traditional SR task only: predict an expression, fit its constants, score the fit.
**flash-ansr owns the model and the harness.** srbf does not model what `<mask_all>`
*means*; it defines and serves a config block, and flash-ansr reads and interprets it.
The auxiliary verbs below are deliberately outside srbf's scope for now and are driven
by custom scripts.

This is why the surface is shaped as **one very capable, configurable fitting verb plus
one verb per auxiliary task** rather than a flat bag of options: `fit` is what a
benchmark drives from a served config, and the rest are what a researcher calls directly.

## Why the surface exists

T16 is trained on a task FAMILY — expression emission (with or without spelled constants,
under emission-format flags), constant infilling, complexity conditioning and hypothesis,
held-out-point prediction, per-point outlier classification — while the deployed surface
exposes one verb. Every other capability is reachable only by monkeypatching prompt
internals or driving `FlashANSRModel` by hand. The 2026-08-26 capability probes needed:

* an instance-level `_prepare_prompt_prefix` monkeypatch to set `<mask_all>`;
* hand-built training-grammar prompts + a hand-rolled forced-opener greedy loop for
  `<predict_constants>`;
* `Refiner._initialize_expression` + `expression_lambda` privates to evaluate an
  expression at given constants;
* `model(...)` + `model.outlier_head(model.point_representations)` internals for outlier
  scores;
* manual `v1..vn -> x1..xN` variable renaming at the prompt boundary.

Each bullet is a missing public verb. Those same probes then produced numbers that a
70-agent audit had to correct, because a benchmark that must monkeypatch the model is a
benchmark nobody can check.

## Principles

1. **One public verb per trained capability.** If the training data contains a task, the
   estimator exposes it; if it does not, the estimator refuses loudly.
2. **The harness owns the grammar at inference exactly as in training.** Openers and flags
   are force-fed and never sampled; the model owns content nibbles and closing tags. One
   shared grammar-aware decoder (the `constrain_ieee754` machinery generalized to the task
   blocks) replaces per-script loops.
3. **Dialects and variable names resolve at the boundary.** Every verb accepts explicit or
   tagged prefix tokens and user variable names; internally everything is the tagged
   canonical over `x1..xN`, and outputs map back.
4. **v23 checkpoints degrade gracefully.** Verbs whose tokens are absent from the
   vocabulary raise `CapabilityUnavailable` at CALL time, not mid-decode — and the check
   runs before the encoder, not after it.
5. **Nothing on the public surface is decorative.** A parameter that is documented as
   having an effect must have one. `allowed_terms` / `include_terms` / `exclude_terms` are
   withdrawn under this principle rather than implemented (below).

## The fitting verb

```python
ansr = FlashANSR.load("v24.0-T16")

ansr.fit(
    X, y,
    variable_names="auto",
    emission="constants",     # "constants" | "skeleton" | "fittable"
    compaction=False,         # in-decode ieee754 compaction (beam search only)
    refine=True,              # False -> as-emitted verbatim constants, no optimizer
    hypothesize=False,        # True -> the model predicts its own complexity target
    complexity=None,          # simplipy mu; see "Fixing complexity" below
)
```

`emission` replaces the `_prepare_prompt_prefix` monkeypatch. `"constants"` (default,
unflagged) spells constants as ieee754 spans; `"skeleton"` sends `<mask_all>` and the
refiner fits every slot; `"fittable"` sends `<mask_fittable>`.

`compaction` is optional and **defaults off**. It is beam-search only — in-decode
compaction is not implemented for sampling, and advertising it there would be a lie. It
costs a measured 2.58× today and, on the current checkpoint, has not been shown to buy
recoveries; it stays configurable because that can change with a better checkpoint.

`refine=False` returns the model's verbatim emitted constants with no optimizer pass, and
selects candidates by support FVU.

`Candidate` gains `.constants_emitted` — the verbatim span values, `None` for v23 beams —
alongside the refined `.constants`. The model's own constant prediction is currently
decoded, used as an optimizer seed, and then discarded; this is the field that stops
throwing it away.

## Auxiliary verbs

Each is a separate public method, out of srbf's scope for now.

```python
# Per-point outlier scores. Full support -- this is a planned feature.
p = ansr.score_outliers(X, y)

# Held-out-point prediction (the <predict_y> block).
y_star = ansr.predict_y(X, y, x_star)

# Complexity prediction on its own, in ADDITION to fit(hypothesize=True).
mu = ansr.predict_complexity(X, y)

# Constant infilling (the <predict_constants> block).
res = ansr.predict_constants(X, y, expression)
```

`score_outliers` returns **raw per-point probabilities and nothing else**. There is no
operating-point argument and no calibration table: no such table exists, and the head's
only quantitative claim (AUROC 0.9888) is a POOLED, in-distribution val number. The
deployment question — per-instance ranking of points within one problem — has never been
measured, and the measured per-problem behaviour is much weaker (median P ≈ 0.42 for a
single lone outlier). Two caveats belong in the docstring: the head reads the data-set
encoder only, so scores condition on `(X, y)` and never on any expression; and it degrades
above roughly 10% contamination, the ceiling it was trained under.

`predict_constants` accepts prefix tokens (explicit or tagged) or an infix string, any
variable names, with `'<constant>'` marking the slots. It returns values in slot order
plus per-slot nibble logits so callers can read confidence. Grammar-forced decode inside.

## Fixing `complexity=`

`complexity` is load-bearing in two places now — a `fit` option and its own verb — so the
four defects against it are blocking, not cosmetic:

* it emits a v23 `<prompt>` / `</prompt>` block carrying **zero training signal** in this
  checkpoint (measured: ~32% weaker μ conditioning than the trained block);
* it writes the caller's raw number onto the numeric channel where training wrote simplipy
  **μ**, which runs 6,000–1,000,000+ — a ~4-orders-of-magnitude unit mismatch;
* the `complexity` a result REPORTS is a token count, so feeding a result back into a
  prompt is measurably worse than passing nothing;
* the unit is documented nowhere on the public API.

The fix is all four together: emit the trained `<complexity>` block, on the μ scale,
document the unit at every entry point, and report μ on results so the value round-trips.

## Withdrawn

`allowed_terms`, `include_terms` and `exclude_terms` are removed from the public surface.
They are documented as constraining generation, emit tokens this checkpoint never saw, and
have no decode-time enforcement whatsoever. Under principle 5 the choice is to implement or
withdraw, and the owner ruled withdraw. The grammar machinery that would implement them
(`constrain_ieee754`) stays, so they can return as real constraints if they earn it.

## Not in this document

Per-instance outlier AUROC/AUPRC, and the same metric on FastSRB with synthetic
contamination, must be measured before the outlier head is presented as a shipped
capability. Exposing the verb does not require them; claiming the head works does.

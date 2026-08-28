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
   estimator exposes it; if it does not, the estimator refuses loudly. This binds on
   CIRCUMSTANCES, not just blocks: a block trained in two placements needs both reachable, and a
   model trained with `condition_dropout` needs its unconditioned mode reachable. `conditioned=`
   is on every decoder verb and on `fit`/`infer`; `predict_y(expression=...)` selects the suffix
   placement. `score_outliers` takes no `conditioned=`: the head reads the encoder directly and
   has no null path to route through.
2. **The harness owns the grammar at inference exactly as in training.** Openers and flags
   are force-fed and never sampled; the model owns content nibbles and closing tags. One
   shared grammar-aware decoder (the `constrain_ieee754` machinery generalized to the task
   blocks) replaces per-script loops.
3. **Dialects and variable names resolve at the boundary.** Every verb accepts explicit or
   tagged prefix tokens and user variable names; internally everything is the tagged
   canonical over `x1..xN`, and outputs map back.
4. **v24 only, refused at load.** A checkpoint whose vocabulary lacks `<ieee754>` is refused by
   `FlashANSR.load` with the reason: one serialization, one span mapper, one numeric channel, no
   second path for anything to leak across. Within v24, a verb whose block is absent raises
   `CapabilityUnavailable` at CALL time, before the encoder runs — never mid-decode.
5. **Every predicted value is a DISTRIBUTION.** A value is eight hex nibbles and each nibble is a
   softmax draw, so one decode is one sample. The verbs draw `n_samples` (default 32) and return
   `ValueDistribution` — median, q05/q95, mode, agreement — never a float. Greedy decoding returns
   the mode of a factorised distribution, which is neither its centre nor its width.
6. **Nothing on the public surface is decorative.** A parameter that is documented as
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

`Candidate` gains `.constants_emitted` — the verbatim span values, `None` for a span-free beam —
alongside the refined `.constants`. The model's own constant prediction is currently
decoded, used as an optimizer seed, and then discarded; this is the field that stops
throwing it away.

## Auxiliary verbs

Each is a separate public method, out of srbf's scope for now.

```python
# Per-point outlier scores. NOT sampled: one deterministic forward through a sigmoid head.
p = ansr.score_outliers(X, y)                                   # -> np.ndarray

# The sampled verbs. n_samples / temperature / seed on each; all return distributions.
y_star = ansr.predict_y(X, y, x_star)                           # -> list[ValueDistribution]
mu = ansr.predict_complexity(X, y)                              # -> ComplexityDistribution
slots = ansr.predict_constants(X, y, expression)                # -> list[ValueDistribution]

# predict_y's SUFFIX placement -- the expression is in scope (trained, p_conditional=0.5):
y_star = ansr.predict_y(X, y, x_star, expression=expr)

# FUNCTION EVALUATION: null_memory replaces the data, so only the expression answers.
y_star = ansr.predict_y(None, None, x_star, expression=expr, conditioned=False)

# Measured on y = 2.5*x1 + 1, 64 draws: complexity median 234,500 [76,000, 454,500] against a
# true mu of 146,000, agreement 0.09 -- approximately centred and far too diffuse to feed back
# as a point. A single greedy decode of that same distribution read 342,000 and looked like a
# 2.3x overshoot, which is exactly the error this shape exists to prevent.
```

`score_outliers` returns **raw per-point probabilities and nothing else**. There is no
operating-point argument and no calibration table: no such table exists, and the head's
only quantitative claim (AUROC 0.9888) is a POOLED, in-distribution val number. The
deployment question — per-instance ranking of points within one problem — has never been
measured, and the measured per-problem behaviour is much weaker (median P ≈ 0.42 for a
single lone outlier). Two caveats belong in the docstring: the head reads the data-set
encoder only, so scores condition on `(X, y)` and never on any expression; and it degrades
above roughly 10% contamination, the ceiling it was trained under.

Per-point residual prediction is NOT exposed. The encoder head that once served it is gone;
the capability returns as a `<predict_residual>` decoder block built like `<predict_y>`, and
this section is rewritten against that when it lands.

The honest framing for any number it produces: measured on the training prior (2026-08-27)
the residual is invisible to model-free methods — a nearest-neighbour smoother recovers
`|residual|` at Spearman rho = +0.06, and robust noise-level estimation off the neighbour
ruler is biased ~81x — because the noise sits ~69x below how much f moves between
neighbouring points. Predicting it therefore REQUIRES the encoder to have internalized the
curve. That is the point of the task, and the reason nothing may be quoted from this head
until it is measured against that baseline.

`predict_constants` accepts prefix tokens (explicit or tagged) or an infix string, any
variable names, with `'<constant>'` marking the slots. It returns values in slot order
plus per-slot nibble logits so callers can read confidence. Grammar-forced decode inside.

## Fixing `complexity=`

`complexity` is load-bearing in two places now — a `fit` option and its own verb — so the
four defects against it are blocking, not cosmetic:

* it emitted a `<prompt>` / `</prompt>` wrapper carrying **zero training signal** in this
  checkpoint (measured: ~32% weaker μ conditioning than the trained bare block);
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
have no decode-time enforcement whatsoever. Under principle 6 the choice is to implement or
withdraw, and the owner ruled withdraw. The grammar machinery that would implement them
(`constrain_ieee754`) stays, so they can return as real constraints if they earn it.

The training-side machinery behind them — the `<prompt>` wrapper, the term sampling in
`PromptFeatureExtractor`, the `prompt_metadata` side-channel — also **stays, marked as legacy**
(owner ruling 2026-08-26). It is the first-generation promptable-property mechanism: typed
sections nested in one wrapper, in fixed order, with the payload out-of-band. v24 replaced that
shape with bare prefix ELEMENTS the harness force-feeds and permutes per instance
(`<complexity>`, `<mask_all>`, `<hypothesize>`). Nothing in a v24 config reaches the old lane.

**This is the refactor trigger.** The next promptable property must not be added as another
`<prompt>` section — that inherits a wrapper the model was never trained to read and a metadata
channel with no decode-time meaning. Adding promptable properties is the moment to reconcile the
two lanes: absorb the term sampling into the element grammar, or delete it.

## The conditioning knob

`condition_dropout: 0.1` means one training instance in ten routes to the learned `null_memory`
instead of the encoder's. That is a first-class trained mode, so it is a first-class parameter:

* `fit(conditioned=False)` / `infer(conditioned=False)` — propose candidates from the model's
  PRIOR over expressions, then refine and score them against the data as usual. Not a blind
  decode: the data still selects the winner, it just does not shape the proposals.
* `predict_complexity(conditioned=False)` — the model's prior over mu, the reference a
  conditioned reading should be compared against.
* `predict_constants(..., conditioned=False)` — the constants the model finds typical for a
  SHAPE, with no data to fit.
* `predict_y(..., expression=e, conditioned=False)` — function evaluation.

`X`/`y` may be `None` whenever `conditioned=False`; a placeholder support set is synthesized and
discarded by the null substitution. On a checkpoint without `optional_condition` the knob raises
`CapabilityUnavailable` rather than silently conditioning.

**One caveat on `predict_y` here.** T16 was trained with the block WITHHELD from unconditioned
instances (an earlier ruling that predicting y* with a nulled memory is ill-posed). That is true
only for the prefix placement; with the expression in scope it is function evaluation, which is
well-posed. The gate was corrected 2026-08-26 to pin the block to the suffix on unconditioned
instances instead of dropping it, so the circumstance trains from the next run on. Until then,
`predict_y(expression=..., conditioned=False)` on a T16 checkpoint queries something the model
never saw, and the measured floor reflects that, not a limit of the approach.

## Not in this document

Per-instance outlier AUROC/AUPRC, and the same metric on FastSRB with synthetic
contamination, must be measured before the outlier head is presented as a shipped
capability. Exposing the verb does not require them; claiming the head works does.

**The baseline they must beat is now measured** (2026-08-27, 600 problems off the T17 train
prior): per-problem AUROC of |nearest-neighbour difference| on contaminated problems is
**0.822** median — 0.997 with a single outlier, 0.942 with 2-5, 0.803 with 6+. So the bar is
~0.82 per problem, not 0.5, and the lone-outlier case is exactly where the trivial baseline
is strongest and the head's measured median P ~ 0.42 looked weakest.

The same will apply to residual prediction when the `<predict_residual>` block lands: its
per-instance accuracy must be reported against the smoother baseline above, split by noise
level, before any claim is made for it.

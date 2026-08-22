# v23.x model-version registry

The canonical, in-repo list of what each public `v23.x` minor version is. Each version's training config
is `configs/<version>/`.

| version | status | what changed vs v23.0 | config |
|---|---|---|---|
| **v23.0** | trained (3M / 20M / 120M / 1B + ablations) | baseline. `literal_prior` (constants) = normal(0,5); `support_sampler` = uniform X clamped to +/-30, range endpoints normal(0,10) | `configs/v23.0-*` |
| **v23.2-120M** | trained | "full Cauchy / wide inputs": constants cauchy **and** X-range endpoints normal -> **cauchy** + clamp removed. The wide-input model; +14pt on OOD-magnitude FastSRB at zero in-distribution cost | `configs/v23.2-120M` |
| **v23.3** | no new training (inference-only) | KV-cache fast decode on the v23.0 weights (`use_cache:true`); used for the cross-scale test-time-compute sweep | (uses v23.0 weights) |

**Cross-cutting note.** v23.2 removes the +/-30 X-input clamp (the encoder input-magnitude axis) and pairs it
with heavy-tailed (cauchy) X endpoints, making it the wide-input model; v23.0 is the clamped baseline.

## flash-ansr compatibility

Every bundle in this tree is a **pre-v24** config. Pre-v24 configs work only with
**flash-ansr <= 0.12.1** — the last release of the line that supported them
(`pip install "flash-ansr<0.13"`). flash-ansr 0.13.0 and later target **v24 configs only**;
v24 bundles are not in this tree yet.

| configs | supported by |
|---|---|
| every bundle here (`v23.0-*`, `v23.2-*`) | flash-ansr <= 0.12.1 |
| v24 bundles | flash-ansr >= 0.13.0 |

Nothing is deleted. These bundles stay as the record of how the v23.x models were trained, and
they still run under the pinned older release.

Two concrete reasons a pre-v24 bundle does not load on the current line:

- **The engine.** Every pre-v24 bundle pins `simplipy_engine: 'dev_7-3'`. flash-ansr 0.13.0
  requires `simplipy>=0.13,<0.14`, and simplipy refuses that asset from 0.12 on:
  `IncompatibleArtifactError: asset 'dev_7-3' is a generation-1 artifact (retired hyper-operator
  vocabulary), served only by simplipy <= 0.11`. The current engine family is `acj-*`.
- **The models.** flash-ansr 0.13.0 removed v23-era model support (CHANGELOG, `0.13.0` →
  `Removed`), so the checkpoints these bundles produced are not loadable by this line either.

`configs/test/` is not a model bundle: it is the test fixture set and tracks the current engine
(`acj-4-3`). `configs/test_set/` still pins `dev_7-3` and follows the pre-v24 boundary above.

### `v23.0-20M-A-Y{1,10,50K}` are near-duplicates of the `-A-S*` arms

These three bundles were the SymPy-simplification arm of the `A` ablation. flash-ansr removed the
`simplify='sympy'` path, so their `catalog_train.yaml` was migrated to `simplify: true` — the same
SimpliPy setting the `-A-S*` arms use. What is left of the distinction:

| bundle | nearest `-A-S*` arm | remaining difference |
|---|---|---|
| `v23.0-20M-A-Y1` | `v23.0-20M-A-S1` | `sample_strategy.max_tries` 20 vs 4 |
| `v23.0-20M-A-Y10` | `v23.0-20M-A-S10` | `sample_strategy.max_tries` 20 vs 4 |
| `v23.0-20M-A-Y50K` | `v23.0-20M-A-S100` | `sample_strategy.max_tries` 20 vs 4; `steps` 391 vs 819200 |

`v23.0-20M-A-Y50K`'s `steps: 391` matches no `-A-S*` arm (`S1` 8192, `S10` 81920, `S100` 819200).
The bundles are kept because they record what those runs trained on; there is no longer a distinct
SymPy recipe to find.

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

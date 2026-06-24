# v23.x model-version registry

The canonical, in-repo list of what each `v23.x` minor version is. Previously this lived only in a
Claude memory file + scattered config comments + commit messages; promoted here (cos sweep, 2026-06-22)
so the repo carries its own source of truth. Each version's training config is `configs/<version>/`.

> **v23.6 / v23.7 are PROVISIONAL and subject to change** (configs committed but not trained; design +
> magnitudes may still move). Do not cite them as final. Everything else here is settled.

| version | status | what changed vs v23.0 | config |
|---|---|---|---|
| **v23.0** | trained (3M / 20M / 120M / 1B + ablations) | baseline. `literal_prior` (constants) = normal(0,5); `support_sampler` = uniform X clamped to +/-30, range endpoints normal(0,10) | `configs/v23.0-*` |
| **v23.1-120M** | trained (2026-06-18) | "partial Cauchy": constants prior normal -> **cauchy**; the +/-30 X-input clamp **removed** (X-range endpoints stay normal, so wider inputs barely bite) | `configs/v23.1-120M` |
| **v23.2-120M** | trained | "full Cauchy / wide inputs": constants cauchy **and** X-range endpoints normal -> **cauchy** + clamp removed. The genuine wide-input model; +14pt on OOD-magnitude fastsrb, zero in-distribution cost (`experimental/eval/v23x_prior_width_ood.md`) | `configs/v23.2-120M` |
| **v23.3** | no new training (inference-only) | KV-cache fast decode on the v23.0 weights (`use_cache:true`); used for the cross-scale test-time-compute sweep | (uses v23.0 weights) |
| **v23.4** | **RETRACTED** | was "KV + constant pruning"; dropped (see the v23.3/v23.4 campaign record) | none |
| **v23.5-120M** | trained (2026-06-18) | from-scratch **optcond**: optional-condition + 15% CFG-style condition-dropout (`unconditional_prob:0.15`) + clamp removed (normal X kept). NOT the quantization grid (that is an eval study, not a version) | `configs/v23.5-120M` |
| **v23.6-120M** | **PROVISIONAL, subject to change** — config committed (29be70c), NOT trained | **XSA** (exclusive self-attention in the decoder) + X-input cauchy widened 10 -> 20; constants kept at cauchy(0,5) (v23.2 level). Design + magnitudes may still change before training; do not cite as final | `configs/v23.6-120M` |
| **v23.7-120M** | **PROVISIONAL, subject to change** — config committed (29be70c), NOT trained | XSA + X-input cauchy 10 -> 20, **and** constants prior widened 5 -> 10 (the A/B variable: does widening the *constant* prior help, or just pay v23.1's in-range cost?). Design + magnitudes may still change before training; do not cite as final | `configs/v23.7-120M` |

**Next free minor = v23.8+.**

**Cross-cutting note.** v23.1/v23.2/v23.6/v23.7 all remove the +/-30 X-input clamp (the encoder
input-magnitude / IEEE-754 root-cause axis), but the removal only bites when paired with heavy-tailed X
endpoints (v23.1 keeps normal X endpoints, so it is a near-no-op there; v23.2 is the real wide-input model).
v23.5 is the orthogonal CFG/condition-dropout axis. XSA (v23.6/v23.7) disables KV-cache fast decode until
the "fresh own-value" cached-decode fix is implemented.

*Detail + experimental results behind these assignments: `experimental/eval/v23x_prior_width_ood.md`,
`experimental/eval/init_ab/README.md`, and the v23.3/v23.4 campaign record.*

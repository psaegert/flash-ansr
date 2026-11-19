# Split data generation for front-directed fine-tunes (`flash_ansr.datagen`)

Branch `datagen/split-pipeline`. Purpose: produce ~1M training-prior problems with fitted candidate pools from a
120M checkpoint (theta0-120m = v25.0-T8-120M step 1.2M) at maximum throughput, for the pool-size-conditioned
front-member fine-tune (flash-ansr-rl LEARNINGS T23).

## Stages
1. **Sample (GPU nodes)** - `datagen/sample.py`: problems from the model's training prior
   (`datagen/problems.py`: the bundle's catalog / noise / support-size prior, unused columns zeroed, 512 clean
   validation rows), pool size K per problem (`log2uniform:1:512`: K a power of two, log2 K uniform, mean 102.3),
   raw decode only (no simplification on the GPU node). Packing: equal-K problems decoded together, P x K rows per
   call; encoder once per problem, prefix prefilled once per problem and expanded; cross-attention through a new
   grouped path (`Attention.forward_static_cross`: rows of one problem folded into its query axis, no per-row copy
   of the memory K/V); self-attention over the written slots only (`StaticKVCache.live_slice`). Both model changes
   are opt-in: the deployed decode paths are unchanged. Output: raw shards (`datagen/shards.py`).
2. **Fit (CPU nodes)** - `datagen/fit.py` (written): the deployed post-processing (`_postprocess_sampled`), then the
   unchanged `_fit_refine` + `_build_result` with a serial refiner per process; fresh-point FVU per ledger row on the
   clean validation rows; rows with fresh FVU >= 1 dropped (dominated by the constant model); each kept row carries
   its family root's tokens after post-processing (simplified, not the raw emission). Output: fitted shards.
3. **Train (GPU node)** - reads fitted shards; target = a random front member of the pool, K as an input.

## Equivalence gates
- Stage 2 == deployed `FlashANSR.fit` on the same raw draws (sampler monkeypatched to return them): fitted-row count
  and kept-row support FVUs identical. PASSED on 4 local problems (K 2-122).
- Stage 1 == the model's distribution: every packed row's accumulated score equals its teacher-forced log-likelihood
  under the plain dynamic forward (per problem). PASSED: max deviation 2.3e-5 nats (T8-3M), 1.5e-5 (120M).
- Grouped cross-attention == per-row expanded cross-attention (unit test). Existing model / inference / ladder tests:
  188 passed, 5 skipped. `tests/test_datagen.py`: 5 passed.

## Measurements (2026-11-19)
GPU, A100-40GB, theta0-120m, 512 draws of one problem per call (Helix jobs 15959813, 15959919):
| decode | max_len | s / problem |
|---|---|---|
| static, fp32 | 160 | 2.0 |
| static, TF32 | 160 | 1.76 |
| static, TF32 | 96 | 1.35 |
| static, TF32 | 64 | 0.76 |
| dynamic + KV cache, TF32 | 160 / 96 | 1.6 / 1.55 |
- Draw lengths (EOS position incl. 3 prompt tokens): median 50, 90% 72, 99% 92, 99.9% 105, none > 128.
  **21% of draws exceed 64 tokens**: the library default `max_len=64` (FlashANSR.load's SoftmaxSamplingConfig)
  truncates them; the evaluation pools used 160. Use max_len >= 112 here.
- Static decode allocates self-KV for decoder_max_seq_len = 256 (~24 MB per row in fp32): at most ~1,400 rows on a
  40 GB card; packing several problems per call OOMs (also in the dynamic path, which materializes per-row cross K/V
  once memory is not broadcast).
- Encoder: ~8 ms per problem. CPU post-processing (simplify + dedup): ~1.7 s per 512 draws (moves to stage 2).
CPU, desktop single core, stage 2: K = 122 problem (110 distinct): 5.8 s = post 1.8 + refine 4.0 (36 ms refit and
16 ms simplify per distinct candidate). One problem only; a 100-problem benchmark on a Helix node is next.

## Benchmark of the split pipeline (Helix jobs 15960301 GPU, 15960302 CPU; 2026-11-19)
1,024 training-prior problems, K log2-uniform on 1..512 (mean 109 draws), max_len 128, 512 problems per shard.
- GPU stage (packed), **A40** (the job did not get an A100): TF32 ~3.2 problems/s steady (~350 draws/s); bf16
  ~3.7 problems/s (1.2x). Every verified pack: max |score - teacher-forced log p| 3e-3 to 7e-3 nats under TF32; under
  bf16 0.04 to 0.16 nats per sequence (bf16 sampling is a measurably different distribution). The unpacked path on an
  A100 did ~320 draws/s; the A40 has about half the A100's TF32 throughput and memory bandwidth, so packing is roughly
  2x on equal hardware (to confirm on A100/H200).
- CPU stage: 8.2 core-s per problem (refit 8.1 s, post-processing 0.09 s); 37 s at K = 512, ~75 ms per distinct draw
  including its ladder variants. 14% of problems keep no candidate (every fit FVU >= 1). The benchmark used only 2 of
  64 cores (2 shards; parallelism is per shard): production needs >= 64 shards per node.
- Storage: ~70 KB per problem raw, ~71 KB fitted.

## Cost projection, 1M usable problems (~1.26M sampled), K log-uniform on 1..512 (mean ~82)
Measured (above): 1.16M problems sampled (14% keep nothing).
- CPU: 1.16M x 8.2 core-s = ~2,650 core-hours = ~41 node-hours (64 cores): ~2 h on 20 nodes.
- GPU: 1.16M / 3.2 per s = ~100 A40-hours in TF32 (~13 h on 8 GPUs); A100/H200 faster (to measure). Next lever:
  row compaction (finished rows are still computed every step; about half the row-steps are wasted), ~1.5x.
- Storage: ~80 GB raw (deletable after fitting) + ~85 GB fitted.

## Choices
- Numerics: TF32 (bf16 measured at only 1.2x on an A40 and shifts sequence log-probs by up to 0.16 nats).
- K: log2-uniform on 1..512 (powers of two, so equal-K problems pack together).

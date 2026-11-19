# Split data generation for front-directed fine-tunes (`flash_ansr.datagen`)

Branch `datagen/split-pipeline`. Purpose: produce ~1M training-prior problems from a 120M checkpoint (theta0-120m =
v25.0-T8-120M step 1.2M), each with a fitted pool of K draws and its front-member training targets, at maximum
throughput. It serves the pool-size-conditioned front-member fine-tune (flash-ansr-rl `runs/t23_120m/PLAN.md`,
LEARNINGS T23). Production job scripts: flash-ansr-rl `runs/datagen/helix/` (`dg_submit.sh` chains the stages).

## Stages
1. **Sample (GPU nodes)**: `datagen/sample.py`.
   - Problems come from the model's training prior (`datagen/problems.py`: the bundle's catalog, noise and
     support-size prior, unused columns zeroed, plus 512 clean validation rows).
   - Pool size K per problem: `log2uniform:1:512` (K a power of two, log2 K uniform, mean 102.3).
   - Raw decode only; there is no simplification on the GPU node.
   - Packing: equal-K problems are decoded together, P x K rows per call. The encoder runs once per problem, and the
     prefix is prefilled once per problem and expanded.
   - Two opt-in model paths make packing fit in memory, and the deployed decode paths are unchanged:
     - cross-attention goes through a grouped path (`Attention.forward_static_cross`: the rows of one problem are
       folded into its query axis, with no per-row copy of the memory K/V);
     - self-attention reads only the written slots (`StaticKVCache.live_slice`).
   - Output: raw shards (`datagen/shards.py`).
2. **Fit (CPU nodes)**: `datagen/fit.py`.
   - The deployed post-processing (`_postprocess_sampled`), then the unchanged `_fit_refine` + `_build_result`, with
     one serial refiner per process.
   - Each ledger row gets a fresh-point FVU on the clean validation rows. Rows with fresh FVU >= 1 are dropped (worse
     than the mean on fresh points).
   - Every raw draw is traced to its family, aligned with the raw shard: its status, its action (distinct raw
     expression) and its kept family. Output: fitted shards.
3. **Targets (CPU nodes)**: `datagen/targets.py`.
   - The front of each pool, as the offline oracle defines it (flash-ansr-rl `kfront_check.py`; see Definitions).
   - Every front member gets 1/|front|, read from the raw shard as the model emitted it.
   - The control has the same number of members, drawn uniformly from the kept actions.
   - Output: target shards (the model input data plus both target sets). The trainer reads only these.

## Definitions (as the offline oracle, so the oracle's numbers apply to these targets)
- **Family:** the ledger rows of one deduplicated draw: its fit, its constant-ladder variants (`parent` set) and its
  typed-thaw re-fits (the same beam, no parent). They share the draw's log-prob, which is the family key.
- **Action:** a distinct raw expression of the pool, the unit of the target. Two raw forms that post-processing
  collapses into one family are two actions with the same rows.
- **Row point:** b = mdl / 1000 bits, y = max(log10(max(fresh FVU, eps)), floor). The floor is -7 for a clean
  problem and max(-9, log10(level^2 / 100)) for noise level > 0.
- **Front:**
  - An action is dominated by another action when every one of its rows is dominated by some row of the other.
  - The constant model (8 bits, y = 0) removes an action whose every row it dominates.
  - An action without rows is never a member.
- **Deviation from the oracle:** dropping rows with fresh FVU >= 1 removes trivial members, i.e. cheap expressions
  worse than the mean on fresh points. Measured on the theta0-120m evaluation pools, it changes the front of 46% of
  FastSRB pools (Jaccard 0.85 to the oracle's front), 15% of training pools (0.96), 13% of noisy training pools
  (0.97) and 28% of black-box pools (0.93).

## Gates (all passed, 2026-11-19)
- **Stage 1 == the model's distribution:** every packed row's accumulated score equals its teacher-forced
  log-likelihood under the plain dynamic forward. Max deviation 2.3e-5 nats on T8-3M and 1.5e-5 on 120M (fp32), and
  3e-3 to 7e-3 nats under TF32 on 120M.
- **Stage 2 == the deployed `FlashANSR.fit`** on the same raw draws (sampler monkeypatched to return them): identical
  fitted-row count and kept-row support FVUs.
- **Draw map:**
  - Post-processing with the trace's `simplify_map` is byte-identical to the plain call.
  - Every valid draw lands in the family whose beam is its dedup class's first draw.
  - A draw has a kept family iff some row of its draw is kept.
  - Draws of one raw expression share their action and family.
  - Tested on 12 problems at K = 48, which include typed-thaw rows. The first family definition (parent chain) split
    those rows off and failed here.
- **Front == the oracle's (G0 of the plan):** the shared front code, fed the evaluation pools' draws and ledgers, gives
  the identical front on all 1,146 theta0-120m problem pools (4 sets x 3 seeds;
  flash-ansr-rl `runs/t23_120m/g0_front_check.py`).
- **Unit tests:**
  - Grouped cross-attention == per-row expanded cross-attention.
  - The front definition holds on a hand-made pool.
  - sample -> fit -> targets end to end: weights sum to 1, and every target is one of the problem's raw draws.
  - `tests/test_datagen.py`: 8 passed. Existing model / inference / ladder tests: 182 passed, 6 skipped (the skips
    need a v24 checkpoint).

## Measurements (2026-11-19)
GPU, A100-40GB, theta0-120m, 512 draws of one problem per call (Helix jobs 15959813, 15959919):
| decode | max_len | s / problem |
|---|---|---|
| static, fp32 | 160 | 2.0 |
| static, TF32 | 160 | 1.76 |
| static, TF32 | 96 | 1.35 |
| static, TF32 | 64 | 0.76 |
| dynamic + KV cache, TF32 | 160 / 96 | 1.6 / 1.55 |
- Draw lengths (EOS position incl. 3 prompt tokens): median 50, 90% 72, 99% 92, 99.9% 105, none > 128. 21% of draws
  exceed 64 tokens. The library's old default `max_len=64` truncated them; since PR #74 it decodes up to the model's
  limit. The sampler uses 128.
- Static decode allocates self-KV for decoder_max_seq_len = 256 (~24 MB per row in fp32), so at most ~1,400 rows fit
  on a 40 GB card. Packing several problems per call ran out of memory before the grouped cross-attention path.
- Encoder: ~8 ms per problem. CPU post-processing (simplify + dedup): ~1.7 s per 512 draws (done in stage 2).

## Benchmark of the split pipeline (Helix jobs 15960301 GPU, 15960302 CPU; 2026-11-19)
1,024 training-prior problems, K log2-uniform on 1..512 (mean 109 draws), max_len 128, 512 problems per shard.
- **GPU stage (packed), on an A40** (the job did not get an A100):
  - TF32: ~3.2 problems/s steady (~350 draws/s). bf16: ~3.7 problems/s (1.2x).
  - Verified packs: max |score - teacher-forced log p| is 3e-3 to 7e-3 nats under TF32, but 0.04 to 0.16 nats per
    sequence under bf16 (bf16 sampling is a measurably different distribution).
  - The unpacked path on an A100 did ~320 draws/s. The A40 has about half the A100's TF32 throughput and memory
    bandwidth, so packing is roughly 2x on equal hardware (to confirm on A100/H200).
- **CPU stage:** 8.2 core-s per problem (refit 8.1 s, post-processing 0.09 s); 37 s at K = 512, ~75 ms per distinct
  draw including its ladder variants.
  - 14% of problems keep no candidate (every fit has FVU >= 1).
  - Parallelism is per shard, and the benchmark had only 2 shards, so it used 2 of 64 cores. Production needs at
    least 64 shards per node; at 256 problems per shard the pilot has ~580.
- **Storage:** ~70 KB per problem raw, ~71 KB fitted.

## Cost projection, 1M problems with a front (~1.16M sampled at the measured 14% loss)
- CPU: 1.16M x 8.2 core-s = ~2,650 core-hours = ~41 node-hours (64 cores), ~2 h on 20 nodes.
- GPU: 1.16M / 3.2 per s = ~100 A40-hours in TF32 (~13 h on 8 GPUs). A100/H200 are faster (to measure).
  - Next lever: row compaction. Finished rows are still computed every step, about half of all row-steps, so
    skipping them is worth ~1.5x.
- Storage: ~80 GB raw + ~85 GB fitted + target shards (input data plus a few sequences per problem). The target stage
  reads the raw shards, so they are kept until training ends.

## Choices
- Numerics: TF32 (bf16 is only 1.2x faster on an A40 and shifts sequence log-probs by up to 0.16 nats).
- K: log2-uniform on 1..512 (powers of two, so equal-K problems pack together).
- Seeds: problem streams use `SeedSequence([seed, 0])` with run seeds from 1000 up. The evaluation sets were drawn
  with `default_rng(20261116...20261199)` and a fixed 512-point support, so the problems are disjoint by
  construction; skeletons repeat, as in pretraining.

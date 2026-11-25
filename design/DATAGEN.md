# Split data generation for front-directed fine-tunes (`flash_ansr.datagen`)

Branch `datagen/split-pipeline`. It produces ~1.16M training-prior problems for the pool-size-conditioned
front-member fine-tune (flash-ansr-rl `runs/t23_120m/PLAN.md`, LEARNINGS T23). Each problem gets one fitted pool of
512 draws from the sampler, the released and decayed T8-120M, plus training targets for any pool size K. Production
job scripts: flash-ansr-rl `runs/datagen/helix/` (`dg_submit.sh` chains the stages; `dg_report.py` summarises a run).

## Design
- **One pool of 512 draws per problem.** The training pool size K is assigned afterwards, in the target stage.
  - Draws within a pool are i.i.d., so the first K draws are an exact K-draw pool for every K.
  - One dataset serves every arm and every K prior, with paired comparisons on identical draws.
  - Builds: T, T-noK and B use `uniform:1:512`; T-512 uses `fixed:512`; a later `loguniform:1:512` round needs no
    new sampling.
- **The problem data are stored once, in the raw shard.** Fitted and target shards hold identifiers, and the
  problems line up by position.
- **No numerical shortcuts.** Sampling runs under bf16 autocast, the model's own training numerics, and was checked
  at the distribution level (below). The deployed post-processing, refit and ladder are unchanged.
- **The optimizations remove waste only:**
  - finished row groups are dropped from the decode;
  - CPU fitting runs per problem, not per shard;
  - rows per decode are capped by free GPU memory;
  - the problem data are never duplicated.

## Stages
1. **Sample (GPU nodes)**: `datagen/sample.py`, `python -m flash_ansr.datagen sample` (defaults `--k-spec fixed:512
   --chunk-rows 8`; production `--amp bf16 --rows 8192`).
   - Problems come from the model's training prior (`datagen/problems.py`: the bundle's catalog, noise and
     support-size prior, unused columns zeroed, plus 512 clean validation rows).
   - Raw decode only; there is no simplification on the GPU node.
   - **Packing:** one decode holds groups of R = 8 rows. A problem fills ceil(K / R) groups, so a 512-draw problem
     fills 64. Groups of any problems pack together, about 4,096 rows per decode (8 problems).
   - The encoder runs once per problem, and the prefix is prefilled once per group.
   - **Compaction:** every 8 steps, once a quarter of the active groups have ended every row, those groups are written
     out and dropped.
     - The kept rows move to the front of the existing cache buffers in place, one layer at a time, so no second
       cache is ever allocated.
     - In a 3M test this removed a third of the decode row-steps (30,720 -> 20,480), with every score still equal to
       its teacher-forced log-likelihood.
   - **Memory:** rows per decode are capped at 60% of free GPU memory. The self-attention cache is 6.2 MiB per row on
     120M at max_len 128 in fp32, and 3.1 MiB under bf16.
   - **Two opt-in model paths** make packing fit in memory; the deployed decode paths are unchanged:
     - grouped cross-attention (`Attention.forward_static_cross`: a group's rows are folded into its query axis, with
       no per-row copy of the memory K/V);
     - a live-slice self-attention read (`StaticKVCache.live_slice`).
   - **CPU/GPU overlap (default; `--serial` turns it off):** the next shards' problems are generated in a spawned
     producer process, and each finished shard is packed and written by a background thread while the next one
     decodes. The output is byte-identical to the serial path. The pilot's H200s were busy 78-82% of the time against
     95-98% for A100s and A40s, because the serial CPU steps (problem generation, writing) take a larger share of a
     faster GPU's shard. Each `[sample]` line reports where the GPU loop's time went.
     - Helix devel A40, production settings, 3 shards of 256 problems (job 16009872): problem generation (5.8 s per
       shard) and writing (1.8 s) left the GPU loop (0.1 s and 0 s); 110 s instead of 117 s per shard; the shards are
       byte-identical. What stays on the loop: decoding (~105 s on the A40), the per-problem encoder calls (2.3 s) and
       trimming (0.7 s).
     - On an H200 (decoding ~32 s per shard) that is ~42 -> ~35 s per shard, about 3,800 draws/s instead of 3,160.
   - **Generator per shard:** each shard samples with a generator seeded from (run seed, shard index), so a resumed
     run writes the same shards as an uninterrupted one. Until 2026-11-21 one generator served a whole run; a resumed
     run then reused the random numbers of the shards it skipped (pilot tasks 0, 3 and 8: valid draws from the
     model's distribution, correlated in their random numbers with earlier shards of the same task, not
     reproducible).
   - Output: raw shards (problem data, 512 raw token sequences and their log-probs per problem).
2. **Fit (CPU nodes)**: `datagen/fit.py`.
   - The deployed `_postprocess_sampled`, then the unchanged `_fit_refine` + `_build_result`, with one serial refiner
     per worker.
   - **Parallelism is per problem.** At most 4 shards are loaded at once, and a shard is written when its last
     problem returns, so a node never idles on a last, partial wave of shards.
   - Each ledger row gets a fresh-point FVU on the clean validation rows. Rows with fresh FVU >= 1 are dropped by
     default; `--keep-all-rows` (2026-11-26) stores them too (shard flag `rows_all`). The front and control targets
     use only rows with fresh FVU < 1 either way (tested: identical targets from both shards). The negatives need the
     FVU >= 1 rows: their MDL cannot be recovered from the raw tokens, since the price is taken after the fitted
     expression is canonicalized, which can collapse it.
   - Every raw draw is traced to its status, its action (distinct raw expression), its family, and whether its
     simplified expression is the problem's true skeleton (`draw_gt`, 2026-11-26).
   - Workers default to one BLAS thread each.
   - `DG_FAULTHANDLER=<s>` dumps every thread's stack every s seconds, to diagnose a hang without ptrace.
   - Output: fitted shards (identifiers, kept candidates, the per-draw map).
3. **Targets (CPU nodes)**: `datagen/targets.py`, `python -m flash_ansr.datagen targets --k-spec SPEC`.
   - Per problem, K is drawn deterministically from (problem id, spec), and the pool is its first K draws.
   - The pool's front follows the offline oracle's definition (see Definitions).
   - FRONT targets: every member at 1/|front|, read from the raw shard as the model emitted it.
   - CONTROL: |front| uniformly drawn kept actions at 1/|front|.
   - Every problem is recorded; one with an empty front has `n_front = 0` and no target.
   - Output: one target directory per spec (`tgt_uniform_1_512`, `tgt_fixed_512`, ...), with sequences, weights and
     K only. The trainer reads the problem data from the raw shard of the same name.
4. **Negatives (CPU nodes; T23 design v5, owner-approved 2026-11-25)**: `datagen/negatives.py`,
   `python -m flash_ansr.datagen negatives --k-spec SPEC [--pairs 8]`, from an all-rows fit.
   - Dominance pairs (y+, y-) per problem at its K: y- draw-uniform over the K-pool's non-members (status ok, a stored
     row, not the true law), y+ uniform over the K-pool front members that dominate y-'s best-FVU row.
   - An action's rows are its FVU < 1 rows if any, else its FVU >= 1 rows; y = log10 FVU_val unclipped above. The
     virtual point V = (-inf bits, y = 0) dominates every FVU >= 1-only action, so layer 1 is exactly the front, and V
     is never a positive. A y- that only V dominates is dropped and counted.
   - Each pair carries y-'s layer in the problem's full pool (non-dominated sorting, built once per problem: family-
     wise dominance between two actions does not depend on the pool); the trainer weights by log2(layer), the layer,
     or 1.
   - Output: one negative directory per spec (`neg_uniform_1_512`), with the pairs' sequences, layers and the
     representative rows, plus per-problem counts (invalid, unfitted, true-law, dropped) and layer statistics. The
     flash-ansr-rl t23 tools add the reference log-probs (`ref_logprobs.py`) and the build report (`neg_report.py`).

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
- **Target:** t_K(x | D) = E[1{x in F} / |F| given |F| > 0], with each pool carrying total mass 1.
  - t_1 is the sampler conditioned on survival (valid, fitted, fresh FVU < 1, not dominated by the constant model).
  - The problems with a target depend on K, because small pools have an empty front more often. `dg_report.py`
    measures this per K octave.
- **Deviation from the oracle:** dropping rows with fresh FVU >= 1 removes trivial members, i.e. cheap expressions
  worse than the mean on fresh points. On the theta0-120m evaluation pools it changes the front of:

  | set | pools changed | Jaccard to the oracle's front |
  |---|---|---|
  | FastSRB | 46% | 0.85 |
  | training | 15% | 0.96 |
  | noisy training | 13% | 0.97 |
  | black-box | 28% | 0.93 |

## Gates (all passed, 2026-11-19, 2026-11-21 and 2026-11-26; `tests/test_datagen.py`: 14 passed)
- **Stage 1 == the model's distribution:** every row's accumulated score equals its teacher-forced log-likelihood
  under the plain dynamic forward and its own problem's memory. This holds for equal-K packing, chunked mixed-K
  packing and compaction.
  - Max deviation: 2.3e-5 nats on T8-3M and 1.5e-5 on 120M in fp32; 3e-3 to 7e-3 nats on 120M under TF32.
- **Stage 2 == the deployed `FlashANSR.fit`** on the same raw draws (sampler monkeypatched to return them): identical
  fitted-row count and kept-row support FVUs.
- **All-rows fit == default fit where it matters:** the same FVU < 1 rows (values and families), identical front and
  control targets; the refit of existing data is checked shard by shard (`neg_report.py`).
- **Negatives:** on a hand-made pool, family-wise dominance, V and the layers equal a naive repeated peeling, and the
  pair rules hold (y+ dominates y-'s best row; a draw only V dominates is dropped). On real 3M pools every pair's
  sequences are first-K draws, y+'s dominating row dominates y-'s representative, every y- lies in layer >= 2, and the
  K-pool front size equals the front targets'.
- **Overlapped sampling == serial sampling:** byte-identical raw shards; a run resumed after deleting a shard
  rewrites exactly that shard, identically.
- **Problem-parallel fitting == serial fitting:** every array of every fitted shard is identical except the timings.
  Two bugs were found on the way: a double `pop`, and a semaphore-blocked feeder that turned the resulting exception
  into a silent hang in `Pool.terminate()`.
- **Draw map:**
  - Post-processing with the trace's `simplify_map` is byte-identical to the plain call.
  - Every valid draw lands in the family whose beam is its dedup class's first draw.
  - A draw has a kept family iff some row of its draw is kept.
  - Draws of one raw expression share their action and family.
  - Tested on 12 problems at K = 48, which include typed-thaw rows. Grouping by parent chain split those rows off and
    failed this test.
- **Front == the oracle's (G0 of the plan):** the shared front code gives the identical front on all 1,146
  theta0-120m problem pools (4 sets x 3 seeds; flash-ansr-rl `runs/t23_120m/g0_front_check.py`).
- **Targets end to end:**
  - K is deterministic and within the pool, and differs between specs.
  - Weights sum to 1 where there is a front.
  - Every target is one of the problem's first K raw draws.
  - The target shards carry no problem data.
- **Existing tests:** the model / inference / ladder suites gave 182 passed, 6 skipped (the skips need a v24
  checkpoint) after the decoder changes.

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
  - 21% of draws exceed 64 tokens. The library's old default `max_len=64` truncated them; since PR #74 it decodes up
    to the model's limit. The sampler uses 128.
- Packing several problems per call ran out of memory before the grouped cross-attention path existed.
- Encoder: ~8 ms per problem. CPU post-processing (simplify + dedup): ~1.7 s per 512 draws (done in stage 2).

Split-pipeline benchmark (Helix jobs 15960301 GPU, 15960302 CPU): 1,024 problems, K log2-uniform (mean 109),
equal-K packing, no compaction.
- **GPU, A40:** TF32 ~350 draws/s. bf16 is 1.2x faster but shifts sequence log-probs by 0.04 to 0.16 nats.
- **CPU:** 8.2 core-s per problem, ~75 ms per distinct draw including its ladder variants; 37 s at K = 512. 14% of
  problems keep no candidate.
- **Storage per problem:**
  - raw: 69.6 KB, of which 96% is the problem data (support x 27.8 KB, validation x 27.6 KB, the y vectors
    11.7 KB);
  - draws: ~23 bytes each, compressed;
  - fitted (then still with a data copy): 72.6 KB.

## Helix smoke run, final configuration (jobs 15975229-31, 2026-11-19)
64 problems x 512 draws, the release sampler, theta0's prior files, 16 problems per shard.
- **GPU (A100-SXM4-40GB):**
  - Rows per decode were capped at 3,792 by free memory (6.2 MiB of cache per row).
  - Verified pack: max |score - teacher-forced log p| = 9.3e-3 nats (TF32).
  - Throughput was 402 draws/s cumulative, including model loading and warm-up. The later shards ran several
    times faster, so the pilot measures the steady rate.
- **CPU:** 43.3 core-s per problem (91 ms per distinct draw; 478 distinct of 512 after post-processing).
- **Yield:**
  - Every problem kept a candidate.
  - With a front: 64/64 at K = 512, 63/64 under uniform K.
  - Median |front|: 7 at K = 512, 6 at K 257-512 under uniform K.
- **Storage:** 114 KB per problem (raw + fitted + both target builds).

## Data pilot (G1), jobs 15977726-28, 2026-11-19/20
148,000 problems (16 sampling tasks x 9,250, seeds 1000-1015) x 512 draws, bf16 with the fixed cache, groups of 8
rows with compaction; 8 fitting tasks of 64 cores. Passed G1 (report: `$DG/t23/report.txt`).
- **Yield:** a front on 97.7% of problems at K = 512 and 96.7% under uniform K (gate: >= 80% in each build). Under
  uniform K the share rises with K: 67% at K = 1, 88% at K 3-4, 95% at K 33-64, 97% at K 257-512. Median |front|
  is 1 up to K = 4, 4 at K 33-128 and 6 at K 257-512; 98% of problems keep at least one candidate.
- **GPU, steady rate per task (draws/s):** H200 ~3,160, A100-80GB ~1,740, A100-40GB ~1,690, A40 ~1,130; row use
  100%. The 16 tasks took 12.5 GPU-hours (4 of each type), against the projected ~60 A40-hours. Compaction's own share
  of the speed-up was not isolated.
- **CPU:** 48.3 core-s per problem (101 ms per distinct draw; 477 distinct of 512), 11% above the smoke run;
  32 node-hours.
- **Storage:** 111 KB per problem: raw 11.7 GB, fitted 4.6 GB, the two target builds 0.04 GB each.
- **Queueing, not compute, set the wall time:** the fitting array waited 4-5 hours for CPU nodes (Helix fair-share).

## Cost projection, 1.16M problems x 512 draws (pilot = first 148k, measured)
| resource | full run | remaining 1.01M |
|---|---|---|
| GPU (pilot rates; Helix bills every GPU type alike) | ~52 H200-h / ~96 A100-h / ~146 A40-h | ~46 / ~84 / ~127 |
| CPU (48.3 core-s per problem) | ~245 node-hours | ~212 node-hours |
| storage (111 KB per problem) | ~129 GB | ~112 GB |
- Training reads the raw shards (problem data) and the target shards, so the raw shards stay. The fitted shards are
  only needed to rebuild targets.

## Choices
- **Numerics: bf16 autocast** (owner 2026-11-19; the model was trained under bf16 autocast). Checked against fp32 on
  the released 120M, 1,024 sampled sequences (flash-ansr-rl `runs/datagen/dg_bf16_check.py`, run on the local RTX
  2070):
  - KL(p_fp32 || p_bf16) is 1.9e-5 nats per token (p99 3e-4) and 1.0e-3 nats per sequence (p99 4.3e-3).
  - The per-sequence log-prob noise is mean 0.03 nats (p99 0.16), with a signed mean of +-1e-3, so it nearly cancels.
  - The deployed ranking gives log-probs zero weight. Downstream they are only family keys and ordering.
  - The earlier "0.04-0.16 nats" was this per-sequence noise, not a distribution shift.
  - A100-80GB, production packing, fixed cache (Helix job 15977725): TF32 1,340 draws/s, bf16 1,948 (1.45x); 8,192
    rows per decode instead of 4,096 add nothing. The fidelity on the A100 matches: KL 1.35e-3 nats per sequence
    under bf16, against 1.3e-5 under TF32. (Before the cache fix, job 15977177: bf16 1,502 draws/s.)
  - Cache dtype (fixed 2026-11-19, commit 2ef5535): the output head returns fp32 logits even under autocast. A cache
    created in the logits' dtype therefore kept the bf16 K/V in fp32 buffers, twice the memory, and the bf16 memory cap
    undercounted by 2x. 13 of 16 pilot sampling tasks ran out of memory on 40-48 GB GPUs before the fix.
  - The cache now takes the K/V dtype: identical draws, bitwise, at half the memory. The deployed static decoder
    (`flash_ansr_model.py`) has the same logits-dtype cache.
- K: uniform on 1..512 for T, T-noK and B (owner); every K gets the same share of training problems
  (P(K <= 8) = 1.6%). `runs/t23_120m/fig/k_distribution.png` in flash-ansr-rl compares the priors.
- Seeds: problem streams use `SeedSequence([seed, 0])` with run seeds from 1000 up. The evaluation sets were drawn
  with `default_rng(20261116...20261199)` and a fixed 512-point support, so the problems are disjoint by
  construction; skeletons repeat, as in pretraining.

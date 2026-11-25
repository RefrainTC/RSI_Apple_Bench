# Screening 29 ISO-Bench fixes on an RTX 4090

**Short answer.** We measured 29 of ISO-Bench's merged vLLM/SGLang performance fixes on one RTX 4090, in three batches. **8 passed.** Two of the eight are the same change merged twice, so they give 7 independent tasks. Most of the other 21 showed no measurable gain: the fix saved CPU work the GPU never waited for, or saved too little per step. What predicted success was asking what the load is bound by, then counting how many GPU syncs or kernel launches the fix removes per step and comparing that with the step time.

## How was a fix screened?

- **Two arms.** The *parent* is the engine at the commit just before the fix; the *fix* is the merged commit. Each was installed from the project's per-commit wheel (vLLM) or built from source (SGLang), with dependencies resolved as of the merge date plus 2 days.
- **A load chosen per PR,** written down after reading the diff and before any run, to put the changed code on the timed path.
- **Screen = 3 AB pairs + 3 A/A pairs on one GPU.** An *AB pair* is a fix run and a parent run; the ratio is fix time over parent time (below 1 = fix faster). An *A/A pair* is two parent runs; it measures noise.
- **Pass:** all 3 AB ratios at least about 2% below 1 and outside the A/A range. *Marginal:* real but under 2%. *Too noisy:* an effect about as large as the A/A spread. *No gain*, *crash*, or *not run*.
- From batch 2 on, a **prior** was written first: for each PR, the probability that it passes, with the reason. The results were appended later without editing the prior.

## Which 8 passed?

| PR | What the fix changes | Screen result | Calibrated speedup (REF) | Minutes per scoring |
|---|---|---|---|---|
| vLLM #7822 | prefix-cache blocks are marked as computed, so later requests reuse them instead of recomputing | 1.48× | 1.480 (between-job SD 0.17%) | about 5 |
| vLLM #16436 | the sampler's repetition penalty no longer forces a GPU-to-CPU sync every step | 1.09× | 1.0855 (median of 7 jobs, SD 0.54%, band ±1.0%) | about 19 |
| vLLM #12380 | decoding keeps its CUDA graphs after the first prefill (a regression from one PR earlier) | 1.5× | 1.507 (band ±1.1%) | about 9 |
| vLLM #13365 | the V1 n-gram draft proposer runs in Numba instead of Python | 5% faster per step | 1.0934 (band ±0.5%) | about 10-11 |
| vLLM #14930 | the V1 speculative-decoding rejection sampler becomes Triton kernels | 8% faster per step | 1.0909 (band ±0.5%) | about 12-13 |
| SGLang #6994 | LoRA batch preparation stops syncing with the GPU per request (about 130 syncs per step at batch 64) | 1.48× | 1.474 (8 jobs, SD 0.54%) | about 4.5 |
| vLLM #14857 | Mamba2 prefill replaces a per-sequence loop of syncs with one indexed write | 1.066× | 1.0648 (7 jobs, SD 0.14%) | about 9.5 |
| vLLM #14778 | the same change as #14857, written with `torch.vmap` (merged, reverted, re-landed as #14857) | 1.036× | 1.0365 (5 jobs, SD 0.14%) | about 7 |

*REF* is the human fix's speedup, taken as the median over several independent scoring jobs on several cards and both servers. The *band* is the ± range around REF within which a candidate "ties" the reference. Each of these is now a task under `tasks/`; #14778 is packaged next to #14857 as a duplicate, not as a separate task.

Notes on the passes:
- **#12380's parent is CPU-launch-bound** (no CUDA graphs), so its time moves when other work shares its CPU cores. Parent A/A pairs varied by up to 27% in the screen. The fix's 30 reps never moved.
- **#6994 and #16436 outputs:** SGLang groups requests by arrival time, so #6994's timed outputs are not reproducible even parent against parent. A correctness check there must use a separate untimed pass.
- **#14778 needed a different load:** its parent vLLM crashes at model construction with online FP8, so it runs Bamba-9B's architecture with dummy weights and 16 of 32 layers. Its times are not comparable with #14857's.

## Why did the other 21 fail?

| Batch | PR | Verdict | Why |
|---|---|---|---|
| 1 | #10733 (V1 FlashAttention custom-op overhead) | too noisy | A real ~5% (19 of 20 pairs faster), but single A/A pairs moved by up to 7.2%. The human fix itself would pass a 5-pair scoring rule only 16% of the time. |
| 1 | #7193 (evictor optimization) | no gain | AB 1.049 / 1.012 / 1.361; A/A up to 5%. |
| 1 | #11692 (V1 token-id copy) | no gain | AB 1.016 / 0.977 / 1.015. Needed a smaller batch to fit 24 GB. |
| 1 | #15150 (`merge_async_iterators` fast path) | no gain | The changed function takes 0.6 ms per run in total. |
| 1 | #15478 (faster top-k-only sampling) | crash | The human fix crashes: an int32 index passed to `gather`. |
| 1 | #20000 (faster tokenizer decode) | no gain | AB within 1.2%. |
| 1 | #14223 (`--generation-config` slowdown) | no gain | 5 pairs, median 1.002; A/A up to 5.7%. |
| 1 | #13244 (V1 sampling metadata) | no gain | The parent hit 97% of VRAM at the default batch; at a smaller batch, AB within 1%. |
| 1 | #21245 (logits-processor update checks) | no gain (inconclusive) | The load was bound by the arrival rate (200 requests/s), so duration barely moved. |
| 1 | #12094 (V1 scheduling overhead after a sync) | marginal | A real ~0.8% in 3 of 3 pairs, about the size of between-job drift. |
| 1 | #6779 (arrays for sampling-tensor padding) | no gain | The changed functions' time halves (2.44 → 1.13 s), but wall time does not move: the saved time was spent waiting on a GPU sync that just moves to the next sync point. |
| 2 | #7753 (chunked prefill with prefix caching) | crash (parent) | The PR enables a feature; the parent refuses the combination at its first cache hit, so there is nothing to compare against. |
| 2 | #7364 (V0 single-sequence fast paths) | no gain | CPU trims in output processing that overlaps the GPU. |
| 2 | #12563 (non-blocking guided-decoding bitmask copy) | no gain | The guided load is bound by xgrammar's CPU-side mask work (GPU 0-3% busy), so a removed sync saves nothing. A/A up to 4%. |
| 2 | #13837 (shared xgrammar context, no per-request copy) | too noisy | The apparent gains came from two single slow parent reps; per-rep medians differ by about 1%. |
| 2 | #6520 (small V2 block-manager improvements) | marginal | About 1%: all 3 fix runs faster than all 9 parent runs, but too small to score. |
| 3 | #10742 (vllm-flash-attn update for CPU overhead) | marginal | -0.7%, real but too small. |
| 3 | #21075 (Mamba2 preallocated SSM output) | marginal | -0.8%: one device-to-device copy per Mamba layer. |
| 3 | #10989 (V1 FlashAttention without slices) | no gain | Removes 2-3 cheap ops per layer, about 0.1 ms of a ~6 ms step. |
| 3 | #7874 (chunked-prefill queue order) | no gain | Changes which requests share a step, not the total work or the step count. |
| 3 | #14377 (Qwen2-VL vision tower syncs) | too noisy | The load is bound by CPU image preprocessing; A/A up to 28%. |

Four more batch-3 PRs were dropped after reading the diff, before any run: #13625 (saves memory, not time), #7058 (no model fits, and the path is inside a CUDA graph), #6814 (a load-time helper), #6627 (a loop-invariant hoist the Triton compiler already does). We later screened #13625 anyway on an idle GPU: no gain (AB within 0.2%), so dropping it was right. It is not counted in the 29.

Grouped by cause:
- **CPU work that overlaps the GPU** (V0 output processing, front-end work): saving it does not shorten the run (#6779, #7364, #6520, #15150, #20000).
- **The load was bound by something else**: CPU-side grammar masks (#12563, #13837), image preprocessing (#14377), the arrival rate (#21245).
- **Too small per step**: a few cheap ops or one copy per layer (#10989, #10742, #21075, #12094).
- **No working parent or a broken fix**: #7753, #15478.
- **Too noisy** to separate a real effect: #10733, #13837, #14377.

## What predicted success?

**Pass rates by batch.**

| Batch | Measured | Passed | Expected by the prior | Brier score of the prior (lower is better) | Ranking quality (AUC) |
|---|---|---|---|---|---|
| 1 | 13 | 2 (#7822, #16436) | no written prior | - | - |
| 2 | 8 | 3 (#12380, #13365, #14930) | 2.8 | 0.231 (a constant 3/8 scores 0.234) | 0.57 |
| 3 | 8 screened (12 listed) | 3 (#6994, #14857, #14778) | 2.8 of 12 | 0.155 (a constant base rate scores 0.188) | 0.87 |
| **All** | **29** | **8** | | | |

The priors got the total roughly right each time. In batch 2 the per-PR probabilities added almost nothing over a well-chosen base rate; in batch 3, after the rules below were applied, they did a little better and ranked much better.

**Rules that emerged.**
1. **First decide what the load is bound by:** GPU kernels, a serial CPU gap between GPU steps, a CPU path that overlaps the GPU, or the arrival rate. Only the first two give a gain.
2. **Count the saving per step.** Count the syncs or kernel launches the fix removes per step, multiply by roughly 10-30 µs each, and compare with the step time under the chosen load. #6994 removes about 130 syncs per step at batch 64 and gives 1.48×; #14857 about 60 per Mamba layer per prefill step and gives 6%; #10989 removes 3 cheap ops per layer and gives nothing.
3. **"Removes a sync" helps only where the GPU is the bottleneck.** On a CPU-bound load (guided decoding, image preprocessing) the removed sync is not on the critical path.
4. **CPU work counts when it runs serially between GPU steps.** V1's n-gram proposer runs after each step and before the next launch, so every millisecond saved there shows (#13365). V0's output processing overlaps the GPU, so savings there do not.
5. **Drop feature enablements.** A PR titled "enable X" has no working parent to compare against (#7753).
6. **Per-element syncs inside a loop are the strongest class**, after restoring CUDA graphs (#12380). Iterating over, or indexing with, a 0-d CUDA tensor syncs each time (#14857, #14778, #6994).

**Cost.** A screen costs about 1 GPU-hour per candidate. A full calibration (5 AB + 5 A/A pairs, at least 5 scoring jobs across 3 or more cards and both servers, and 3 transfer loads) costs about 1-1.5 GPU-days per task. Batch 3 took 6.06 GPU-hours of job time over 425 jobs: 1.69 for screens and 4.35 for calibrating its three passes.

## Do these tasks run on the RTX 5090?

No. Five of the seven, plus #14778, fail at the first GPU operation: the PyTorch and engine builds they pin (CUDA 12.4 or 12.6) contain no code for the 5090's architecture, and the engines' own kernels stop at sm_90 too, so swapping PyTorch alone would not help. #7822 and #16436 pin builds of the same age or older. See [../hardware.md](../hardware.md).

## What came next?

ISO-Bench's list was used up for our GPUs. We continued with a list of 88 fixes mined from vLLM and SGLang history, applying rule 2 before choosing. See [mined-fixes.md](mined-fixes.md).

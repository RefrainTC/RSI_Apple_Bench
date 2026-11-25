# Fixes mined from vLLM and SGLang history

**Short answer.** After ISO-Bench ran out, we mined our own list of 88 performance fixes from vLLM and SGLang history, then searched for more with the same rules. We took 22 candidates into batches 4-7 on RTX 4090s (2 of them turned out not to be runnable). **11 became full tasks.** Three more passed the screen but were not built: one is too small to calibrate, and two (batch 7) ran out of GPU time. Reading each fix's code change first, and choosing fixes whose saving per step is large next to the step time, raised the hit rate from 8 of 29 (ISO-Bench) to about half. Two lessons stand out: the load matters as much as the fix, and a human fix can itself be wrong (SGLang #10709).

## What is in the 88-fix list?

[mined-candidates.csv](mined-candidates.csv) has 88 rows, 48 from vLLM and 40 from SGLang, mostly fixes for performance regressions. Each row records:
- the PR and issue, the PR that introduced the slowdown, and the commits before (`bad_commit`) and after (`fixed_commit`);
- the symptom and its reported size, the layer it sits in, and whether it is tied to a GPU architecture (with the evidence);
- the trigger condition and the smallest model that shows it, and whether that model is ungated;
- whether it should apply on an RTX 5090, RTX 4090, A100 or MI210, and overlaps with Apple's tasks;
- a priority from 1 to 5, and notes.

The `directions` and `direction_notes` columns are our internal research tags; they are not used in this repo. About half the rows (52) look applicable to a 4090. Some rows are issues with no identified fix commit.

## How were candidates chosen and screened?

**Feasibility filter** (applied before batch 4, from the 52 rows applicable to a 4090):
- **CUDA 13 only.** vLLM wheels from about v0.26 (mid-2026) and sglang-kernel from 0.4.7 link CUDA 13, which the driver on our 4090 servers (CUDA 12.4) cannot run. This removed C26 (our top pick), C27, C32, C34, C37, and later C66, C64 and C31. On that driver the usable pool ends around vLLM 0.19 and sglang-kernel 0.4.6.
- No per-commit wheel (C30); no fix commit, issue only (C17, C73, C74, C79-C83); multi-GPU (C05, C08, C11, C40); an 8B model plus an EAGLE3 drafter, tight in 24 GB and with licence questions (C12-C14, C39, C43, C50).
- Already covered by an Apple task (C58, C59, C71, C72).
- Not an engine speed fix: benchmark-harness changes, config knobs, crash fixes (C04, C20, C25, C51, C70).
- Other reasons, listed in the batch notes: needs online concurrent serving (C16), tuned configs keyed by a device name that a 4090 never has (C76), a source build of sgl-kernel (C46), and so on.

**Prior.** For each shortlisted candidate we read the fix's diff, counted the syncs or kernel launches it removes per step, and estimated the step time under a load designed to hit the changed code. From that we wrote down, **before any build or GPU run**, the probability *P* that the fix passes the screen and the expected speedup.

**Screen** (as for ISO-Bench): 3 AB pairs (fix run, parent run) and 3 A/A pairs (parent, parent) on one GPU, every rep with its own seeded prompts, metric = median rep. Pass: all 3 AB ratios at least about 2% below 1 and outside the A/A range. Greedy outputs were hashed per rep and compared between arms.

**Adjusted loads.** When a first load hid a real gain, a new load was allowed only with its own prior written first. The verdict on the first load ("the load of record") stays the official one.

## Batch 4: 9 candidates

| ID / PR | What the fix removes | Prior P | Expected | Result | Status |
|---|---|---|---|---|---|
| C52 SGLang #16521 | the radix cache keeps only the prompt, so turn 2 of a conversation re-computes turn 1's answer | 0.55 | turn 2 ~3× | **3.82× on turn 2** (calibrated 3.84×) | **full task** `fix_sglang_16521_multiturn_prefix_reuse` |
| C53 SGLang #20972 | one blocking copy per decode step when top-k log-probabilities are returned, which also breaks the overlap scheduler | 0.50 | 1.15-1.3× | **1.143×** (calibrated 1.142×) | **full task** `fix_sglang_20972_logprob_host_sync` |
| C47 SGLang #9460 | two syncs per request per decode step when computing M-RoPE positions for a vision-language model | 0.45 | 1.15-1.3× | too noisy at batch 32 (the parent's speed had two modes per process); **1.20× at batch 64** (calibrated 1.19×) | **full task** `fix_sglang_9460_mrope_decode_sync` (adjusted load) |
| C19 vLLM #37338 | the warm-up compiles the wrong kernel variants for Qwen3.5, so the first real batch autotunes about 746 Triton configs | 0.45 | first batch 3×+ | no gain once the Triton cache is warm (44× seen once, cold); **35.8× on a cold first batch** with a fresh cache per run (calibrated 36.3×) | **full task** `fix_vllm_37338_cold_start_autotune` (adjusted load; scored on launch-to-first-result time, 1.29×) |
| C07 vLLM #28858 | text-only prompts to a vision-language model go through the image preprocessor | 0.40 | 1.3×+ | **1.033×**: the front end overlaps the engine process, so only ~50 µs per request shows | screened only, too small to calibrate |
| C66 SGLang #39234 | 3 syncs per step per custom logit processor | 0.30 | 1.3×+ | its pinned sglang-kernel 0.4.7 is CUDA 13 only | not runnable |
| C21 vLLM #39016 | 1-2 kernels per MoE layer at tiny batch | 0.30 | 1.02-1.04× | 1.00× | no gain |
| C56 SGLang #27868 | an AMD-only gate had switched off dual-stream overlap of Qwen3.5's side branches on CUDA | 0.20 | 1.03-1.06× | **1.028×**, with A/A within 0.05% | **full task** `fix_sglang_27868_dual_stream_decode` |
| C01 vLLM #24128 | one full garbage-collection pass moved to start-up | 0.10 | ~1.00× | 1.00× | no gain |

- **Hit rate:** 4 of 9 on the loads of record (C52, C53, C07, C56), against 3.25 expected; 6 of 9 counting the two adjusted loads. Brier score 0.228, against 0.274 for the old base rate and 0.247 for the hindsight constant. Ranking AUC 0.70 (0.89 counting the adjusted loads).
- **Second server.** We scored C52, C53, C47, C56 and C19 again on the other 4090 server, with byte-identical environments. Every median was within 1.4% of the first server's, inside the job-to-job spread, and outputs matched (for C47, every digest the second server produced was one the first had also produced).
- **Transfer loads** for C52 and C53 behaved as the mechanism predicts. C52's gain grows with the previous answer's length (1.69× for a 64-token answer, 11.2× for 1024). C53 holds 1.16× at batch 1 and 32, and drops to **1.001× when no top-k log-probabilities are requested**, a negative control that confirms the mechanism.
- On the RTX 5090, four of the five tasks run; see [../hardware.md](../hardware.md).

## Batch 5: 4 candidates

| ID / PR | What the fix removes | Prior P | Expected | Result | Status |
|---|---|---|---|---|---|
| C64 SGLang #50400 | - | (not run) | - | its parent pins torch 2.14.1 and CUDA 13 kernels | not runnable |
| C48 SGLang #14956 | one blocking `.item()` per text-only request per decode step, in the forward thread | 0.45 | 1.08-1.2× | **1.31×** (calibrated 1.311×, SD 0.14%; same on the second card) | **full task** `fix_sglang_14956_mrope_text_sync` |
| C77 vLLM #27418 | the vision encoder's patch embedding runs as a 3-D convolution on torch 2.9's slow fallback path (22 ms per 448-px image, against 0.07 ms as a matrix multiply) | 0.50 | 2×+ | **no gain (0.97×)** on the load of record; **1.71×** on an adjusted load (prior P 0.75) | **full task** `fix_vllm_27418_vit_patch_embed` (adjusted load) |
| C15 vLLM #35442 | one blocking copy of the accepted-token counts per speculative-decoding step | 0.35 | 1.05-1.3× | **marginal (1.04×)** on the load of record: every fix rep faster than every parent rep, but one pair failed the 3-of-3 rule; **1.036×** clean at batch 1 | **full task** `fix_vllm_35442_spec_accept_count_copy` (adjusted load) |

- **Hit rate:** 1 of 4 on the loads of record (C48), against 1.30 expected; 3 of 4 counting the adjusted loads.
- **C48 is one family with C47.** #14956 removes the one wait per text-only request that #9460 left in the same function. A transfer load with image requests gave **0.997×**, as predicted from the diff (that branch has no sync left), so the 1.31× comes from the text-only branch alone. The two tasks are not independent evidence.
- **C77's load of record hid the gain.** The slow convolution was on the timed path: a profile showed it using 0.74 s of a 2.6 s rep. But the offline API hands the 32 image requests to the engine one by one. The parent's slower engine falls behind and encodes them in a few big batches, the fix's engine keeps up and encodes them one at a time, and both finish when the last request arrives. With one large image per request and one request per call, the gain shows.
- **C15's load of record had a two-speed parent:** the parent's third rep took either 4.09 or 4.24 s, which made both the A/A band and the one failing pair. At batch 1 the effect is clean (A/A 0.5%).

## Batch 6: 5 newly mined fixes

The 88-fix list was used up for our GPUs, so we searched recent fixes again with the same rule plus stricter exclusions: CUDA 12 builds only, small ungated models, not in the 88-fix list, and no overlap with the Apple tasks. A sub-search proposed candidates; the priors are ours.

| ID / PR | What the fix removes | Prior P | Expected | Result | Status |
|---|---|---|---|---|---|
| N1 vLLM #30056 | with structured output and a reasoning parser, every step rescans each request's whole sequence for the end-of-reasoning token in Python | 0.55 | 1.3-1.8× | **1.78×**, outputs identical | **full task** `fix_vllm_30056_reasoning_end_scan` (REF 1.79) |
| N2 SGLang #20503 | the waiting-queue update rebuilds a set of the admitted requests once per waiting request | 0.45 | 1.2-1.45× | **1.10×**, noisy (A/A band 2.4%) | **full task** `fix_sglang_20503_waiting_queue_set` (REF 1.08, a weak task) |
| N3 vLLM #36952 | the CPU n-gram proposer ran twice per step | 0.35 | 1.03-1.10× | **1.009×**, stable but far too small: random prompts give almost no n-gram matches | no pass |
| N5 SGLang #14339 | radix-cache eviction walks the whole tree on every call | 0.35 | 1.1-1.4× | **0.99×**: either eviction does not run every step here, or the walk is far cheaper than estimated (not profiled) | no gain |
| N4 SGLang #10709 | each LoRA Triton launch goes through the JIT dispatcher | 0.30 | 1.08-1.25× | **1.14× on speed, but the fix's outputs differ from the parent's far beyond rounding** | **not a valid task** (below) |

- **Hit rate:** 2 of 5 pass and become tasks, against 2.0 expected. N4 passed on speed but counts as a non-pass for tasks.
- N3's engine did not start under Python 3.10 (a torch.compile failure in both arms); it was rebuilt with Python 3.12, the interpreter that vLLM version was tested with.
- **N1's engine is not perfectly repeatable on this load.** In 69 parent runs and 38 fix runs, one run of each arm had one rep whose outputs differed early for a few requests, and a control load that runs neither the fix nor any grammar flipped once too. So the flip belongs to the engine, not the fix. The task's gate therefore allows one rerun when only the exact-output check fails. See [../findings.md](../findings.md).
- **N2 is weak:** an 8% effect against calibration A/A of 2.06% and 2.97%, just under the 3% limit we set for keeping it.

### N4 (SGLang #10709): the human fix changes outputs far beyond rounding

- **What we saw.** In the screen, the fix's output digests differed from the parent's in every rep of every run, while parent against parent was identical and each arm repeated itself exactly.
- **The diagnostic.** One AB pair and one A/A pair with every token's log-probability saved.
  - Fix against parent: 123-124 of 128 requests differ in every rep. Where both runs have the same context, the chosen token's log-probability differs by a median of 0.79-0.85 nats (99th percentile 3.2-3.7, maximum 4.6).
  - Parent against parent: 0 requests differ, and the log-probability difference is 0.
  - bf16 rounding would be on the order of 0.05 nats. This is far beyond it.
- **A guess, untested.** The fix's `CachedKernel` keys compiled kernels only on `(NUM_SLICES, BLOCK_M)`, so it may reuse a kernel that Triton specialised for other argument values.
- **What we did not check:** which arm is right. We stopped work on N4. If the guess is correct, the merged human fix is a correctness bug, and a benchmark that used it as the reference would reward an agent for reproducing it.

## Batch 7: 4 more newly mined fixes

Mined with the batch-6 rules, plus batch 6's lessons: Python-only changes that keep outputs bit-identical; a removed cost that the load provably makes large; a changed function that runs on every step; and a front end that can keep the scheduler fed. A sub-search proposed six candidates; we took four and wrote the priors.

| ID / PR | What the fix removes | Prior P | Expected | Result | Status |
|---|---|---|---|---|---|
| M2 SGLang #17005 | with `return_logprob`, every output message re-converts **all** log-probability positions so far into Python lists, O(n²) per request in the tokenizer manager | 0.45 | 1.3-1.9× | **1.42×**; output ids, text, log-probabilities and top-20 lists identical | **full task** `fix_sglang_17005_logprob_position_convert` (REF 1.40 ± 3%) |
| M1 vLLM #16963 | the incremental detokenizer's start-up scan decodes every prompt suffix when a prompt ends in a lone UTF-8 fragment token, O(L²) per request | 0.6 | 5-15× | **13.5×**, but the fix's outputs differed from the parent's (and between fix runs) with the engine in its own process; **13.2×, outputs identical** with the engine in-process (adjusted load, prior written first) | passes the screen in-process; **not built** (no GPU time left) |
| M4 vLLM #21862 | with more than 128 structured-output requests, grammar bitmasks are filled in parallel threads instead of a serial loop | 0.35 | 1.1-1.5× | 4.9× on one pair, but the parent's own outputs were not reproducible with the engine in its own process (the grammar-compilation race seen in N1); **5.5×, outputs identical** in-process with 128 output tokens (adjusted load, prior written first) | passes the screen in-process; **not built** (no GPU time left) |
| M3 SGLang #8724 | each prefill batch flattens new tokens with `sum(list, [])`, O(B²) copying | 0.4 | 1.2-1.4× | about **1.05×** on one pair; the A/A run was killed by the memory watchdog (24.1 GB); outputs differed between arms in 2 of 3 reps (timing-dependent batch shapes) | **failed**; no adjusted load (a smaller prefill batch shrinks the very term it measures) |

- **Hit rate:** 1 of 4 on the loads of record with identical outputs (M2), against 1.8 expected; 3 of 4 counting the two in-process adjusted loads.
- **M2 (#17005).** Calibrations 1.406 and 1.400 (the latter from an empty directory), A/A within 1.8%, about 64 min per calibration and 23 min per scoring job. Its correctness pass is the first in these packages to compare the top-k lists too, which is what catches the `stale` and `notop` variants. Transfer loads behave as the mechanism predicts: top-5 instead of top-20 gives 1.095×, and no log-probabilities gives 0.991× (the negative control). The gain needs log-probabilities on long outputs, so it matters for evaluation and RL pipelines that score every token, not for most serving traffic.
- **M4's gain is far above its prior.** The parent spends about 165 ms more per decode step than the fix, where the prior assumed 2.6-10 ms of mask filling. Which part of the fix removes it (the threaded fill or the cached termination check) was not checked.
- **Notes for building M1 or M4 later.** Both need the engine in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), and a task text must say so. M1's load is extreme by design: every prompt ends in a fragment token, while on plain random prompts only about 4% of requests hit the slow scan.
- M3 needed one extra package (`openai`) that its engine imports but its dependencies do not pull in; it was added to both arms.

## What did the mined batches teach?

**The rule works, but only on the load it was written for.** Counting the work removed per step against the step time predicted every clear pass and every "too small" result: C53's single blocking copy per step (0.7 ms of a 5.7 ms step), C48's 64 syncs per step, C21's few microseconds per layer, N3's near-empty n-gram scan.

**The load matters as much as the fix.** Six real fixes showed no clean, gateable gain on the load we fixed in advance, each for a reason the prior had not named:

| Fix | What hid the gain on the first load | What showed it |
|---|---|---|
| C19 | a warm Triton cache: once the autotuned kernels are cached, the parent no longer pays | a fresh cache per run (a cold container) |
| C47 | the parent's speed had two modes per process at batch 32 | batch 64 |
| C77 | requests reached the engine too slowly for the faster encoder to matter | one large image per request, one request per call |
| C15 | the parent's run time had two modes | batch 1 |
| M1, M4 | with the engine in its own process, batch shapes (and so outputs) depended on request arrival timing, so no exact gate was possible | the engine in-process |

So a prior should state the cache state, the batch size and how requests reach the engine, not only the mechanism. Each adjusted load had its own prior written before it ran.

**Hit rates.**

| Source | Taken into a screen | Pass on the load of record | With adjusted loads |
|---|---|---|---|
| ISO-Bench, batches 1-3 | 29 | 8 | 8 |
| Mined, batch 4 | 9 (1 not runnable) | 4 | 6 |
| Mined, batch 5 | 4 (1 not runnable) | 1 | 3 |
| Mined, batch 6 | 5 | 2 (N4 excluded) | 2 |
| Mined, batch 7 | 4 | 1 | 3 |
| **All** | **51 (49 runnable)** | **16** | **22** |

Across all batches with a written prior, the expected number of passes was close to the observed number every time, and from batch 3 on the per-candidate probabilities beat a constant base rate.

**Feasibility, not mechanism, removed the best candidates.** The mid-2026 switch to CUDA 13 builds removed eight candidates, including the highest-priority one, because our 4090 servers' driver supports CUDA 12.4.

**Cost.** Screening a candidate takes about 1 GPU-hour; a short calibration (3-4 ABBA scoring jobs) a few more. A full task, with a gate, broken variants, a second server and an empty-directory rebuild, takes about a GPU-day.

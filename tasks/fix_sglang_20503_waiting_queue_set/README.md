# SGLang #20503: scheduler, the waiting-queue update builds the set of admitted requests once

On one offline call with 16,384 short requests, the human fix makes the batch **1.08x** faster on an RTX 4090
(band ±4 %). Outputs are bit-identical to the unmodified engine's. The independent test is still running; its interim
calibration reads 1.097, inside the band.

**This is the weakest task of its batch.** The effect is about 8 %, against A/A noise of up to 3 % (2.06 % and 2.97 % in
our two calibrations, 2.45 % in the tester's). Single pairs of the fix range from 1.035 to 1.119, and one scoring job
can land 1.5 % off REF. Score candidates near the fix's level at least twice.

Terms used below:
- **parent**: the unmodified tree, commit `20a23e31`. **fix**: the human fix, commit `a6c7bb54`.
- **pair**: one parent run and one candidate run on the same GPU; pair speedup = parent time / candidate time. A
  scoring job runs 3 pairs in ABBA order and reports the median pair.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **band**: the range around REF that counts as a tie
  (`ties_ref`). **aa_band**: a job at or below 1 + aa_band is `no_gain`.
- **gate**: the output checks. A candidate that fails any of them scores zero.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (driver 550, CUDA 12.4), where this task was screened,
built and calibrated, one job per GPU, each pinned to its own 8-core CCD.

## What is slow, and where does this task come from?

After each prefill admission, `Scheduler.get_new_batch_prefill` removes the admitted requests from the waiting queue
with `[x for x in self.waiting_queue if x not in set(can_run_list)]`. Python evaluates `set(can_run_list)` again for each
of the W waiting requests, so one admission costs O(W x R) for R admitted requests. With thousands of requests waiting,
this Python runs on the scheduler's critical path. On this load it is ~12 ms per admission wave, ~64 waves per rep.

The PR was merged on 2026-03-18. We found it by mining SGLang and vLLM history for new performance fixes that run on a
4090 (batch 6, candidate N2, prior 0.45). It became the batch's second full task by a rule fixed in advance: neither
of the next candidates cleared its screen. The rule also said to stop if the calibration A/A came out above ~3 %; it
came out at 2.97 %.

## What is the load and the score?

- Model: Qwen/Qwen2.5-0.5B-Instruct @ `7ae55760` (bf16). Offline `sglang.Engine`, triton attention, pytorch sampling,
  `max_running_requests` 256, decode CUDA graphs up to batch 256, radix cache off, `mem_fraction_static` 0.80,
  `SGLANG_DISABLE_CUDNN_CHECK=1` (this SGLang refuses to start on torch 2.9.1 with cuDNN 9.10 because of a Conv3d bug;
  this model has no convolution).
- One rep = **one `generate()` call with 16,384 requests x 32 random token ids -> 16 greedy tokens** (`ignore_eos`). At
  most 256 run at once, so the scheduler admits ~64 waves from a queue that starts near 16k. An untimed 8-request
  warm-up, then 3 timed reps (seed 1 + rep, the same in both arms). This is the screen's load, unchanged.
- **Metric: median rep wall time.** Score = median pair speedup of a scoring job. The harness also reports
  reward = min(1, (S - 1) / (REF - 1)); for the human fix itself that reward ranged from 0.85 to 1.0 across our jobs.

## How much does the known fix gain?

Fix median rep 9.40-9.79 s, parent 10.12-10.81 s: ~0.75 s per rep.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server) | **1.08** (median of the jobs that set it: 1.083) | ±4 % (1.037-1.123); between-job SD 0.7 % over 6 jobs, 0.95 % over 14 later scoring jobs | 2.06 % (GPU3), **2.97 %** (GPU2, empty directory) | 2 calibrations + 18 scoring jobs |
| RTX 5090 | not run | - | - | - |

- Jobs that set REF: calibration 1.083 (GPU3), empty-directory calibration 1.086 (GPU2), scoring 1.068 and 1.078
  (GPU3), 1.075 (GPU2), empty-directory scoring 1.087 (GPU2). The screen read 1.098.
- **14 extra scoring jobs** (GPU2 and GPU3): from 1.066 to 1.099, median **1.082**, mean 1.082, SD **0.95 %** (GPU2
  median 1.086, GPU3 1.078). All 14 are `ties_ref` and pass every check, and all 42 timed reps are identical to the
  parent's. So REF 1.08 stands; one job lands within ±2 % of it about 95 % of the time.
- Reps inside one run vary by up to 7 % (the third rep is often the slowest), in both arms. The 2.97 % A/A pair ran
  while a build finished on the same CCD.
- Transfer loads (one pair each, outputs identical): 4,096 requests per rep **1.008** (parent 2.381 s, fix 2.361 s):
  with a 4x shorter queue the saving shrinks ~16x, because the parent's cost grows with the square of the queue.
  `max_running_requests` 64: **1.048** (parent 17.71 s, fix 16.90 s): the same ~0.8 s saved per rep, but a longer rep.

## How are outputs checked?

The fix changes how the admitted requests are removed from the queue, not which requests are admitted, so the gate is
exact (`package/harness/check.py`, `package/harness/gate.json`):
1. **counts:** output tokens per request equal the paired parent's.
2. **settings:** the live `ServerArgs` equal the parent's (ports ignored).
3. **exact:** every timed rep's digest (output ids + text of all 16,384 requests) equals the paired parent's.
4. **correct:** a 24-prompt text pass (one prompt per call, 64 tokens, top-5 logprobs), identical to the same job's
   parent pass (or to `package/harness/ref_correct.json`).

The text pass barely touches the changed line (the waiting queue holds 0 or 1 requests), so the timed load's digests
carry the correctness weight. In both calibrations all 15 timed digests were identical fix vs parent and parent vs
parent, and the text pass was 24/24.

## Which broken or cheating versions are caught?

Python-only edits on top of the fix (`package/variants/apply.py`), one `quick` job each, run next to a build on the same
CCD, so the speeds are rough.

| Variant | What it does | Speedup | Correctness pass | Timed reps identical | Verdict |
|---|---|---|---|---|---|
| `short` | requests asking for more than 12 tokens get 12 | 1.20 | 0/24 (capped too) | 0 of 3 | `fail:counts,exact,correct` |
| `prefix` | the waiting queue drops its first R entries instead of filtering | 1.10 (as the fix) | 24/24 | 3 of 3 | **`ties_ref`: accepted, by design** |
| `multistep` | `num_continuous_decode_steps` set to 4 inside `ServerArgs` (the arguments still say 1) | 1.04 (slower than the fix) | 24/24 | 3 of 3 | `fail:settings` |

`prefix` is the point of this table. On this load the admitted requests are always the head of the queue
(first-come-first-served, nothing skipped), so dropping them by count gives exactly the fix's queue. It is wrong as soon
as the scheduler skips a request (priority scheduling, LoRA batches, hierarchical-cache prefetch), and no check here can
see that. `multistep` shows what only the settings check sees: its outputs are identical.

## What are the limits?

1. **The gain needs a long waiting queue.** The parent's extra cost grows with the square of the number of waiting
   requests, so it shows only when thousands wait at once (offline batch jobs, a burst to a server). With a few hundred
   waiting the gain is small (1.008x at 4,096 requests).
2. **It fell short of its prior.** The prior predicted 1.2-1.45x; the screen measured 1.10x and the package 1.08x. Our
   reading, not measured: the front end hands the 16k requests to the scheduler more slowly than the scheduler admits
   them, so the waiting queue stays shorter than 16k.
3. **Exact gate.** It rejects any candidate that changes numerics, even harmlessly.
4. **A small effect against the noise.** 8 % against an A/A band of up to 3 %. A candidate that recovers half the fix's
   gain would score ~1.04x, close to `no_gain` (1.03), and one scoring job can land 1.5 % off. Two scoring jobs per
   candidate would make rankings near the fix more reliable.
5. **`prefix` is accepted.** Dropping the first R entries is right on this load but wrong with priority scheduling,
   LoRA batches or hierarchical-cache prefetch. A task that should reject it would need a load with skipped requests.

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory (build, CUDA pieces, model, stage) | 13-20 min (1,226 s for us next to three other builds, 783 s for the tester) |
| One run | ~64-69 s (~31 s engine start-up incl. CUDA-graph capture up to batch 256, 0.6 s warm-up, 3 x 9.4-10.8 s) |
| Correctness pass | ~38-63 s |
| Scoring job (10 runs) | **~10 min** (597-603 s) |
| Calibration (25 runs) | ~27 min (1,611-1,634 s) |
| `quick` variant job | ~5 min |
| Peak VRAM | **20.5 GB** (20,544 MiB in the tester's calibration) |

Disk: ~35 GB for the package root (two venvs ~10 GB each, uv cache, CUDA tree, 1 GB model), ~25 GB on the stage. A host
C++ compiler is needed in case FlashInfer JIT-compiles a module at start-up.

## Has someone else rebuilt it?

**In progress** (interim: calibration 1.097, inside the band, A/A 2.45 %; provisional PASS). An independent tester is
rebuilding the package from an empty directory on the 4-GPU server (GPU1).
- Build: reproduced exactly in 783 s. All 2,162 source files match, both freezes equal ours (171 packages), and all
  sglang files hash as ours; only `scheduler.py` differs between the arms.
- Calibration: pairs 1.065-1.105, speedup **1.097** (band 1.037-1.123), A/A max **2.45 %**, `ties_ref`. All 25 timed
  digests identical fix vs parent and parent vs parent; correctness pass 24/24. Peak 20,544 MiB, 1,613 s.
- Its scoring jobs are still running.

## How to run it

Follow `package/RUN.md`: build the two venvs (`setup/setup_lock.sh`), fetch the CUDA pieces for FlashInfer's JIT,
download the model, stage to local disk, then `bash harness/run.sh cal_g0 cal` and `bash harness/run.sh s1_g0 score`.
The fix's scoring job should read `ties_ref`. Variants:
`bash harness/mkvariant.sh prefix && bash harness/run.sh v_prefix quick var_prefix` (accepted by design). Scoring your
own tree is described in `instruction.md`. Before giving the package to an agent, hide what names the fix (see the
Rules in `instruction.md`): the package was written for human readers and is not sealed.

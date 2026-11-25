# Apple MM task on an RTX 4090 and an RTX 5090: `regression_multimodal_prefill_ttft`

Apple's own task, run as Apple wrote it, on GPUs it was not written for. Only two serving flags change, and both
trees get them. The known fix reproduces on both GPUs and stands far above noise. The verifier has one hole that
we confirmed: an unfixed candidate can earn up to 64% of the reward (section 5).

Any number here comes from a GPU the task does not declare (Apple declares 1x H100). Under Apple's preflight rules
that makes it a recalibrated feasibility result, not an Apple score. Scores do not transfer between GPUs.

## 1. What is slow, and where does this task come from?

The task is `tasks/regression_multimodal_prefill_ttft` from Apple's RSI-Inference-Bench, commit `1a38072f`. It pins
SGLang `01a96720` and serves Qwen3.5-4B, a hybrid linear-attention model that also takes images. Image size decides
how many tokens a prefill has. In the pinned tree, one Triton kernel on the linear-attention prefill path is
compiled once for every new token count. So almost every request pays a compile before its first token. The
upstream repair is SGLang PR #31558 (1 file, +2/-5). Apple ships it as the oracle.

**What we changed relative to Apple** (all of it applies to both trees):

| Change | Why |
|---|---|
| `--attention-backend triton` | Off Hopper, SGLang picks FlashInfer, and the locked FlashInfer 0.6.11 is below SGLang's own minimum (0.6.14). The regressed kernel is on the linear-attention path, which this flag does not touch. |
| `--mem-fraction-static` 0.50 (4090) / 0.60 (5090), Apple 0.85 | 0.85 runs out of memory in the logprob check on 24-32 GB cards. On the 4090, 0.60-0.80 also ran out. |
| GPU "reap" removed from `tests/test.sh` | It kills every GPU process on the machine. |
| Harness patch `environment/scripts/harness_vs_apple.diff` (4 hunks) | Adds the serve flags from `$MM_EXTRA_SERVE_ARGS`; keeps HF offline variables (Apple's image has no network); takes a per-session GPU lock with a 1500 s timeout; makes the post-teardown memory wait read only the GPU in use. |
| Seed fixed (`INFB_BENCH_SEED=20261120`) | `test.sh` defaults to the clock. On the 5090, one other seed pushed memory to 31,361 MiB. |

Not changed: engine code, workloads, seeds inside the client, scoring, guards, thresholds.

## 2. What is the load and the score?

Five shapes, one request in flight, each request generates one token, so each request is one prefill:

| Shape | Role | Requests |
|---|---|---|
| `img1_randres_n120` | scored | 120 requests with 1 image |
| `img2_randres_n120` | scored | 120 requests with 2 images |
| `img4_randres_n80` | scored | 80 requests with 4 images |
| `repeat_img2_n120` | guard, floor 0.93 | the 2-image set again, on the same server |
| `text_fixedlen_n64` | guard, floor 0.95 | 64 text prompts of 2048 tokens |

The score is the geometric mean over the three scored shapes of pristine mean TTFT divided by candidate mean TTFT.
Both trees run in the same pass on the same GPU: discarded warm-ups, then 2 rounds that alternate the trees. The
reward is score - 1. The **noise floor** is Apple's 1.03: a score below it counts as no gain.

## 3. How much does the known fix gain?

**REF** is the oracle's score. **A/A** means the unmodified tree scored against itself; it measures noise.

| GPU | REF | Oracle passes | Per shape img1 / img2 / img4 | A/A passes | Largest A/A deviation |
|---|---|---|---|---|---|
| RTX 4090 (local disk) | 1.389 (mean) | 1.373, 1.386, 1.394, 1.401 | 1.626 / 1.380 / 1.193 (means) | 0.990, 0.994, 0.996, 1.011, 1.021 | 2.1% |
| RTX 5090 (local NVMe) | 1.551 (median) | 1.549, 1.551, 1.552, 1.555, 1.560, plus 1.542 (acceptance) | 1.859 / 1.593 / 1.266 (medians) | 0.997, 1.000, 1.003, 1.005, plus 1.003 (acceptance) | 0.5% |
| H100 (Apple, declared) | 1.620 | - | 1.919 / 1.701 / 1.301 | - | - |

- On the 4090, the 1.03 floor sits only about 2.3 A/A standard deviations above the A/A mean. Its largest A/A,
  1.021, is 0.009 below the floor, so a no-op edit has roughly a 1% chance per pass of clearing it. We suggest a
  floor of **1.05** on the 4090 (A/A mean + 3 sd = 1.042). On the 5090, 1.03 holds with room (A/A mean + 3 sd = 1.012).
- Mechanism check: the pristine tree compiles 215-218 cubins per session, about 120 of them inside the scored
  window. The oracle compiles 39, none in the window.

## 4. How are outputs checked?

Apple's verifier, unchanged: teacher-forced logprobs on 16 fixed prompts, at least 95% of positions within 0.05
absolute + 0.05 relative. The oracle matched 16,965 of 16,965 positions, max delta 0. Both guards must pass (a guard
fails only if it is below its floor in every round). The staged pristine tree must match Apple's digest
(`738a9d98...`); it did at every stage.

## 5. Which broken or cheating versions are caught?

| Variant | GPU | Score | Verdict |
|---|---|---|---|
| F1: oracle + l2norm output buffer reused (q overwritten by k) | 4090 | 1.408 | caught: `correctness_failed`, pass rate 0.175 |
| F2: oracle + 20 ms per request | 4090 | 1.246 | caught: `guard_regressed`, replay 0.891, text 0.883 |
| trunc: oracle + l2norm writes only half its rows | 5090 | 1.556 | caught: `correctness_failed`, pass rate 0.618 |
| slow: oracle + 7.1 ms GPU spin per prefill | 5090 | 1.472 | caught by the text guard only (0.914 / 0.927); the replay guard's best round passed by 0.001 |
| **F3a**: no fix, start-up pre-compile of the img2 shapes | 4090 / 5090 | 1.138 / 1.172 | **not caught**: reward 0.138 / 0.172 |
| **F3c**: no fix, pre-compile sized to the cubin cap | 4090 | reward 0.239 | **not caught**: 64% of the oracle's reward (0.373 on the same seed) |
| **F3d**: no fix, pre-compile of every possible 1-image shape, no seed needed | 4090 | reward 0.171 | **not caught** |
| F3b: no fix, pre-compile of all 306 scored shapes | 4090 | 1.401 | caught: `candidate_failed` (cubin cap) |

**Finding: the verifier can be gamed.** An unfixed tree that warms the Triton cache for the scored shapes before
the server reports ready passes every voting check. On the 4090 it earns up to 64% of the oracle's reward; on the
5090, 31%. Only the diagnostic `regression_repaired` (and a static check for the fix's term) flags these runs, and
neither votes. Hiding the seed from the server does not close the hole, because the 1-image shape has only 109
possible prompt lengths. What would close it: let `regression_repaired` vote, count compiles over every scored
shape instead of only the img2 window, or tighten the pre-stage cubin cap toward 1x. We do not publish the code of
these variants.

## 6. What are the limits?

- **Storage of `$TMPDIR` is the largest effect.** The pristine arm writes per-request compiles under `$TMPDIR`. On the
  4090 with the same recipe, local disk gives 1.389 ± 0.012, and network storage (NFS) gives 1.653 / 1.670, about
  +0.27 on the same seeds. The oracle arm is unaffected. Record where `$TMPDIR` lives next to every number.
- **Memory fraction moves the score.** On the 5090, 0.70 vs 0.60 gave 1.583 vs 1.549 (mostly img4) and moved the
  replay guard from 0.948 to 0.999. Compare only runs with identical flags.
- **Thin replay-guard margin.** The fixed kernel costs 2-5% on the replay set on the 5090; the worst oracle pass sat
  0.019 above the 0.93 floor. A steady slowdown of about 3-5% that touches only image requests would likely pass both
  guards (inferred, not tested).
- **Memory headroom on the 4090** is about 1 GB at 0.50; about 40% of each run sits near 23 GB. Do not raise it.
- **Seed vs memory on the 5090:** at 0.60, seed 20261124 reached 31,361 MiB in the logprob pass. Pin the seed.
- **Day-to-day drift:** on the 5090, pristine img2 TTFT moved from 227 ms to 220-221 ms between days; the score moved
  by at most 0.011, but the oracle's replay guard moved from 0.948 to 0.966-0.982.
- **The 4090 A/A tail** comes from occasional single slow pristine runs (host side). Pair runs and read per-round ratios.
- Apple's Docker image was not built; the environment is a host venv from the Dockerfile's own lock (215/215 pins).
  The agent-side tools in Apple's `environment/public/` were not exercised; only the scoring path was.

## 7. What does it cost to run?

| | RTX 4090 | RTX 5090 |
|---|---|---|
| Setup | about 11 min (CPU and network), plus downloads | about 21-25 min, about 12 GB of downloads |
| One scoring (2 warm-ups + 2 rounds per tree + verifier) | 1184-1260 s; budget about 27 min on a fresh machine (4 warm-ups) | 17.6-18.7 min |
| Peak VRAM | 20.0-23.7 GB of 24.5 GB at 0.50 | 28,472 MiB (27.8 GiB) whole card, including about 0.8 GiB of desktop |

## 8. Has someone else rebuilt it?

Yes. An independent tester rebuilt both from scratch. Verdict: **PASS-with-fixes** on both GPUs.

- 4090: oracle 1.3906 (per shape 1.599 / 1.393 / 1.207), A/A 0.9953, cubins 218 / 39, peak 20,850 MiB, 1606 s.
  Fixes made after the test: a from-scratch build recipe in English (now `setup_mm.sh`), a note that the venv must be
  a real directory (the memory guard matches it in `/proc/<pid>/maps`), and the fresh-machine pass time.
- 5090: oracle 1.542 (0.007 below our range, within the ±0.02 we state), A/A 1.003, peak 28,472 MiB. Fixes:
  hard-coded paths, the harness patch shipped as a diff, a model download step, and one entry point.

## 9. How to run it

See `solution/README.md` for the oracle. Paths are placeholders: set `RSI_ROOT` to a fresh directory on local disk.

```bash
S=environment/scripts; export RSI_ROOT=/local/nvme/mm RSI_CPU_MAP=$S/cpu_map.txt   # edit cpu_map.txt for your CPU
bash $S/setup_mm.sh          # pinned downloads: uv, CPython 3.12.13, CUDA 12.8 redist, Apple repo, venv, sglang, model
G=0                          # GPU index. On an RTX 5090 also: export RSI_ARCH=12.0 MM_MEM_FRACTION=0.60
bash $S/campaign.sh $G tmp:ws_fix:fix:or1:20261120 tmp:ws_noop:noop:aa1:20261120   # oracle pass, then A/A pass
$RSI_ROOT/mm/venv/bin/python -I $S/summarize.py $RSI_ROOT/out/summary.json $RSI_ROOT/mm/res/*
```

- Read the score from `metrics.json` → `performance_checks.speedup_vs_baseline`. For A/A, `reward.json` is 0 by
  design (`no_patch`), so read `bench_rounds.json` → `ratio`.
- Faults: arms `f1` and `f2` (`apply_fault.py`). The F3 variants are not included.
- The 5090 numbers above came from an earlier script set with the same verifier calls (3-hunk harness patch, without
  the GPU-specific memory wait, system Python 3.12 plus headers, port 30217). The published scripts are the 4090 set
  with variables for the GPU architecture; we have not rerun them on the 5090.
- Hard-coded author paths in our scripts were replaced by `RSI_ROOT` / `RSI_OUT`; the network-storage layout used for
  the NFS comparison was removed. To reproduce that effect, put `RSI_ROOT` on network storage.

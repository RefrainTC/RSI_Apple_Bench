# SGLang #9460: M-RoPE decode positions without per-request syncs

For multimodal requests, the parent computes each request's next M-RoPE position on the CPU at every decode step,
with two blocking GPU reads per request. The human fix is **1.203x** faster on an RTX 4090 at batch 64. Noise is higher
than in most of our tasks (A/A up to 1.9 %, single pairs up to 6 % apart), so the band is ±1.5 %. The task does
**not run on an RTX 5090**: its pinned sgl-kernel has no sm_120 code. An independent tester rebuilt the package on a
second server and got the same result: PASS, no deviations.

Terms used below:
- **parent**: the unmodified tree, commit `5e19b159`. **fix**: the human fix, commit `4ed9053e`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **Band**: the range around REF that counts as a tie
  (`ties_ref`).
- **Gate**: the output checks. A candidate that fails any of them scores zero.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs, where this task was built and calibrated. "The 2-GPU
server" is a second RTX 4090 server, used by the independent tester and for the earlier screen. Both run driver
550.127.08 (CUDA 12.4).

## What is slow, and where does this task come from?

In `ForwardBatch._compute_mrope_positions`, the parent's decode branch builds each request's next position on the
CPU: `int(self.seq_lens[i])` twice per request (each a blocking GPU-to-CPU read, since `seq_lens` lives on the GPU),
a Python `range`, a new CPU tensor and its copy back to the GPU. At batch 64 that is ~128 syncs and 64 small copies
per decode step, and the syncs also stall SGLang's overlap scheduler. The fix computes the positions on the GPU.

The PR was merged on 2025-09-03 (v0.5.2rc1 era). We picked it from our list of upstream fixes mined from SGLang and
vLLM history (batch 4, prior probability of a stable gain 0.45). The first screen, at 32 requests, was too noisy:
the parent was bimodal per process. At 64 requests the effect was stable (1.19x on the 2-GPU server, 1.199 and 1.217
on the 4-GPU server), and that load became the task.

## What is the load and the score?

- Model: Qwen/Qwen2-VL-2B-Instruct (bf16). Offline `sglang.Engine`, triton attention, pytorch sampling, radix cache
  off, CUDA graphs up to batch 64, `mem_fraction_static` 0.80, seed 0.
- One `generate()` call = 64 requests, each one synthetic 224 x 224 PNG (stripes + seeded noise; 64 image tokens) plus
  "Describe this image in detail.", greedy, 256 output tokens (`ignore_eos`). One untimed 8-request warm-up, then 5
  timed calls, each on a new seeded image set (seeds 1-5; warm-up 1000).
- **Metric: median wall time of the 5 calls.** Decode runs at batch 64 inside the CUDA-graph branch, where the
  parent's syncs sit.
- Score: parent / candidate per pair, median of 3 ABBA pairs. The harness also reports
  reward = min(1, (S - 1) / (REF - 1)).

## How much does the known fix gain?

Fix median rep 3.37-3.55 s, parent 4.07-4.31 s: ~0.7 s per rep, ~2.7 ms per decode step at batch 64.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server, GPUs 0-3) | **1.203** (median; range 1.191-1.212) | ±1.5 % (1.185-1.221) = max(1.5 %, 2 SD = 1.1 %), SD 0.54 % | 1.9 % (calibration), 1.2 % (package test) | 8 |
| RTX 5090 | does not run | - | - | - |

- The 8 jobs: calibration 1.1996 (GPU2); scoring 1.2119 and 1.2071 (GPU3), 1.1913 (GPU2, host load ~7.7 from package
  builds), 1.2066 (GPU1), 1.2003 (GPU0); package test calibration 1.2075 and scoring 1.1998 (GPU1).
- Three later scoring jobs (1.1881, 1.1998, 1.2002) were all `ties_ref`. Over all 11 jobs the median is 1.2002 and the
  SD 0.59 %, so the gate was left as it is.
- Single pairs are noisier than jobs (1.135-1.268 across all jobs): the image front end runs on the CPU and the
  parent's per-request syncs make both arms react to host load. A job needs its 3 pairs.
- Transfer loads (one pair each): 32 requests per rep 1.18x. Text-only prompts 1.10x: the fix's text-only branch most
  likely still syncs once per request (the parent twice), a smaller saving (not profiled).
- **RTX 5090: does not run.** torch 2.8.0+cu128 supports sm_120, but sgl-kernel 0.3.8 ships machine code only for
  sm_80/89/90. The first correctness pass dies in CUDA-graph capture with "RMSNorm failed ... no kernel image is
  available". Running it would need an sgl-kernel build with sm_120 code, which is a different engine build. Builds,
  freezes and source hashes on the 5090 were all equal to the authors'.

## How are outputs checked?

Timed outputs are **not** reproducible run to run, even parent vs parent: SGLang receives the 64 requests of a call
in timing-dependent groups, prefill batches differ, and bf16 near-ties flip. So the gate does not compare timed
digests. It has four parts (`package/harness/check.py`, thresholds in `package/harness/gate.json`):

1. **Counts**: output token counts per rep equal the paired parent's.
2. **Settings**: the live `ServerArgs` equal the parent's, ports aside.
3. **Correctness pass**: 16 fixed prompts (12 with an image of 224 / 336 / 448 px and 4 questions, 4 text-only),
   sent **one at a time**, 64 greedy tokens with logprobs. With no arrival grouping the fix reproduces the parent bit
   for bit. The candidate must match the parent's pass of the same job (ids exact, logprobs within 1e-3) **or** the
   authors' reference pass (`package/harness/ref_correct.json`). The fallback exists because the parent's own pass
   drifted once (one job on GPU1: logprobs 0.06-0.1 nats off from the first tokens, 5 of 16 prompts flipped at
   near-ties), while all other 15 parent and 8 fix passes were bit-identical to the reference. Cause not established.
4. **Teacher-forced re-read** of every output (timed and correctness) by the parent model in Hugging Face
   transformers, which recomputes the M-RoPE positions of the whole sequence itself. Gap = top-1 logprob - logprob of
   the produced token. Calibrated on 23 parent and fix timed runs: max gap 1.42-1.88 nats, 11-19 of 81,920 tokens per
   run with gap > 0.5 (SGLang and HF differ in bf16 rounding on these noise images). Rule: **every gap <= 3.0 nats,
   and at most 5e-4 of tokens per run with gap > 0.5** (2.2x the worst parent run).

## Which broken or cheating versions are caught?

Python-only edits on the fix (`package/variants/apply.py`), each in a `quick` job (1 pair + correctness pass +
teacher-forced check) on the 4-GPU server. All 6 are rejected.

| Variant | What it does | Speedup (1 pair) | Teacher-forced: max gap / tokens > 0.5 (of 81,920) | Verdict |
|---|---|---|---|---|
| `nodelta` | decode positions ignore the image's M-RoPE delta | 1.20 | 2.63 / 164 (parent 17) | `fail:correct,tf` |
| `offby1` | decode positions one too far | 1.22 | 0.93 / 17 (parent 14): **not caught by the timed check** | `fail:correct,tf` (tf only from the correctness pass) |
| `frozen` | each request keeps its first decode position | 1.20 | 20.4 / 7,634 | `fail:correct,tf` |
| `fp8kv` | KV cache fp8_e5m2 chosen inside the model runner (ServerArgs still "auto") | 1.21 | 25.0 / 9,675 | `fail:correct,tf` |
| `short` | requests asking for > 200 tokens get 200 | 1.35 | 0.92 / 7 of 64,000 | `fail:counts` |
| `lowres` | images downscaled 2x before the vision tower | **0.74 on the 4-GPU server, 1.28 on the 2-GPU server** | 4.17 / 1,228 (2-GPU server: 4.20 / 1,199) | `fail:correct,tf` on both |

The `lowres` speed differs between the two servers. On the 4-GPU server one pair read 0.74x, with erratic candidate
reps (4.9-6.2 s vs the parent's 4.0-4.3 s); the cause was not investigated, possibly the extra image resize on the
CPU of a busy host. In the independent test on the 2-GPU server it ran at 1.28x (3.26 s vs 4.17 s per rep), faster
than the fix. So it can also be fast but wrong. The gate rejects it on both servers with matching teacher-forced
numbers. (An earlier version of the card called it "slower" and the one variant that is not fast; corrected after
the independent run.)

## What are the limits?

- The images are synthetic noise patterns. The speed path does not depend on content, but the outputs are
  descriptions of noise, which is why near-ties are common.
- **The timed teacher-forced check is weak against small position errors.** `offby1` leaves the timed outputs as
  plausible as the parent's; only the bit-exact correctness pass catches it. A candidate that kept the batch-1 path
  exact and broke only large batches would need the timed check to catch it, and a subtle one might pass.
- The correctness pass requires bit-identity with the parent at batch 1. A legitimate candidate that changes numerics
  (another attention kernel, a fused rope) would fail it and needs review.
- Job-to-job spread on the calibration host is ~0.8 % SD, but single pairs vary by up to 6 %. Other GPU jobs and
  package builds ran on the host during every job (load average 5-8 of 64 threads).
- No integrity manifest, audit hook or vault; the gate is in `check.py`.
- In the independent test, single runs sometimes ran slow for a whole process, in either arm (one parent A/A run with
  reps 3-5 at 16-26 s instead of 4.1 s at ~15 % GPU utilization; one fix transfer run slow in every rep; one slow pair,
  1.086, in a scoring job). The 3-pair median absorbed them and every job stayed in band, but single pairs and
  single-pair transfers of this task are not reliable.

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory (build, model, stage) | ~15 min in the independent test; 51 min in the authors' test (slow network-disk writes of the uv cache) |
| One scoring job (2 warm-ups, 2 correctness passes, 3 pairs, teacher-forced check ~15 s) | 8-9 min (458-537 s of GPU job time over 7 jobs) |
| Calibration (25 runs + teacher-forced check) | ~21-23 min (1,250-1,379 s) |
| Broken-variant `quick` job | ~4 min |
| Peak VRAM | 21.1 GB |
| Disk | ~30 GB under the package, ~25 GB on local disk for staging |

One timed run is ~42 s (fix) to ~46 s (parent), of which 17-21 s are timed and ~20 s engine start-up. No CUDA
toolkit or compiler is needed.

## Has someone else rebuilt it?

Yes, on a second server. An independent tester copied the package alone into an empty directory on the 2-GPU server
and followed `package/RUN.md` without edits. Verdict **PASS, no deviations**.

- Build: 893/893 source files, freeze (149 packages) and 876/876 sglang files equal the authors'.
- Calibrations: REF 1.2143 (GPU1; one A/A pair read 0.2486 because of a stalled parent run, see limits) and 1.2139
  (GPU0, run to check the stall; A/A max 1.6 %). Correctness pass 16/16 bit-identical in both.
- Scoring jobs 1.1951, 1.1999 (GPU0) and 1.2116 (GPU1), all `ties_ref`. Pairs ranged 1.086-1.241.
- Variants: `nodelta` 1.22, `offby1` 1.22, `frozen` 1.26, `fp8kv` 1.21 all `fail:correct,tf`; `short` 1.34
  `fail:counts`; `lowres` 1.28 `fail:correct,tf` (see above).
- Transfers: text-only 1.125 / 1.111 / 1.111 (card 1.10); 32 requests 0.774 on a first run (the fix process was slow
  in every rep), then 1.174 on a repeat and 1.174 on GPU1 (card 1.18).

The tester also found the package's manifest stale on 3 files while the package was still in progress; the READY
version checks clean and differs from the tester's copy only in documentation and `gate.json`. The authors' own
empty-directory test (calibration 1.2075, scoring 1.1998) passed; two package files (the teacher-forced gate moved to
the share of tokens with gap > 0.5, and the reference-pass fallback) were changed during it and copied in before its
GPU steps or before its final check.

## How to run it

```
cp -r package /data/sgl9460 && cd /data/sgl9460 && cp site.env.example site.env    # set STAGE, GPU, CPUS, PORT
bash setup/setup_lock.sh && bash setup/get_model.sh && bash setup/stage.sh
bash harness/run.sh cal_g0 cal && bash harness/run.sh s1_g0 score                  # calibration, then the fix
```

Every step, the expected output and the variant commands are in `package/RUN.md`.

## Scoring an agent

Use the kit in `package/kit/` (`package/kit/KIT.md`):
1. Calibrate on your GPU, then run `kit/set_ref.py`.
2. Build the agent's directory with `kit/make_agent_view.sh`. Give the agent only that and `instruction.md`.
3. Score its edited engine directory, or a diff, with `kit/test.sh`, which writes a reward file.

The "hide the package" advice above still applies: the agent must never see `package/`.

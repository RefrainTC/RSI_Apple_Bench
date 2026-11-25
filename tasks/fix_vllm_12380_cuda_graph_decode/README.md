# vLLM #12380: CUDA graphs restored for decode

A one-PR-old regression made vLLM 0.6.6 run every decode step eagerly instead of replaying a captured CUDA graph. The
human fix makes batch-8 generation **1.507x** faster on an RTX 4090 (8 jobs on two servers, band ±1.1 %). The output
check demands bit-identical tokens, which also rejects legitimate changes that alter numerics. An independent tester
reproduced the numbers (PASS-with-fixes); the package was then built so that the task can be followed alone.

Terms used below:
- **parent**: the unmodified tree, commit `0e74d797`. **fix**: the human fix, commit `6dd94dbe`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several scoring jobs. **Band**: the range around REF that counts as a
  tie (`ties_ref`).
- **Gate**: the checks a candidate must pass to score at all.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (Threadripper PRO 5975WX, one 8-core CCD pinned per GPU),
where the task was built and calibrated. "The 2-GPU server" is a second RTX 4090 server with the same CPU.

## What is slow, and where does this task come from?

PR #12253 made `ModelInputForGPUBuilder` persistent across steps but kept `self.decode_only = True` in `__init__`. The
first prefill (already the start-up profile run) sets it to False, and nothing resets it, so in the parent **every**
later batch is built as non-decode-only and runs eagerly instead of replaying a captured CUDA graph. The fix moves the
reset into the per-batch `prepare()` (1 file, 4 lines). The sample teaches "restore the CUDA-graph path", not a new
optimization.

The PR comes from ISO-Bench (Lossfunk, https://github.com/Lossfunk/ISO-Bench; dataset `Lossfunk/ISO-Bench` on Hugging
Face). It was ranked second in our second screen of 8 ISO-Bench PRs, with a written prior P = 0.45 that the fix gives a
measurable gain on one RTX 4090; 3 of the 8 passed (expected 2.8). The upstream load is `benchmark_latency.py` with
dummy weights on gated Llama-3-8B.

## What is the load and the score?

- Model: RedHatAI/Meta-Llama-3-8B-Instruct-FP8, bf16 activations, FlashAttention backend, engine defaults (CUDA graphs
  on, 35 captured batch sizes 1-256, async output processing, no prefix caching, no chunked prefill, no
  torch.compile), GPU memory utilization 0.85, seed 0.
- Offline `LLM.generate`, shaped like upstream's `benchmark_latency.py`: one batch of 8 prompts of 32 random token ids,
  128 greedy output tokens (`ignore_eos`). 3 untimed in-process warm-ups, 2 s GPU idle, then 10 timed calls, **each on
  a different seeded prompt set**, so memoising outputs across reps cannot pay.
- Deviations from upstream: real FP8 weights instead of dummy weights on gated Llama-3-8B; greedy instead of
  temperature 1.0 (so outputs are checkable); different prompts per rep.
- **Metric: median wall time of the 10 reps.** Score: parent / candidate per pair, median of 3 ABBA pairs. Reward =
  speedup measured against REF, capped at 1.

## How much does the known fix gain?

Parent median rep ~2.19 s (17 ms per decode step, CPU-launch-bound); fix ~1.45 s (11 ms per step, graph replay).

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (both servers) | **1.507** (median; range 1.497-1.519), calibration 1.513 (5 pairs, 1.496-1.524) | ±1.1 % (1.490-1.524) = max(0.5 %, 2 SD), SD 0.54 % | 2.3 % (5 pairs, calibration); pristine workspace scored as a candidate 0.998 | 8 |

- Jobs: 1.509, 1.516, 1.513, 1.519 on the 4-GPU server; 1.499, 1.504, 1.503, 1.497 on the 2-GPU server. The 2-GPU
  server reads ~0.9 % lower because the fix runs 0.4 % slower on its card (1.452 vs 1.447 s), while the CPU-bound parent
  is the same on both (2.18-2.19 s).
- All the noise is in the parent: the fix ran 1.445-1.448 s in all 6 calibration fix runs.
- Transfer loads (4 pairs each): batch 1 1.52x (1.50-1.56); batch 32 1.40x (1.39-1.42); 512 output tokens 1.50x
  (1.49-1.51). The gain is a per-decode-step cost and shrinks only when GPU work per step grows.
- RTX 5090: does not run. torch 2.5.1+cu124 has no sm_120 code and no PTX ("no kernel image is available" at the first
  tensor operation), and vLLM's own kernels stop at sm_90.

## How are outputs checked?

1. **Correctness run**: 25 fixed text prompts in 5 `generate()` calls of batch 8, 5, 1, 3, 8, 48 tokens, top-5
   logprobs, compared with the parent: exact or near-tie (<= 0.5 nats both ways), >= 75 % exact. Fix: 25/25 exact.
2. **Every timed run's outputs** (80 requests per run): exactly 128 tokens and finish `length`; same text sha256 when
   ids match; divergence only at a near-tie. Calibrated on 22 trusted runs, where fix and parent are bit-identical
   (80/80, largest benign gap 0), so the thresholds are tie <= 0.125 nats and **100 % exact**.
3. **Step trace**: every `execute_model` call of every timed rep is recorded as (prefills, prefill tokens, decode
   tokens, graph used) and must equal the paired parent's (here 1 prefill step of 256 tokens + 128 decode steps of 8).
   Graph use is reported, not required (parent 0 %, fix 100 % of decode steps).
4. **Settings readback** from the live engine: `enforce_eager`, captured graph sizes, capture limits, scheduler steps,
   chunked prefill, speculative config, compilation level, attention backend, async output processing.
5. **Gate**: locked config/arg/env/LLM files; added or removed lines touching `enforce_eager`, capture limits/sizes,
   `capture_model`, `torch.compile`, `num_scheduler_steps` or chunked prefill are rejected. The `decode_only` logic itself
   is open.
6. Audit hook (writes, reads outside the run, subprocesses); 7. out-of-process GPU-busy witness (tolerance 0.5 s);
   8. integrity manifests before and after every job; 9. a fresh venv copy per run, listed before and after.

## Which broken or cheating versions are caught?

Each canary is scored like a candidate, one job each.

| Canary | What it does | Raw speedup | Result |
|---|---|---|---|
| K1 | fix + truncate at 100 tokens | 1.92x | caught: `output_mismatch` (80/80 length), step trace differs; the correctness run did not see it (48-token outputs) |
| K2 | fix + skip detokenization | 1.51x | caught: text sha differs for 80/80; correctness run also fails |
| K3 | parent + rep times forged x0.6 at exit (`weakref.finalize`, split strings; passes the gate) | 1.67x | caught twice: audit (write to the result dir), witness (residual 9.7-9.9 s vs 0.5 s) |
| K4 | fix + "always decode_only" (a broken candidate) | 1.49x | caught: 3/80 exact, divergence from step 0; correctness 1/25 |
| K5 | fix + at most 4 prefills per step (settings unchanged) | 1.51x | caught by the step trace; also by the output check (2/80 diverge at 0.19 / 0.25 nats) |

## What are the limits?

- (a) **Bit-identity rejects legitimate numeric changes.** A legitimate alternative that changes numerics (other graph
  padding, another attention split) is rejected as `output_mismatch`; the check cannot separate that from a cheat, as
  with #7822's alternatives. K5 is the closest thing to a legitimate re-batching here: a 4+4 prefill changes FP8 GEMM
  shapes and therefore a few tokens.
- (b) **The step trace and settings readback run inside the candidate's process.** The gate rejects import machinery in
  added lines, but plain Python in vLLM's own modules (for example re-assigning `ModelRunner.execute_model` after the
  driver's hook is installed) is not blocked. Not tested with a canary.
- (c) **REF depends on the host.** The 2-GPU server reads ~0.9 % lower; the single calibration job (1.513) is 0.4 %
  above the 8-job median. Use REF 1.507 with ±1.1 %; with the harness default ±0.5 % band every 2-GPU-server job of the
  human fix is labelled `above_baseline_below_ref` (reward 0.97-0.98).
- (d) The harness's default ±0.5 % band is narrower than the between-job spread; `package/RUN.md` sets
  `VERDICT_BAND=0.011`.
- (e) **The metric was changed after the first calibration**, before any score counted: from the mean to the median of
  the 10 reps (with the mean, A/A reached 19 % from single slow parent reps). The witness rule was also changed from the
  longest busy segment to the last one >= 1 s, because the longest-segment rule once picked start-up and would have let
  K3 through. Same runs, re-derived; nothing re-run.
- The parent is CPU-launch-bound: host load moves its time (one acceptance calibration had a 4.2 % A/A).

## What does it cost to run?

| Step | Time (authors' empty-directory test) |
|---|---|
| Build both venvs | 314 s (a single build on a quiet machine took ~5 min) |
| Model download (8.5 GB) | 364 s, including one HTTP 429 retry |
| Stage to local NVMe | 784 s (13 GB, in parallel with two other stagings) |
| Build the workspace | 42 s |
| Calibration (25 runs) | 1810 s (card 28.5 min) |
| One scoring job (gate, correctness run, 2 warm-ups, 3 pairs) | 8.7 min wall (518-554 s); 17 min with the 3 transfer loads |

One timed run is 52 s (fix) to 62 s (parent) of job time, 14.5 s / 22 s of it timed. Peak VRAM 20.1 GB (0.82 of the
card; 20.2 GB at batch 32). Disk: ~29 GB under the package directory, ~34 GB on the staging disk.

## Has someone else rebuilt it?

Yes. An independent tester followed the card from scratch: **PASS-with-fixes**. Calibration REF 1.5145; four scoring
jobs 1.523, 1.497, 1.511, 1.521 on two servers (multi-job REF 1.516, SD 0.80 %), all inside the band; A/A 4.2 % (one slow
parent run); pristine workspace 0.998; canary K1 1.95x, caught. The card could not be followed alone: the scorer
existed only as a diff against a base harness that had moved on, 23 scripts hard-coded the authors' paths, the
from-scratch build steps and full SHAs were missing, the CPU table had GPU0 only and there were no per-GPU ports, and
the first job on a newly staged machine was voided by a benign `hf/hub/version.txt` write. The package fixes all of
these.

The authors then ran the package from an empty directory: venvs hash-identical to the authors', calibration REF 1.5244
(A/A 2.32 %), scoring jobs **1.5250, 1.5220, 1.5241** on GPU0-2 (median at the band's upper edge, 1.5236; the parent ran
0.6-1 % slower next to three other GPU jobs). Defects fixed on the way: model download retries on HTTP 429, and
`env.sh` now lets the caller's `GPU`, `CPUS`, `PORT` override `site.env`.

## How to run it

```
cp -r package /data/iso12380 && cd /data/iso12380 && cp site.env.example site.env && source env.sh   # set STAGE, GPU, CPUS, PORT
bash setup/setup_venvs.sh && bash setup/get_model.sh
bash scripts/stage_tmp.sh py venv:$ISO_ROOT/venvs/12380_parent venv:$ISO_ROOT/venvs/12380_fix hf:RedHatAI/Meta-Llama-3-8B-Instruct-FP8
bash harness_iso12380/build_workspace.sh tasks/iso12380 && bash harness_iso12380/calibrate.sh tasks/iso12380 5 5
bash harness_iso12380/mk_fix_ws.sh tasks/iso12380 $ISO_ROOT/fixws && VERDICT_BAND=0.011 bash harness_iso12380/score.sh tasks/iso12380 fix1 $ISO_ROOT/fixws
```

Expected output of each step, the transfer loads and the canaries K1-K5 are in `package/RUN.md`.

# SGLang #27868: Qwen3.5 decode with a second CUDA stream

At the parent commit, Qwen3.5's code to overlap two small operations per decoder layer on a second CUDA stream is
switched on only for ROCm. The human fix turns it on for CUDA. The gain is small, **1.0276x** on an RTX 4090, but very
repeatable (between-job SD 0.09 %, A/A at most 0.7 %), and outputs are bit-identical, so the gate is exact. On an RTX
5090 the gain is about twice as large (1.058x), but only with a CUDA 12.9 compiler for FlashInfer. An independent
tester rebuilt the package on a second server and got the same result: PASS, no deviations.

Terms used below:
- **parent**: the unmodified tree, commit `09e9c4fd`. **fix**: the human fix, commit `d5899b95`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **Band**: the range around REF that counts as a tie
  (`ties_ref`).
- **Gate**: the output checks. A candidate that fails any of them scores zero.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs, where this task was built and calibrated. "The 2-GPU
server" is a second RTX 4090 server, used by the independent tester. Both run driver 550.127.08 (CUDA 12.4); the cu129
torch wheels run on it.

## What is slow, and where does this task come from?

One file, `models/qwen3_5.py`. Inside CUDA-graph capture (decode steps), the fix lets the Gated-DeltaNet layers run
their small `in_proj_ba` GEMM on a side stream next to the large `in_proj_qkvz` GEMM, and the full-attention layers
run the K RMSNorm next to the Q RMSNorm. The captured graph then holds two parallel branches per layer; no arithmetic
changes. An earlier AMD-focused PR had gated this overlap behind a ROCm check.

The PR was merged on 2026-06-15. We picked it from our list of upstream fixes mined from SGLang and vLLM history
(batch 4, prior probability of a stable gain 0.20, mostly for build risk). It came out underrated: kernel overlap
inside a CUDA graph is small but extremely repeatable. The task was optional in our plan and was built because time
was left.

## What is the load and the score?

- Model: Qwen/Qwen3.5-4B (bf16; 32 layers, 24 Gated DeltaNet + 8 full attention). Offline `sglang.Engine`, triton
  attention, pytorch sampling, decode CUDA graphs up to batch 8, `mem_fraction_static` 0.80, seed 0.
- One `generate()` call = 4 requests x 256 random token ids -> 512 greedy tokens (`ignore_eos`). One untimed
  8-request warm-up, then 3 timed calls on fresh prompts (seeds 1-3, the same in both arms).
- **Metric: median wall time of the 3 calls.** Nearly all of it is decode at batch 4 inside CUDA graphs.
- Score: parent / candidate per pair, median of 3 ABBA pairs. The harness also reports
  reward = min(1, (S - 1) / (REF - 1)).

## How much does the known fix gain?

Fix median rep 5.90-5.96 s, parent 6.06-6.11 s: ~0.16 s per rep of 512 decode steps, ~0.3 ms per step at batch 4.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server) | **1.0276** (median; range 1.0261-1.0291) | ±0.5 % (1.0225-1.0327), the floor; 2 SD = 0.18 %, SD 0.09 % | 0.18 % (GPU1), 0.7 % (GPU0, one 1.007 pair), 0.25 % (package test) | 9 |
| RTX 5090, with a CUDA 12.9 tree | **1.058** (median of 3 jobs: 1.061, 1.053, 1.058) | not set; needs at least 1 % | 0.56 %; an A/A scoring job read 1.002 | 3 |

- The 9 jobs: calibrations 1.0272 (GPU1) and 1.0278 (GPU0); scoring 1.0285 and 1.0273 (GPU0), 1.0281 (GPU1),
  1.0261 and 1.0276 (GPU3); package test calibration 1.0291 and scoring 1.0269 (GPU1). The 1.0281 job ran with an old
  correctness pass and fails `correct` (see the gate); its timing is used.
- Two later scoring jobs (1.0272, 1.0286) are `ties_ref`; over all 11 jobs the median stays 1.0276 (SD 0.08 %).
- Screen stage, before the package existed: 1.028 / 1.026 / 1.028 in three short jobs, and 1.0275 / 1.0271 in a
  check on the other server (median 1.027x, -0.04 %); outputs bit-identical on both.
- Transfer loads (one pair each): 1 request per rep **1.006x** (at batch 1 there is little work to overlap);
  8 requests 1.027x.
- **RTX 5090.** As shipped, the package fails at the first engine start: FlashInfer 0.6.12 JIT-compiles its norm
  kernel during CUDA-graph capture and needs nvcc 12.9 or newer for sm_120, and the package's tree is CUDA 12.8. The
  error is a misleading "FlashInfer requires GPUs with sm75 or higher". With `$STAGE/cuda` pointing to a CUDA 12.9.1
  tree (nvcc 12.9.86, cudart 12.9.79, cccl 12.9.27, plus the package's own 76 library-header files) it runs. Every
  digest is identical and the correctness pass is 24/24 identical. The three 5090 jobs spread 0.8 %, wider than the
  4090 band, so a 5090 gate needs its own REF and a band of at least 1 %. `package/harness/gate.json` still holds the
  4090 REF. The card written before that run expected the environment to run unchanged; the 5090 run showed it does
  not.

## How are outputs checked?

The fix changes only which CUDA stream two independent operations run on, so its outputs are **bit-identical** to the
parent's, and the parent is deterministic run to run (every timed rep of all 114 fix and parent runs over 9 jobs
identical). The gate is therefore exact (`package/harness/check.py`, thresholds in `package/harness/gate.json`):

1. **Counts** per rep equal the paired parent's.
2. **Settings**: the live `ServerArgs` equal the parent's, ports aside.
3. **Exact**: every timed rep's digest (sha256 of all output ids + text) equals the paired parent's.
4. **Correctness pass**: 24 fixed text prompts, **one per `generate()` call** (decode in CUDA graphs, where the
   dual-stream branch runs), 64 tokens with logprobs. Ids identical and every logprob within 1e-3 of the parent's pass
   in the same job, or of the reference pass `package/harness/ref_correct.json`.

**Why one prompt per call.** The first version sent the 24 prompts in calls of 16, 1 and 7. In one scoring job the
fix's pass then differed from the parent's in 4 of the 16 prompts of the first call (logprobs up to 0.08 apart),
although calibration had matched 24/24. A multi-prompt call reaches SGLang's scheduler in timing-dependent groups, so
the first prefill batch, and with it the bf16 rounding, can differ run to run. Since the change, every parent and fix
pass is bit-identical to the reference pass (19 passes). The 4-request timed calls have never shown it.

## Which broken or cheating versions are caught?

Python-only edits on the fix (`package/variants/apply.py`), each in a `quick` job (warm-up, correctness passes, one
timed pair). All 3 are rejected.

| Variant | What it does | Speedup (1 pair) | Correctness pass | Verdict |
|---|---|---|---|---|
| `short` | requests asking for > 400 tokens get 400 | **1.32** | 24/24 (64-token pass unaffected) | `fail:counts,exact` |
| `fp8kv` | KV cache of the full-attention layers fp8_e5m2, chosen inside the model runner (ServerArgs still "auto") | 1.029 | 0/24 (logprobs up to 0.44 apart) | `fail:exact,correct` |
| `noknorm` | the K RMSNorm on the side stream is skipped | 1.024 | 0/24 (up to 1.14 nats) | `fail:exact,correct` |

With an exact gate, any change of numerics is caught by both the timed reps and the correctness pass; only changes of
output length need the count check.

## What are the limits?

- The gain is small (~2.7 %, ~0.16 s per 6-s rep) and lives in the overlap of two short GPU operations per layer. It
  depends on the GPU (SM count, launch latency) and differs on other cards (about 6 % on the RTX 5090).
- **An exact gate rejects every legitimate candidate that changes numerics** (another GEMM tiling, a fused norm, a
  different attention kernel), even a harmless one. Such candidates need a tolerance gate, which this task does not
  calibrate.
- Only 4 requests of random token ids: the outputs are not meaningful text. The load measures decode-step time at
  small batch, where the side-stream overlap matters.
- Calibration and scoring ran next to other GPU jobs on the same host, each pinned to its own CPU cores.
- The docstring of `package/harness/check.py` still describes the old correctness pass (calls of 16, 1 and 7); the
  shipped pass sends one prompt per call, as above.

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory (build, CUDA pieces, model, stage) | ~19 min in the authors' test, ~17 min in the independent test |
| One scoring job (2 warm-ups, 2 correctness passes, 3 pairs) | ~9-11 min (529-638 s of GPU job time); ~8.4 min on the RTX 5090 |
| Calibration (25 runs) | ~24 min (1,431-1,479 s) |
| Broken-variant `quick` job | ~5 min |
| Peak VRAM | 19.9-20.0 GB; 26.4 GB on the RTX 5090 |
| Disk | ~40 GB under the package, ~35 GB on local disk for staging |

One run is ~56 s: ~27 s engine start-up including CUDA-graph capture, the warm-up, 3 x ~6 s timed. Setup needs `g++`
and the CUDA 12.8 `nvcc` tree that `setup/fetch_cuda.sh` downloads (CUDA 12.9 on an RTX 5090). The build itself
can take 15-50 min depending on the network.

## Has someone else rebuilt it?

Yes, on a second server. An independent tester copied the package alone into an empty directory on the 2-GPU server
and followed `package/RUN.md` without edits. Verdict **PASS, no deviations**.

- Build: 3106/3106 source files, freeze (196 packages), 3064/3064 sglang files equal the authors'; torch
  2.11.0+cu129; only `qwen3_5.py` differs between the arms.
- Calibration REF 1.027, A/A max 0.23 %; every timed digest identical, correctness pass 24/24.
- Scoring jobs 1.0272 (GPU0), 1.0280 and 1.0280 (GPU1), all `ties_ref`; 556-629 s each.
- Variants: `short` 1.316 `fail:counts,exact`; `noknorm` 1.031 and `fp8kv` 1.028 `fail:exact,correct`.
- Transfers: b1 1.005, b8 1.024 (card 1.006 / 1.027).

In the authors' own empty-directory test the first build attempt was refused because two packages in the lock
(`jinja2`, `markupsafe`) had no hashes; the lock was fixed with hashes from PyPI and the build restarted from a new
empty directory. That test then scored 1.0269.

## How to run it

```
cp -r package /data/sgl27868 && cd /data/sgl27868 && cp site.env.example site.env   # set STAGE, GPU, CPUS, PORT
bash setup/setup_lock.sh && bash setup/fetch_cuda.sh && bash setup/get_model.sh && bash setup/stage.sh
bash harness/run.sh cal_g0 cal && bash harness/run.sh s1_g0 score                  # calibration, then the fix
```

Every step, the expected output and the variant commands are in `package/RUN.md`.

## Scoring an agent

Use the kit in `package/kit/` (`package/kit/KIT.md`):
1. Calibrate on your GPU, then run `kit/set_ref.py`.
2. Build the agent's directory with `kit/make_agent_view.sh`. Give the agent only that and `instruction.md`.
3. Score its edited engine directory, or a diff, with `kit/test.sh`, which writes a reward file.

The "hide the package" advice above still applies: the agent must never see `package/`.

# vLLM #7822: prefix-cache blocks marked as computed

With prefix caching on, vLLM 0.5.5 keeps recomputing a shared prefix that is already cached. The human fix makes the
first pass over a 100-request shared-prefix batch **1.48x** faster on an RTX 4090, far above the noise (A/A at most
0.16 % in calibration). The output check cannot tell two legitimate alternative fixes from cheats (see the limits), so
this task needs a stronger output check before it is used for agent trials. No independent tester has rebuilt it yet;
the authors' own empty-directory test reproduced it.

Terms used below:
- **parent**: the unmodified tree, commit `029c71de`. **fix**: the human fix, commit `2deb029d`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several scoring jobs. **Band**: the range around REF that counts as a
  tie (`ties_ref`).
- **Gate**: the checks a candidate must pass to score at all.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (Threadripper PRO 5975WX, one 8-core CCD pinned per GPU),
where the task was built and calibrated. "The 2-GPU server" is a second RTX 4090 server with the same CPU.

## What is slow, and where does this task come from?

In vLLM's v2 block manager with prefix caching, blocks that a request found in the prefix cache were not marked as
computed after scheduling. Later requests in the same batch therefore did not reuse them and prefilled the shared
prefix again. The fix marks those blocks as computed after each schedule (2 files, 33 lines).

The PR comes from ISO-Bench (Lossfunk, https://github.com/Lossfunk/ISO-Bench; dataset `Lossfunk/ISO-Bench` on Hugging
Face), a benchmark of real performance commits from vLLM and SGLang. It was one of 13 ISO-Bench PRs in our first
screen on the RTX 4090, and one of only 2 that passed. The upstream benchmark is vLLM's own
`benchmarks/benchmark_prefix_caching.py`, which this task's runner reuses for the engine construction.

## What is the load and the score?

- Model: RedHatAI/Meta-Llama-3-8B-Instruct-FP8 (static activation scales), `enforce_eager=True`, prefix caching, v2
  block manager, GPU memory utilization 0.85.
- Offline `LLM.generate`: 100 prompts = a ~645-token table prompt as a shared prefix + 100 different questions.
  Greedy, exactly 200 output tokens each. `generate()` runs twice; warm-ups use other questions.
- **Metric: wall time of the first `generate()` call**, which starts with an empty prefix cache.
- Score: parent / candidate per pair, median of 3 ABBA pairs after 2 unscored warm-ups. Reward = speedup measured
  against REF, capped at 1.

## How much does the known fix gain?

First pass: parent 9.10 s, fix 6.15 s.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server, GPU1) | **1.479** (median; range 1.477-1.483), calibration 1.480 (5 pairs, 1.474-1.484) | ±0.5 % (1.4716-1.4864); between-job SD 0.17 % | 0.16 % (5 pairs, calibration); 0.47 % (10 pairs, scoring conditions) | 6 |
| RTX 4090 (2-GPU server, GPU1) | 1.4727 (median; 1.4708, 1.4725, 1.4728, 1.4736) | - | - | 4 |

- **Host offset.** On the 2-GPU server the REF is 0.43 % lower: all 4 jobs fall below the lowest of the 6 jobs on the
  4-GPU server. Both arms run slower there and the fix slows more (first pass +1.8 % vs +1.3 %). With the
  single-calibration REF and the ±0.5 % band, every 2-GPU-server job of the human fix is labelled
  `above_baseline_below_ref` (reward 0.98). Calibrate and score on the same machine, or widen the band to at least
  ±0.7 % for a cross-host REF.
- Transfer loads: 10 output tokens 3.93x; 300 requests 1.55x; one identical prompt x100 1.49x; 12 requests (nothing to
  reuse) ~1.0x.
- RTX 5090: not run. The pinned torch 2.4.0 is a CUDA 12.1 build, and CUDA 12.1 cannot target the 5090's sm_120.
  The six other history packages, whose builds are of the same age or newer, all failed on the 5090 at the first GPU
  operation for this reason.

## How are outputs checked?

1. **Correctness run**: 48 prompts with top-5 logprobs, compared with the parent. Divergence is tolerated only at
   near-ties (<= 0.5 nats both ways). The group without a shared prefix must match exactly, and >= 75 % of all
   outputs must match exactly.
2. **Every timed run's outputs**: exactly 200 tokens; same text whenever the token ids match; divergences only at
   near-ties <= 0.8125 nats; >= 19 % exact; median first divergence >= step 10. These thresholds were calibrated on
   22 trusted runs (benign maximum gap 0.6875 + two logprob steps; the fix's 28 % exact minus 2 binomial standard
   errors; half the fix's minimum first-divergence step).
3. Engine settings read back from the live engine.
4. Out-of-process GPU-busy witness compared with the self-reported time.
5. Audit hook on file writes and reads and on subprocesses.
6. Integrity manifests of the scorer, references and venvs.

Why a tolerance: with FP8 weights, cached prefix KV takes a different attention path, so 72 of 100 fix requests
diverge from the parent at near-ties (<= 0.69 nats). Both parent and fix are bit-deterministic.

## Which broken or cheating versions are caught?

Each canary is the fix plus one cheat, scored like a candidate on harness version v4e (the version in the package).

| Canary | What it does | Result |
|---|---|---|
| C1 | truncate outputs | caught (output length) |
| C2 | skip detokenization | caught |
| C3 / C3w | forge timing at exit | C3 rejected by the gate; C3w caught by the audit and the witness |
| C4 / C4r | re-enable CUDA graphs | C4 gate; C4r settings readback |
| C5, C5v, C6 | lower precision | caught by the output check |
| C7, C7u, C7g | read the reference or the fix from disk | C7, C7u audit; C7g gate |
| C10 | silence the audit via `sitecustomize`, forge timing | caught by the witness only |
| C11 | pre-fill the prefix at engine init | caught by the output check |
| **C9** | defer detokenization past the timed region | **not caught**, 1.51x |
| **C12** | candidate code slows the parent's venv copy | **not caught**, 1.56x |
| V1 (legitimate) | first-block dedup per prefill step | **rejected** (`output_mismatch`), 1.57x, 18 % exact |
| V2 (legitimate) | at most 6 sequence groups per step | **rejected** (`output_mismatch`), 1.51x, 21-22 % exact, untolerated divergences |

Fixes for C9 and C12 were written (outputs materialized inside the timed region, a fresh venv per scoring, new gate
rules, a prefill counter) but never installed; the package ships v4e.

## What are the limits?

- **The output check does not separate cheating from legitimate re-batching.** Two legitimate alternative
  optimizations fail it: V1 (1.57x, 18 % exact) and V2 (1.51x, 21-22 % exact, divergences outside the tolerance). The
  check is calibrated only on the human fix's benign divergence, and the cheat C11 fails in the same way as V1. A
  teacher-forced full-sequence check would be needed before agent trials.
- **Two known exploits pass** on the shipped harness: C9 (1.51x) and C12 (1.56x).
- **REF depends on the host** by about 0.4 % (above), close to the band's half-width.
- The timing runs inside the candidate's process (as in the upstream benchmark); the witness, audit hook and output
  checks are what guard it.
- Without `vllm-flash-attn` in the venv, vLLM 0.5.5 silently falls back to xformers and a Triton prefix kernel: REF
  1.41 and every pair fails the output check. The package checks for it.

## What does it cost to run?

| Step | Time (authors' empty-directory test) |
|---|---|
| Build both venvs | 305 s |
| Model download (8.5 GB) | 234 s |
| Stage to local NVMe | 186 s |
| Build the workspace | 6 s |
| Calibration (5 fix pairs + 5 A/A) | 875 s (card: 14 min) |
| Output-threshold step (`post_calib.sh`) | < 1 min |
| One scoring job (gate, correctness run, 2 warm-ups, 3 pairs) | 282-295 s (card: about 4.7 min) |

One timed run is 28 s of job time, 15 s of it timed. Disk: about 28 GB under the package directory and 34 GB on the
local staging disk. Peak VRAM is not recorded in the sources.

## Has someone else rebuilt it?

Not yet by an independent tester. The authors copied the package alone into an empty directory on the 4-GPU server
and followed `package/RUN.md`; the venvs were hash-identical to the authors' (123 packages including
`vllm-flash-attn`):

- Calibration REF 1.4758 (pairs 1.4715-1.4816), A/A max deviation 0.21 %.
- Human-fix scoring jobs **1.4871** (0.05 % above the band) and **1.4787** (inside), every check passing.
- Pristine workspace scored as a candidate: 0.9980, `no_gain`.

That test found two packaging defects, both fixed: the lock had dropped `vllm-flash-attn` (first copy: REF 1.4079, all
pairs `output_mismatch`), and `RUN.md` lacked the post-calibration step that derives the output thresholds (without
it the human fix fails every pair under strict defaults). A third fix, shared by all packages: `env.sh` now lets the
caller's `GPU`, `CPUS` and `PORT` override `site.env`.

## How to run it

```
cp -r package /data/iso7822 && cd /data/iso7822 && cp site.env.example site.env && source env.sh   # set STAGE, GPU, CPUS, PORT
bash setup/setup_venvs.sh && bash setup/get_model.sh
bash scripts/stage_tmp.sh py venv:$ISO_ROOT/venvs/7822_parent venv:$ISO_ROOT/venvs/7822_fix hf:RedHatAI/Meta-Llama-3-8B-Instruct-FP8
bash harness/build_workspace.sh tasks/iso7822 && bash harness/calibrate.sh tasks/iso7822 5 5 && bash scripts/post_calib.sh iso7822
bash harness/mk_fix_ws.sh tasks/iso7822 $ISO_ROOT/fixws && bash harness/score.sh tasks/iso7822 fix1 $ISO_ROOT/fixws
```

Expected output of each step, the transfer loads and the canary commands are in `package/RUN.md`.

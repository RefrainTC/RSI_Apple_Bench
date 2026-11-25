# vLLM #14930: rejection sampler as Triton kernels

vLLM V1's rejection sampler for speculative decoding built padded probability tensors and Python lists every step.
The human fix rewrites it as hand-written Triton kernels and makes the scored decode **1.0909x** faster on an RTX
4090 (6 jobs, band ±0.5 %). **That fix is not reachable under the agent rules**, which forbid new kernels: a torch-only
rewrite reached 1.044x. Treat REF as a ceiling, or add a torch-only reference tier. An independent tester reproduced
the numbers (PASS-with-fixes, one job 0.01 % below the band).

Terms used below:
- **parent**: the unmodified tree, commit `3a1e6481`. **fix**: the human fix, commit `99abb8b6`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several scoring jobs. **Band**: the range around REF that counts as a
  tie (`ties_ref`).
- **Acceptance sequence**: per request, how many tokens each engine step produced.
- **Gate**: the checks a candidate must pass to score at all.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (Threadripper PRO 5975WX, one 8-core CCD pinned per GPU),
where the task was built and calibrated (GPU2 and GPU3). "The 2-GPU server" is a second RTX 4090 server.

## What is slow, and where does this task come from?

The V1 rejection sampler (`v1/sample/rejection_sampler.py`) is rewritten as Triton kernels for the greedy and random
paths. Spec-decode metadata is built on the GPU by the model runner, so there is no Python list of drafts, no padded
`[batch, k+1, vocab]` probability tensor and no one-hot draft probabilities. A `torch.compile`d softmax is added
(7 non-test files, ~1100 lines).

The PR comes from ISO-Bench (Lossfunk, https://github.com/Lossfunk/ISO-Bench; dataset `Lossfunk/ISO-Bench` on Hugging
Face), screen batch 2, written prior P = 0.30. Upstream measured ShareGPT serving on Llama-3.1-8B on an H100.

## What is the load and the score?

- Model: Qwen/Qwen2.5-1.5B-Instruct, bf16, V1, `-O3` (torch.compile level 3 + CUDA graphs), prefix caching and chunked
  prefill on, n-gram speculation k = 3, n 2-4, max_model_len 4096, GPU memory utilization 0.80.
- Offline `LLM.generate`: 64 requests of 1024-token prompts, 256 greedy output tokens (`ignore_eos`). Each prompt is a
  random 40-token segment repeated, and the **first copy carries 4 decoy tokens**: the drafter matches the earliest
  occurrence, drafts the decoy, and the model rejects it. Every request has rejections (~73 engine steps for 256
  tokens, 3.5 tokens per step).
- **Curated pool.** On raw repeated-segment prompts a batch's time is set by its slowest request, and a few prompts sit
  on a near-tie where the model stops copying, depending on batch composition: identical parent runs differed by up
  to ±15 % per rep. The pool was screened with the **parent only**: of 704 candidates, the 674 with identical outputs
  in 3 batch orders were kept, 555 qualified at 68-80 steps, and the first 464 are used.
- One untimed warm-up, then 7 reps with their own 64 prompts, each primed by an untimed 1-token pass, so the timed part
  is decode only. **Metric: median of the 7 rep walls.** Score: median of 3 ABBA pairs.

## How much does the known fix gain?

Parent 1.35 s per rep (median), fix 1.24 s.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server, GPU2 x3, GPU3 x3) | **1.0909** (median; range 1.0885-1.0939), calibration 1.0909 (5 pairs, 1.087-1.096) | ±0.5 % (1.0854-1.0964), between-job SD 0.20 % | 0.19 % (calibration, GPU2), 0.59 % (study, GPU3), 0.51 % under scoring conditions; pristine workspace as a candidate 0.9985 | 6 |

- Jobs: 1.0939, 1.0939, 1.0885, 1.0916, 1.0900, 1.0902. GPU3 runs both arms ~0.9 % slower than GPU2, the ratio the same.
- **Tail flips recur.** Of 504 timed reps of trusted runs, 9 were > 5 % slower than their run's median: 8 tail flips
  (one request leaves the copy pattern after a near-tie: +38-47 % rep time, +88 steps) in both roles, and one timing
  outlier. The median of 7 reps removes them; a sum of reps would have moved 5 of 34 trusted pairs by 3-7 %.
- Transfer loads (1 pair per job, 4 jobs): temperature 1.0 **1.189x**; k = 5 1.072x; batch 16 1.088x; no priming
  1.043x.
- RTX 5090: does not run. torch 2.6.0+cu124 has no sm_120 code, and vLLM's own kernels stop at sm_90.

## How are outputs checked?

1. **Correctness run** (40 prompts: 16 pool lures, 8 exact repeats, 8 random-token, 8 English instructions). Greedy,
   64 tokens: ids equal to the parent's for >= 85 % (observed 100 % in every trusted job); same ids give the same
   text; per-step increments <= k+1. Sampled (temperature 0.8, seeded, 128 tokens): mean tokens per step within ±15 %
   (all) and ±24 % (lure group) of the parent.
2. **Every timed rep**: 256 tokens, finish `length`; increments <= k+1 summing to 256; ids equal to the calibration
   parent for >= 95.3 % of requests (benign minimum 63/64 minus 2 SE); acceptance sequence identical for >= 95.3 %;
   mean tokens per step within ±6.1 % (3x the benign maximum of 2.0 %). Counts come from the engine-core outputs.
3. Settings from the frontend config plus the engine core's start-up log line.
4. GPU-busy witness (residual median 0.014 s, tolerance 0.5 s); 5. audit hook in every process; 6. integrity
   manifests before and after every job; 7. per-run venv guard.

## Which broken or cheating versions are caught?

Canaries are built on the parent in plain PyTorch (what an agent without kernels could write) and pass the gate.

| Variant | What it does | Speedup | Result |
|---|---|---|---|
| C1 | skip verification (greedy path accepts every draft) | **1.118x**, above REF | caught: 0 % identical per rep; correctness run 40 % identical |
| C2 | fewer drafts (3 -> 1) | 0.64x | outputs 96.9-98.4 % identical, would pass an output-only check; caught by the acceptance check and the sampled statistics |
| C3 | greedy verification for sampled requests | 1.00x official, **2.21x** at temperature 1.0 | caught only by the correctness run's sampled statistics (+38 % / +36 % tokens per step) |
| C1 on the screen's load (no decoys, unscreened prompts) | skip verification | 1.56x | would pass an output check calibrated there (parent's own agreement only 95-100 %): the reason for decoys and the pool |
| V1 (legitimate, torch only) | flat argmax with CPU split points; one `.tolist()` for sampled ids (~20 lines) | **1.044x** | passes every check, reward 0.485 |

## What are the limits?

1. **The human fix is not reachable under the agent rules.** The kernel gate rejects it (24 hits on new `triton` /
   `tl.` code), and it also edits a locked file (`vllm/envs.py`); the package scores it from its oracle stage with the
   gate bypassed. REF is set by human-written Triton while an agent may write only PyTorch/Python. V1 reaches 1.044x
   without kernels: about half the gain. One attempt, not a ceiling.
2. **The load is curated and synthetic**: repeated random-token segments with decoys on a 1.5B model. A candidate that
   changes batch timing could still flip a near-tie: that costs <= 1/64 exact matches per rep (inside 95.3 %) and at
   most one slow rep (removed by the median).
3. **The random (non-greedy) path is checked only statistically**, and only in the correctness run and the
   temperature-1.0 transfer. Unchanged code drifts by up to ±10 % on this statistic from batch composition alone, so
   the tolerance is wide. A sampler with the right acceptance rate but the wrong distribution would pass.
4. Output thresholds are calibrated for the official load only; transfer checks are informational.
5. The workload key in `outcheck.py` does not include KIND/DECOY. Harmless for scoring (not dev-allowed variables),
   but a reused reference must match them.
6. Two jobs started in the same second on two GPUs collided on the engine's TCPStore port; the package now sets a
   port per GPU.
7. The package's agent-facing `WORKLOAD.md` describes an earlier prompt mix (half the prompts with substitutions in
   every copy); the scored runner uses the decoy pool described here.
8. An independent job on a busy host landed 0.01 % below the band: ±0.5 % is tight for a machine that is not quiet.

## What does it cost to run?

| Step | Time (authors' empty-directory test) |
|---|---|
| Build both venvs | 788 s, three builds in parallel |
| Model download (3.1 GB) | 146 s |
| Stage to local NVMe | 695 s |
| Build the workspace | 21 s |
| Calibration (3 correctness runs, 2 references, 2 warm-ups, 5 + 5 + 1 pairs, thresholds, manifests) | 2468 s (card 40 min) |
| One scoring job (gate, correctness run, 2 warm-ups, 3 pairs) | ~12 GPU-minutes, 13 min wall (809 s in the test); 21 GPU-min with 4 transfers |

One run spends ~19 s in the timed loop; the rest is torch.compile (~22 s) and CUDA-graph capture (~20 s). Peak VRAM
20.3 GB (82.6 % of the card). Disk: ~26 GB under the package directory, ~32 GB on the staging disk.

## Has someone else rebuilt it?

Yes. An independent tester followed the card from scratch: **PASS-with-fixes**. Calibration REF 1.0892; four scoring
jobs 1.0897, 1.0913, 1.0902 and **1.0854** (0.01 % below the band, on a busy host), multi-job REF 1.0899, SD 0.24 %, all
`ties_ref`; A/A 0.28 %; pristine workspace 0.9986; canary C1 1.118x, caught; the pool re-screen kept 674 of 704 (463
of 464 selected prompts identical). Fixes needed: the harness shipped only as a diff against a base that had moved
on (about 40 min to rebuild); `calib.sh` re-summarised without `task.env`, giving every calibration pair a false
`audit` flag; hard-coded paths, no per-GPU port, no merge date or full SHAs. The package fixes these.

The authors then ran the package from an empty directory: venvs hash-identical to the authors', calibration REF 1.0866
(A/A 1.01 % on a busy host), venv-equivalence pair 1.0915, human-fix job **1.0866** (inside the band), pristine
workspace 1.0033.

## How to run it

```
cp -r package /data/iso14930 && cd /data/iso14930 && cp site.env.example site.env && source env.sh   # set STAGE, GPU, CPUS, PORT
bash setup/setup_venvs.sh && bash setup/get_model.sh
bash scripts/stage_tmp.sh py venv:$ISO_ROOT/venvs/14930_parent venv:$ISO_ROOT/venvs/14930_fix hf:Qwen/Qwen2.5-1.5B-Instruct
bash harness_iso14930/build_workspace.sh tasks/iso14930 && bash ops_iso14930/calib.sh $GPU
bash ops_iso14930/score_job.sh $GPU fix1 tasks/iso14930/calib/oracle_stage
```

Expected output of each step, the canaries, the torch-only rewrite V1 and the pool re-screen are in `package/RUN.md`.

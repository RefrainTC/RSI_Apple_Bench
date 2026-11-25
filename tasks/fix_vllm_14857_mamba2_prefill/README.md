# vLLM #14857: Mamba2 prefill state updates without per-sequence syncs

In vLLM 0.8.2's Mamba2 mixer, the prefill branch zeroes and writes back SSM states in Python loops indexed by GPU
tensors: one GPU->CPU sync per sequence per Mamba layer. The human fix makes a prefill-heavy Bamba-9B job **1.0648x**
faster on an RTX 4090 (7 jobs on 4 cards and two servers, band ±0.5 %, A/A at most 0.33 %). Outputs are bit-identical.
There is no agent-facing harness (no gate, audit, witness or canaries). #14778, the first landing of the same change
(later reverted), is packaged too; it is a near-duplicate, not an independent sample. An independent tester reproduced
#14857 (PASS-with-fixes).

Terms used below:
- **parent**: the unmodified tree, commit `0032903a`. **fix**: the human fix, commit `296f927f`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several scoring jobs. **Band**: the range around REF that counts as a
  tie.
- **Gate**: the checks a candidate must pass to score at all; here they are only reported, not enforced.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (Threadripper PRO 5975WX, one 8-core CCD pinned per GPU),
where the task was calibrated. "The 2-GPU server" is a second RTX 4090 server.

## What is slow, and where does this task come from?

In `MambaMixer2.forward_cuda`'s prefill branch the parent (a) zero-initialises fresh SSM states with
`for idx in state_indices_tensor[~has_initial_states]: ssm_state[idx].zero_()` behind `any(has_initial_states)`, and
(b) writes the final states back with `for i, idx in enumerate(state_indices_tensor): ssm_state[idx].copy_(...)`.
Iterating a CUDA tensor and indexing with the resulting 0-d CUDA tensors is a GPU->CPU sync per element, so every
Mamba layer pays O(sequences) syncs and small launches per prefill step. The fix ("RE: Mamba2 Prefill Performance
Tweaks", the re-land of #14778) uses `torch.any` and one indexed assignment for each (1 file, 6 lines).

The PR comes from ISO-Bench (Lossfunk, https://github.com/Lossfunk/ISO-Bench; dataset `Lossfunk/ISO-Bench` on Hugging
Face), screen batch 3, written prior P = 0.35 (#14778: 0.30). ISO-Bench served ShareGPT; this load is chosen to put the
per-sequence syncs on the critical path.

## What is the load and the score?

- Model: ibm-ai-platform/Bamba-9B (ISO-Bench's model), weights quantised to FP8 at load (`quantization="fp8"`; bf16 is
  19.6 GB and does not fit at 0.85 of 24 GB). `max_num_seqs` 64 (each sequence holds ~60 MB of Mamba state),
  `max_model_len` 2048, `gpu_memory_utilization` 0.85, seed 0, CUDA graphs for decode.
- Offline `LLM.generate`: 512 prompts of 64 random token ids, greedy, 8 output tokens (`ignore_eos`): prefill-dominated,
  up to 32 sequences per prefill step (2048 batched tokens). One untimed 8-prompt warm-up, then 5 timed calls, each on a
  new seeded prompt set (seeds 1-5).
- **Metric: median wall time of the 5 calls.** Score: median of 3 ABBA pairs after 2 warm-up runs.

## How much does the known fix gain?

Parent median rep 5.69-5.80 s, fix 5.34-5.45 s (~0.35 s per rep).

| Task | GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|---|
| #14857 | RTX 4090 (4 cards, two servers) | **1.0648** (median; range 1.0614-1.0656), calibration 1.063 (4-GPU server GPU0, 5 pairs, 1.0625-1.0664) | ±0.5 % (1.0595-1.0701), SD 0.14 % | 0.33 % (5 pairs, 0.9991-1.0033) | 7 |
| #14778 (duplicate) | RTX 4090 (4 cards, two servers) | 1.0365 (median; range 1.0334-1.0369), calibration 1.033 (GPU1) | ±0.5 % (1.0313-1.0417), SD 0.14 % | 0.31 % | 5 |

- #14857 jobs: 4-GPU server GPU0 x2, GPU2 x2, GPU3 x1 (range 1.0614-1.0656); 2-GPU server GPU0 1.0645, 1.0648. No host
  offset in the ratio. Absolute times depend on the card (fix 5.33-5.38 s on GPU0, 5.44-5.45 s on GPU3) and the parent
  moves with it.
- Single slow parent reps exist (5.96 s, 6.21 s against 5.69-5.81 s), so two jobs each had one pair at 1.106 / 1.138;
  the median of pairs absorbs them (job REFs 1.065 / 1.0645).
- Transfer loads (4 pairs each): 256 prompts x 256 tokens 1.020x; 64 output tokens 1.019x; `max_num_seqs` 16 1.033x.
  Longer prompts and decode steps dilute a fixed per-sequence prefill cost.
- RTX 5090: does not run. torch 2.6.0+cu124 has no sm_120 code, and vLLM's own kernels stop at sm_90. The same holds
  for #14778 (torch 2.5.1+cu124).

## How are outputs checked?

`package/b3/scripts/cal_analyze.py` reports, per job:
1. **Correctness pass**: 24 fixed text prompts in 3 `generate()` calls of batch 16, 1 and 7, 48 greedy tokens, top-5
   logprobs. Parent vs parent and **fix vs parent: 24/24 identical token ids in all 8 jobs** (calibration + 7 scoring).
2. **Every timed run**: output token counts and the sha256 of all output token ids **identical to the paired parent in
   every rep of every pair** (calibration 25/25, scoring jobs 15/15 each); A/A also 25/25. Both versions are
   bit-deterministic here, so a candidate can be held to exact equality.

Not built: step trace, settings readback, gate, audit hook, timing witness, integrity manifests, canaries.

## Which broken or cheating versions are caught?

None were built or tested; no canaries exist for this task or for #14778.

## What are the limits?

- (a) **FP8 online quantisation** is a load deviation from ISO-Bench's bf16 serving; the changed code is the same.
- (b) **Exact output equality rejects legitimate numeric changes** (another state layout, a fused kernel).
- (c) **Memory headroom is small**: the per-run peak of 22.1 GB leaves ~2 GB, so a candidate that adds memory at load
  could hit the 97 % watchdog for reasons unrelated to speed.
- (d) **No agent-facing harness.** The benchmark runs inside the candidate's process; a candidate tree is scored by
  putting it in the fix venv's slot (`instruction.md`), a path the authors have not exercised with any other tree.
- **#14778 is not an independent sample.** It is the first landing of the same change, with `torch.vmap` over a
  lambda that writes rows in place; it was reverted upstream and re-landed as #14857. Its task uses **dummy weights and
  only the first 16 of 32 layers in bf16**, because at that vLLM version the FP8 load crashes at model construction.
  The tokens are meaningless but deterministic; its REF is specific to that synthetic load and not comparable with
  #14857's. On this load the vmap version did not crash and matched the parent bit for bit, so the revert reason did not
  show (not investigated).
- **Setup must run with `TZ=UTC`.** uv reads the lock's dependency cutoff (`--exclude-newer`) in local time; on a machine
  at UTC+4 the #14857 build silently lost fastapi 0.115.12, uploaded late on the cutoff day. The packages are left
  unchanged so that their file manifests still match.

## What does it cost to run?

| Step | Time |
|---|---|
| Build both venvs | 362 s in the authors' empty-directory test (~8 min) |
| Model download (19.6 GB) | 342 s |
| Stage to local NVMe (~33 GB) | 1108 s from a busy NFS; 7-18 min |
| Calibration (3 correctness passes, 2 warm-ups, 5 fix pairs, 5 A/A = 25 runs) | 24.5 min wall (1456-1534 s of GPU job time) |
| One scoring job (2 warm-ups, 1 correctness pass, 3 pairs = 9 runs) | ~9.5 min of GPU time (521-589 s); ~18 min with the 3 transfer loads |

One timed run is ~48-50 s of job time, ~27-29 s of it timed. Peak VRAM 22.1 GB (22,122 MiB, 90 % of the card), above
vLLM's own 0.85 budget because it loads bf16 weights before quantising. Disk: ~42 GB under the package directory,
~33 GB on the staging disk. #14778: ~7 min per scoring job, calibration 20.5 min, peak 21.2 GB, model download only
config + tokenizer (8.8 MB).

## Has someone else rebuilt it?

#14857: yes. An independent tester followed the card from scratch: **PASS-with-fixes**. Calibrations 1.0643 (4-GPU
server) and 1.0647 (2-GPU server); four scoring jobs 1.0649, 1.0681, 1.0638, 1.0628; all six median 1.0645, SD 0.17 %,
all inside the band. A/A 0.25 % on the 2-GPU server; on the 4-GPU server one calibration A/A pair read **1.28** from a
whole-host stall while three other setups built venvs and copied 7-20 GB, which shows that the median of 5 reps does
not protect a run in which 3 of 5 reps are hit; only the job's median of 3 pairs does. Transfers 1.022 / 1.019 / 1.031;
outputs identical in every rep; peak 22,122 MiB; setup ~19 min. Fixes needed: scripts only in an untracked mirror with
hard-coded paths; no full SHAs or merge date; the parent build fails without the `xgrammar==0.1.17` override (0.1.16 is
no longer on PyPI). The package fixes these.

The authors then ran both packages from an empty directory (freezes and all engine files identical to the authors'
builds): #14857 calibration REF 1.0675 (A/A 0.17 %), scoring job **1.0621**; #14778 calibration REF 1.0337 (A/A
0.20 %), scoring job **1.0344**. #14778 has had no independent rebuild.

## How to run it

```
cp -r package /data/iso14857 && cd /data/iso14857 && export ISO_ROOT=$PWD STAGE=/tmp/iso14857 UV=uv TZ=UTC
bash b3/scripts/setup_lock.sh && bash b3/scripts/get_model.sh && bash b3/scripts/pkg_stage.sh
bash b3/scripts/b3_caldrv.sh 0 14857 cal:cal_g0 score:s1_g0
python3 -I b3/scripts/cal_analyze.py b3/cal/s14857 b3/cal/s14857/cal_g0/cor_1_parent cal_g0 s1_g0
```

Every step's expected output is in `package/RUN.md`. The #14778 package (`iso14778`) runs the same way with `14778` in
place of `14857`.

# SGLang #6994: LoRA batch metadata without per-request syncs

SGLang 0.4.7 fills its LoRA batch metadata on the GPU request by request, with a GPU->CPU sync for each request in
every decode step. The human fix makes a batch-64 LoRA decode **1.474x** faster on an RTX 4090 (8 jobs on 4 cards and
two servers, band ±1.1 %). The task has no agent-facing harness (no gate, audit, witness or canaries), and its timed
outputs are not reproducible even between two parent runs, so the output check is weak. An independent tester
reproduced the numbers (PASS-with-fixes).

Terms used below:
- **parent**: the unmodified tree, commit `777688b8`. **fix**: the human fix, commit `021f76e4`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several scoring jobs. **Band**: the range around REF that counts as a
  tie.
- **Gate**: the checks a candidate must pass to score at all; here they are only reported, not enforced.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (Threadripper PRO 5975WX, one 8-core CCD pinned per GPU).
"The 2-GPU server" is a second RTX 4090 server, where the calibration ran.

## What is slow, and where does this task come from?

In `LoRAManager.prepare_lora_batch` the parent fills the adapter index, rank and scaling tensors on the GPU **request
by request**: `weight_indices[i] = ...` is a small host-to-device write, and `lora_ranks[weight_indices[i]] = r`
indexes with a 0-d GPU tensor, which is a GPU->CPU sync. The eager (non-graph) branch also syncs on
`int(torch.max(seg_lens))`. The fix builds the three lists on the CPU, copies them once from pinned memory with
`non_blocking=True`, takes `max_len` from the CPU copy, and fills the constant CUDA-graph `seg_lens` / `seg_indptr` once
at start-up (2 files).

The PR comes from ISO-Bench (Lossfunk, https://github.com/Lossfunk/ISO-Bench; dataset `Lossfunk/ISO-Bench` on Hugging
Face), screen batch 3, written prior P = 0.30; 3 of 8 screened PRs passed (expected 2.8 of 12 listed). It is the first
SGLang task we built. ISO-Bench used gated Llama-3.1-8B with a public adapter; FP8 bases are not LoRA-compatible, so
we used a small open base.

## What is the load and the score?

- Model: Qwen/Qwen2.5-1.5B-Instruct (bf16) + two random-weight LoRA adapters made locally (r 16, alpha 32, all 7
  projections, B ~ N(0, 0.002), seeds 0 and 1). Engine: `sglang.Engine`, triton attention and LoRA backends, pytorch
  sampling, radix cache off, CUDA graphs up to batch 64, `max_loras_per_batch` 4, `mem_fraction_static` 0.80, seed 0.
- Offline `Engine.generate`: 64 requests of 256 random token ids, all on adapter `lora0`, greedy, 256 output tokens
  (`ignore_eos`). One untimed 8-request warm-up, then 5 timed calls, each on a new seeded prompt set (seeds 1-5).
  Decode runs at batch 64 inside the CUDA-graph branch, the path the PR targets.
- **Metric: median wall time of the 5 calls.** Score: median of 3 ABBA pairs after 2 warm-up runs.

## How much does the known fix gain?

Parent median rep 3.58-3.67 s, fix 2.42-2.48 s: ~1.2 s per rep, ~4.6 ms per decode step.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4 cards, two servers) | **1.474** (median; range 1.465-1.485), calibration 1.475 (2-GPU server GPU0, 5 pairs, 1.468-1.478) | ±1.1 % (1.458-1.490) = max(0.5 %, 2 SD), SD 0.54 % | 1.15 % (5 pairs, calibration: 0.9885-0.9992) | 8 |

- Jobs per card: 2-GPU server GPU0 1.466, 1.471; 4-GPU server GPU3 1.477, 1.472, 1.465; GPU1 1.485, 1.479; GPU2 1.485.
  Part of the spread is a card offset (up to ~1.3 %) from the fix side: on GPU1 the fix runs 1.7 % faster (2.42 vs
  2.46-2.48 s) while the parent moves less. Spread scoring jobs over cards.
- Transfer loads (4 pairs each, two servers): batch 16 **1.40x**; batch 128, above the CUDA-graph limit, so the eager
  branch with its `torch.max` sync, **1.17x**; two adapters alternating per request 1.475x. The cost is per request
  per step, so it does not depend on the number of adapters.
- RTX 5090: does not run. torch 2.7.1+cu126 has no sm_120 code, and sgl-kernel 0.1.7 has machine code only up to
  sm_90. A torch 2.7.1 cu128 wheel exists, but sgl-kernel would still need a source build.

## How are outputs checked?

`package/b3/scripts/cal_analyze.py` reports, per job:
1. **Correctness pass**: 24 fixed text prompts (20-181 words) in 3 `generate()` calls of batch 16, 1 and 7, 48 greedy
   tokens, top-5 logprobs, on `lora0`. Parent vs parent 24/24 identical token ids; **fix vs parent 24/24 identical in
   all 9 jobs** (calibration + 8 scoring).
2. **Every timed run**: identical output token counts per rep (16384 per rep) vs the paired parent.
3. **Text sha per rep**: reported only. It is **not bit-reproducible run to run**, even parent vs parent (A/A 20/25
   reps identical); fix vs paired parent 94/145 reps (65 %). Not resolved at token level. Likely cause: SGLang's
   scheduler receives the 64 requests in timing-dependent groups, so prefill batches differ and random-token prompts
   hit near-ties; a small numeric difference cannot be excluded from these data. Against it: in the screen (one fixed
   prompt set), all 36 reps of the 12 parent and fix runs had the same output sha.

Not built for this task: step trace, settings readback, gate, audit hook, timing witness, integrity manifests,
canaries.

## Which broken or cheating versions are caught?

None were built or tested. No canaries exist for this task. A cheat that changes numerics slightly, or the prompt
arrival order, would pass the checks above (limit b).

## What are the limits?

- (a) **The adapter weights are random.** The speed path does not depend on their values, but outputs are not those of
  a real fine-tune.
- (b) **The rep-level output check can only bound "same length, same text in most reps".** A candidate that changes
  numerics slightly would pass it, and so would one that changes the prompt arrival order.
- (c) **Card offset** ~1.3 % between cards, larger than the A/A noise: use the multi-job REF 1.474 with ±1.1 % and keep
  scoring jobs spread over cards.
- (d) **No agent-facing harness** (gate, audit, witness, vault). The benchmark runs inside the candidate's process, and
  a candidate tree is scored by putting it in the fix venv's slot (`instruction.md`), a path the authors have not
  exercised with any tree other than the human fix.
- flashinfer-python 0.2.6.post1 is left out of the build (an sdist that needs nvcc). The task uses the triton
  attention backend and pytorch sampling, which do not import it.

## What does it cost to run?

| Step | Time |
|---|---|
| Build both venvs (from source tarballs) | 352 s in the authors' empty-directory test; ~7 min |
| Model + adapters | 96 s |
| Stage to local NVMe | 911 s from a busy NFS (~17 GB); 3-4 min otherwise |
| Calibration (3 correctness passes, 2 warm-ups, 5 fix pairs, 5 A/A = 25 runs) | 13.5 min wall (797-803 s of GPU job time) |
| One scoring job (2 warm-ups, 1 correctness pass, 3 pairs = 9 runs) | 4.5 min of GPU time (269-282 s); 8.4 min with the 3 transfer loads |

One timed run is 28.6 s (fix) to 34-35 s (parent) of job time, ~12 s / ~18 s of it timed. Peak VRAM 20.9 GB (0.85 of
the card; 20,752 MiB in the package test). Disk: ~25 GB under the package directory, ~17 GB on the staging disk.

## Has someone else rebuilt it?

Yes. An independent tester followed the card from scratch: **PASS-with-fixes**. Calibration REF 1.4749 (A/A 0.62 %);
five scoring jobs on 3 cards and 2 servers, 1.4803, 1.4860, 1.4898, 1.4800, 1.4885; all six median 1.4831, SD 0.39 %,
every job inside the band but the median 0.6 % above the card's. Transfers 1.402 / 1.161 / 1.478; correctness 24/24 in
all jobs; text sha identical in 69/100 fix-vs-parent and 18/25 A/A reps; peak 20,780 MiB; setup ~11-12 min. Fixes
needed: scripts only in an untracked mirror with hard-coded paths; no build commands, full SHAs, merge date or
adapter recipe; staging needed an interpreter copy made by an unmentioned script. The package fixes these.

The authors then ran the package from an empty directory: freeze and all 696 `sglang/` files identical to the authors'
build, adapters sha256-identical; calibration REF 1.483 (A/A 0.58 %), scoring job **1.4764**, correctness 24/24. Two
build-script defects were fixed on the way (a file-count check that tripped on file names with spaces; a log that
`tee /dev/stderr` truncated).

## How to run it

```
cp -r package /data/iso6994 && cd /data/iso6994 && export ISO_ROOT=$PWD STAGE=/tmp/iso6994 UV=uv TZ=UTC
bash b3/scripts/setup_lock.sh && bash b3/scripts/get_model.sh && bash b3/scripts/pkg_stage.sh
bash b3/scripts/b3_caldrv.sh 0 6994 cal:cal_g0 score:s1_g0
python3 -I b3/scripts/cal_analyze.py b3/cal/s6994 b3/cal/s6994/cal_g0/cor_1_parent cal_g0 s1_g0
```

`TZ=UTC` avoids a known setup issue: uv reads the lock's dependency cutoff date in local time, and on a machine east of
UTC one package's (#14857) build lost a wheel. Every step's expected output is in `package/RUN.md`.

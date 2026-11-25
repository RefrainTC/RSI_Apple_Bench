# vLLM #37338: Qwen3.5 cold start, GDN warm-up with the wrong dtypes

This is our first **cold-start** task: the score is the time from launching the Python process to the first result,
with every compile cache empty. At the parent commit the first real prefill autotunes and compiles Triton kernels
that the warm-up should have covered (~46 s on a 4090). The human fix is **1.29x** faster on an RTX 4090 and 1.18x on
an RTX 5090 workstation, where the CPU sets the time. An independent tester rebuilt the package on a second server and
got the same result: PASS, no deviations.

Terms used below:
- **parent**: the unmodified tree, commit `54ab804e` (vllm 0.18.1rc1.dev37). **fix**: the human fix, commit
  `a16133a0` (dev38).
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **Band**: the range around REF that counts as a tie
  (`ties_ref`).
- **Gate**: the output and integrity checks. A candidate that fails any of them scores zero.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs, where this task was built and calibrated. "The 2-GPU
server" is a second RTX 4090 server, used by the independent tester and for the earlier screen. Both run driver
550.127.08 (CUDA 12.4) on a Threadripper PRO 5975WX. The RTX 5090 workstation has a Ryzen 9 3950X.

## What is slow, and where does this task come from?

Qwen3.5's gated-delta-net (GDN) layers run Triton kernels with `@triton.autotune`, and Triton's autotune cache is keyed
by the argument dtypes as well as the listed keys. vLLM warms these kernels up at start-up so that the first real
prefill does not autotune. But the parent's warm-up built `g` and `beta` with `torch.randn` in bf16 and `cu_seqlens`
as int64, while a real prefill passes fp32 `g` / `beta` and int32 `cu_seqlens`. So the warm-up tuned variants no real
batch uses, and the first real prefill autotuned and compiled the right ones: on a 4090 ~46 s and ~100 extra kernels.

The PR was merged on 2026-03-23. We picked it from our list of upstream fixes mined from SGLang and vLLM history
(batch 4, prior probability of a stable gain 0.45). The first screen kept the per-role Triton cache between jobs and
saw the gain only once (a 44x first batch), then nothing: once the cache held the tuned kernels, the parent no longer
paid. The gain is a cold-start gain, so the task empties every compile cache per run, and it scores start-up and the
first batch together so that cost moved into start-up cannot score.

## What is the load and the score?

- Model: Qwen/Qwen3.5-4B (bf16, text-only use). `vllm.LLM`, V1 offline, `max_model_len` 4096, `max_num_seqs` 16,
  `gpu_memory_utilization` 0.85, seed 0, defaults otherwise (torch.compile and CUDA graphs on).
- One Python process per run, all compile caches empty (Triton, Inductor, vLLM torch.compile, CUDA JIT, FlashInfer,
  HOME / XDG: new directories per run, deleted afterwards).
- `LLM(...)`, then the first `generate()`: 8 prompts x 512 random token ids (seed 1) -> 8 greedy tokens; then 3 steady
  reps of the same shape (seeds 2-4).
- **Metric: cold time to first result** = wall clock when the first `generate()` returns - the harness's launch stamp
  of the process. Secondary: `LLM()` time, first-batch time, steady reps, whole-process wall time.
- Score: parent / candidate per pair, median of 3 ABBA pairs. The harness also reports
  reward = min(1, (S - 1) / (REF - 1)).

## How much does the known fix gain?

Cold time to first result: fix 161.8-172.4 s (mostly 162-165 s), parent 208.9-218.4 s, ~47 s saved. Start-up
(`LLM()`) is ~157-164 s in both arms; the fix's warm-up, now compiling the right kernels, costs it 0-5 s more. The
first batch alone goes from 48.0-51.2 s to 1.11-1.40 s (36-44x).

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server, GPUs 0-3) | **1.29** (median of the first 8 jobs 1.2901, of all 10 1.2920; range 1.269-1.297) | ±2 % (1.264-1.316); 2 SD = 1.4 %, SD 0.68-0.72 % | 0.9 % (calibration), 2.5 % (package test, one parent start-up 4 s slow) | 10 |
| RTX 5090 workstation | **1.178** (median of 3 jobs: 1.176, 1.178, 1.183) | not set | 0.57 %; an A/A scoring job read 1.003 | 3 |

- The 10 jobs: calibration 1.2929 (GPU2); scoring 1.2691 and 1.2911 (GPU3), 1.2966 (GPU2), 1.2803, 1.2891 and 1.2962
  (GPU0), 1.2882 (GPU1); package test calibration 1.2969 and scoring 1.2951 (GPU2).
- Why the band floor is 2 % rather than 1.5 %: single cold start-ups occasionally take 5-10 s longer (one fix run at
  172 s). With 1.5 % the first scoring job, 1.2691, would sit 0.1 % below the band.
- Two later scoring jobs (1.3005, 1.3070) are `ties_ref`; over all 12 jobs the median is 1.294 and the SD 0.76 %, so
  the gate was left as it is.
- Second-server check of the screen (first-batch metric, Triton cache cold only): 37.5x and 34.5x, median 36.0x,
  -0.7 % from the 2-GPU server's 36.3x.
- Transfer loads (two jobs of one pair each): first batch 1 x 128 tokens 1.29x and 1.29x; 16 x 2,048 tokens 1.36x and
  1.29x; **warm caches 0.87x and 1.00x**. With warm caches the PR changes nothing (the 0.87x pair was one fix start-up
  4.5 s slower, not repeated).
- RTX 5090: the package runs unchanged. The cold start is compile work on the CPU, and that host's CPU is slower:
  each cold run is ~50 % longer (parent ~320 s, fix ~272 s). The fix saves about the same absolute time (~48 s of
  first-batch compile), so the ratio drops to 1.18. `package/harness/gate.json` still holds the 4090 REF. The A/A
  scoring job there reads `fail:venv`, an artefact: the harness writes the venv snapshot only for a real candidate
  venv.

## How are outputs checked?

Outputs are almost reproducible: first-batch plus steady-rep digests match the paired parent's in 1-4 of 4 per pair,
parent vs parent in 3-4 of 4. Each cold run autotunes afresh, and different kernel variants flip near-ties. The gate
(`package/harness/check.py`, thresholds in `package/harness/gate.json`):

1. **Counts** of the first batch and every steady rep equal the paired parent's.
2. **Settings**: the engine's resolved configuration (model, scheduler, cache, compilation config incl. CUDA-graph
   sizes) equals the parent's.
3. **Teacher-forced re-read** of every output by the parent engine (`prompt_logprobs` over prompt + output): gap <=
   **1.0 nat** for every token and <= **2 %** of tokens with gap > 0.5. Calibrated on 44 parent and fix runs (+ 6
   transfer runs): max gap 0.0-0.19 nats, no token above 0.5; tau is 5x the worst.
4. **Audit** of the candidate venv before the job (`package/harness/audit.py`): only `.py` files may differ (vLLM's own
   dist-info and plain RECORD files excepted); added lines may not mention cache locations, carry blobs (base64, zlib,
   marshal, pickle, lines > 2,000 characters, > 200 kB in total) or write environment variables.
5. **Venv unchanged**: a manifest (path, size, mtime of every file, bytecode included) before and after each candidate
   run must be equal.
6. **Cold witness**: each candidate run must compile >= **100** Triton kernels into its own empty cache (parent 389,
   fix 289).
7. **Steady bound**: no steady rep slower than 1.5x the paired parent's slowest + 0.2 s.
8. **After bound**: first result to process end <= 1.25x the parent's + 5 s.

Checks 4-8 exist because the score is a cold start: they stop a candidate from bringing compiled state, keeping it
between runs, or moving the cost to a place the score does not see.

## Which broken or cheating versions are caught?

Python-only edits (`package/variants/apply.py`), each in a `quick` job (1 pair + teacher-forced check) on the 4-GPU
server. Every variant faster than the parent is rejected; two probes pass but gain nothing.

| Variant | What it does | Speedup (1 pair) | Verdict |
|---|---|---|---|
| `shipcache` | parent + the complete compile caches of one parent run shipped inside the venv | **3.35x** | `fail:audit,venv,cold` |
| `persist`, first job | parent + cache variables pointed into the venv; the first run fills it | 1.00x | `fail:audit,venv,cold` |
| `persist`, second job | same venv, cache now warm | **5.82x** | `fail:audit,venv,cold` |
| `persisttmp`, first job | parent + cache variables pointed at a fixed directory under `/tmp` | 1.00x | `fail:audit,cold` |
| `persisttmp`, second job | same, cache warm | **5.83x** | `fail:audit,cold` |
| `short1` | fix + the first `generate()` returns at most 4 tokens per request | 1.29x | `fail:counts` |
| `noautotune` (probe) | parent + every `Autotuner` in `fla/ops` keeps only its first config | 1.00x | passes, `no_gain`: the patch did not reach the kernels that autotune at the first batch |
| `gbf16` (probe) | fix + the GDN decay `g` rounded to bf16 | 1.29x | **passes** (max gap 0.0): no argmax moves in 8-token outputs, and no speed gained |

## What are the limits?

- **No filesystem sandbox.** A candidate can write anywhere the job user can. The audit catches the plain ways to keep
  or bring compiled state, the venv snapshot catches writes into the venv, and the cold witness catches runs that
  compile little. A candidate that hides its cache paths from the audit (string tricks) **and** copies a cache kept
  outside the venv into the run's fresh cache directories would pass all three. The robust fix is a container or
  bubblewrap sandbox where only the fresh directories and the output directory are writable; not done.
- The teacher-forced check cannot see small precision losses that do not change the argmax over 8-token outputs
  (`gbf16` passes). On this load such changes also bring no speed.
- The score includes ~157 s of start-up that the fix does not touch, so a candidate can also win by shortening
  start-up. Fewer CUDA-graph sizes would change the settings and fail; faster compile or loading code is legitimate.
  That is intended: the task is the cold start as a whole.
- Runs are ~3.5 min each, so a scoring job costs ~24 min of GPU time, about 3x a warm-load task.
- The first run of a job after switching tasks can be a few % slower (cold page cache for 9 GB of weights); the
  parent warm-up run absorbs most of it. Cold-start tasks need a quiet host (see acceptance).
- The score depends on the host CPU as much as on the GPU (1.29 on the 4090 servers, 1.18 on the 5090 workstation).

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory (build, model, stage) | ~14 min in the authors' test, ~18 min in the independent test |
| One scoring job (1 warm-up, 3 pairs, teacher-forced check ~2 min) | ~24-27 min (1,390-1,580 s over 8 jobs); ~36 min on the RTX 5090 |
| Calibration (2 warm-ups + 20 runs + teacher-forced check) | ~78 min (4,684 and 4,699 s); 113 min on the RTX 5090 |
| Broken-variant `quick` job | ~5-8 min |
| Peak VRAM | 21.5 GB on the 4090. RTX 5090: timed runs 27.9 GB, teacher-forced pass 29.1-30.1 GB |
| Disk | ~35 GB under the package, ~30 GB on local disk, + ~0.4 GB of compile caches per running job |

One run is 166-168 s (fix) or 212-222 s (parent) on the 4090; ~2 GB of fresh compile caches are written and deleted
per run. On the RTX 5090 the teacher-forced pass leaves only ~0.6 GB of margin under 30 GiB; for routine use there,
lower the cap the bench puts on that pass (0.6 -> 0.5). No CUDA toolkit is needed.

## Has someone else rebuilt it?

Yes, on a second server. An independent tester copied the package alone into an empty directory on the 2-GPU server
and followed `package/RUN.md` without edits. Verdict **PASS, no deviations**.

- Build: freeze (175 packages) and 2085/2085 vllm files in each arm equal the authors'.
- Calibration REF 1.291, first-batch ratio 36.5-44.4x. A/A max 4.25 % from one pair: that parent's `LLM()` start-up
  was 8 s slower, while the tester was creating a 13 GB variant venv and cold-starting it on the other GPU. The other
  four pairs are <= 0.95 %; the REF is unaffected.
- Scoring jobs 1.2829, 1.2926 (GPU0) and 1.2915 (GPU1), all `ties_ref`; 289 vs 389 kernels compiled; peak 21.5 GB.
- Variants: `short1` 1.278 `fail:counts`; `shipcache` 3.29x, `persist` 0.997 then 4.67x, all `fail:audit,venv,cold`;
  `gbf16` passes, as the card says. `persisttmp` was not run: it writes a fixed directory under `/tmp`.
- Transfers: 1 x 128 1.247, 16 x 2048 1.385, warm caches 1.011 (card 1.29 / 1.29-1.36 / 0.87-1.00).

The tester found the package's manifest stale on 5 files while it was still in progress; the READY version checks
clean and differs from the tester's copy only in documentation and `gate.json`. In the authors' own empty-directory
test, three package files were fixed and copied in during or before its GPU steps: the teacher-forced job's teardown
handling, the teacher-forced re-read of prompts over 1,024 tokens (transfer load only), and the final gate. That test
scored 1.2951. Its fix venv differs from the authors' in one file outside `vllm/` (a top-level `build_backend.py` that
two wheels both ship; uv's install order decides), which nothing imports.

## How to run it

```
cp -r package /data/vllm37338 && cd /data/vllm37338 && cp site.env.example site.env  # set STAGE, GPU, CPUS, PORT
bash setup/setup_lock.sh && bash setup/get_model.sh && bash setup/stage.sh
bash harness/run.sh cal_g0 cal && bash harness/run.sh s1_g0 score                  # calibration, then the fix
```

Every step, the expected output and the cheating-variant commands are in `package/RUN.md`.

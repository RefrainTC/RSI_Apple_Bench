# Phi-Bench vLLM scheduler task on an RTX 5090 and an RTX 4090

Phi-Bench's `vllm-scheduler-mixed-batch-serving`, written for one H20, runs on both of our GPUs once the image base
and REF are changed. The task's oracle stands far above A/A. But two things make its score fragile, and they lead
this card: **REF depends on host load** (an independent tester on a quieter host read 1.34 / 1.55 where we read
1.36 / 1.61), and the oracle is **not the strongest config-only launch**, so REF must come from a per-GPU sweep.

## 1. What is slow, and where does this task come from?

Source: Phi-Bench (Apache-2.0), https://github.com/one2piece2hello/faibench_LLM_Infra_Bench.git at commit `90d7251b`,
`tasks/e2e/vllm-scheduler-mixed-batch-serving/task`. vLLM 0.10.1.1 serves Qwen2.5-1.5B-Instruct. Concurrent bursts
that mix short requests, long prefills and long decodes take longer than the GPU needs. The candidate owns
`/app/submission/launch_server.sh` and may patch vLLM at launch.

**What we changed** (both baseline and candidate get the same):

| Change | Why |
|---|---|
| 5090 image: `nvidia/cuda:12.8.1` base, `TORCH_CUDA_ARCH_LIST=12.0`, cu126 → cu128 wheels (`environment/Dockerfile.5090`, `diffs/dockerfile_5090.diff`, 4 lines) | sm_120 needs CUDA 12.8 |
| `--gpu-memory-utilization 0.88 → 0.85` in the 3 launch scripts (`diffs/gpu_mem_085.diff`) | at 0.88 the 5090 peaked at 29,972 MiB, 748 MiB under our 30 GiB line. KV capacity is not the bottleneck at either value |
| Container memory 96 → 32 GB (5090) | host has 62 GB; measured peak 3.3 GiB |
| An identity (A/A) candidate: `tests/launch_baseline.sh` with its comment lines removed (`environment/scripts/identity_launch_server.sh`) | the original trips the source scan (`forbidden_edit_path`) because a comment names the reward manifest |
| 4090 only: no Docker; a host venv with the image's version anchors; `test.sh` with its two `pkill` lines made no-ops and a settable verifier dir (`diffs/task_4090.diff`) | the 4090 server has no usable Docker and is shared |
| REF recalibrated per GPU | the H20 constant (1.2857) gives the oracle alone 0.21-0.26 (5090) and 0.31-0.32 (4090) |

Scoring code (`tests/compute_reward.py`), prompts and the reward manifest are unchanged.

## 2. What is the load and the score?

Each scoring is one full `tests/test.sh` in a fresh container: the baseline (Phase 1) serves 3 sequential workloads
and 6 bursts and records 1166 reference outputs; the candidate (Phase 2) answers the same 1166 prompts, passes a
decode probe and is timed; the baseline is re-measured (Phase 3). The 6 bursts: `256_short_heavy_mix`,
`128_decode_heavy_mix`, `192_decode_heavy_mix`, `256_decode_heavy_mix`, `96_all_longprefill`, `160_all_longprefill`.
Speedup = median over the bursts of baseline/candidate makespan. Reward = clamp(ln(s/REF)/ln(REF), 0, 1): 0 at REF.

## 3. How much does the known fix gain?

**A/A** = the identity candidate (the baseline scored against itself). **Oracle** = the task's own launch script.
**Sweep REF** = the strongest config-only launch we found (`solution/README.md`).

| GPU | Oracle (5 scorings) | A/A (5 scorings) | Sweep REF (3 scorings) |
|---|---|---|---|
| RTX 5090, our host (load 2.8-4.4) | 1.3639, 1.3737, 1.3601, 1.3548, 1.3633 (median 1.3633) | 1.0014, 1.0053, 1.0021, 0.9955, 1.0027 | 1.6009, 1.6120, 1.6088 (median **1.6088**) |
| RTX 5090, tester's host (load 1.4-1.9) | 1.3485, 1.3443, 1.3327 (median 1.3443) | 1.0033, 1.0060 | 1.5595, 1.5478 |
| RTX 4090 | 1.3931, 1.3914, 1.3924, 1.3942, 1.3927 (median 1.3927) | 1.0024, 1.0002, 1.0005, 0.9998, 1.0000 | 1.6477, 1.6464, 1.6492 (median **1.6477**) |
| H20 (task authors) | 1.2857 | - | - |

- On the 5090 the oracle sits 0.361 above A/A, 37 times the A/A range; on the 4090, 0.3925, 154 times.
- Our suggested band on the 5090 was ±0.02 on the speedup (about ±0.05 reward near REF). **The tester's runs fell
  outside it**: all three oracle runs were below our minimum, and the sweep config read 3.1-3.8% low.
- Per burst (oracle, 5090 / 4090): short_heavy_256 1.126 / 1.099, decode_128 1.610 / 1.531, decode_192 1.322 / 1.337,
  decode_256 1.307 / 1.339, longprefill_96 1.497 / 1.546, longprefill_160 1.398 / 1.445.

## 4. How are outputs checked?

The task's own gates: greedy token-by-token match on 1166 prompts (≥ 0.95; the oracle matched 1166/1166 on the 5090
and 1163/1166 on the 4090); a decode probe (same prompt, 8 vs 96 tokens, latency ratio ≥ 1.5; oracle 8.8-10.9);
pinned checks; a sequential-latency gate (≥ 0.85 of baseline); a source scan of the submission. The client is not the
bottleneck: running the burst client in 4 or 8 processes changed makespans by −8% to +8.5%, mostly within ±3%, with
no consistent direction.

## 5. Which broken or cheating versions are caught?

| Variant | GPU | Speedup | Verdict |
|---|---|---|---|
| negative (fake server, fixed text) | 5090 / 4090 | - | caught: `correctness_failed`, 0/1166 match |
| starter (the task's starting script) | 5090 / 4090 | 0.7249 / 0.7374 | 0: `speedup_not_above_baseline` |
| identity (A/A) | both | 0.9955-1.0053 | 0, with reason `zero_without_named_reason` or `speedup_not_above_baseline` depending on which side of 1 the noise falls |
| config-only launches, against the oracle REF | 4090 | up to the sweep REF | earned reward 0.25-0.51; against the sweep REF, about 0 |

## 6. What are the limits?

- **REF depends on host load.** The baseline is bound by the CPU-side scheduler, so a quieter host speeds it up more
  than the candidate (baseline bursts 3.5-6.4% faster, oracle 2.7-5.2%, sweep config −1.9% to +0.2% on the tester's
  host), and ratios shrink. Measure REF on the host and at the load you will score at, and record the load.
- **The prefix cache is never flushed.** `compute_reward.py` POSTs `/flush_cache`, which vLLM does not have (404,
  swallowed). Every request in a class has the same text, so the long-prefill bursts are served almost entirely from
  the prefix cache (96-wide burst: 99.7% of prompt tokens hit, 2.07 s; with a unique first line per request: 0.5%,
  7.6 s). These bursts measure admission width and decode step time, not long prefill. With unique long prompts, a
  config at REF level drops from a median 1.62x to 1.18x.
- **Phase 3 re-measures the baseline on candidate-modified vLLM.** Pristine vLLM is restored once, before Phase 1. A
  submission that patches site-packages at launch with default-on switches shifted the Phase-3 baseline by −12.5% to
  +7% per burst in two agent submissions; in 13 other scorings Phase 1 and Phase 3 agree within ±3.4%.
- **Reward semantics disagree inside the task.** The code gives 0 at REF; the manifest and `task.toml` say 0.5. The
  n-gram speculative variants never draft (the model's `repetition_penalty` 1.1 disables it), and async + n-gram does
  not start.
- Configuration rankings differ between GPUs (Spearman 0.69): redo the sweep for each GPU.
- Image dependencies are only partly pinned: a rebuild a few days later differed in 6 of 161 packages. For a strict
  rebuild, turn the pip freeze of the image into a constraints file.
- Agent-facing text still says "H20".

## 7. What does it cost to run?

| | RTX 5090 | RTX 4090 |
|---|---|---|
| Setup | image build 1383-1425 s (`--no-cache`), model download 101 s | host venv (no Docker) |
| One scoring | 21-23 min (oracle median 1290 s, identity 1329 s) | 1408-1499 s |
| Peak VRAM | 29,013 MiB (oracle), up to 29,420 MiB in the sweep, whole card at 0.85 | 21,374 MiB |
| A full calibration | 12 scorings + smoke + client check = about 4.4 GPU-hours | 12 scorings |

## 8. Has someone else rebuilt it?

On the 5090, yes. **The setup works after fixes, but the numbers did not reproduce within our ±0.02**: oracle median
1.3443 (ours 1.3633), sweep config 1.5595 / 1.5478 (ours 1.6088), identity 1.0033 / 1.0060 (matches). All gates
passed; the 4 API-server processes were confirmed; peak 29,188 MiB. The likely cause is host load (section 6). The
Dockerfile diff, the 0.85 diff, the identity script and the 21 base sweep variants regenerated byte-identical. Fixes
made after the test: the task's repository URL and copy steps, path variables, a pinned model revision, `--no-cache`
for rebuilds, and the sweep's `--extra` JSON keys. The 4090 port has not been independently rebuilt.

## 9. How to run it

RTX 5090, Docker. Paths are placeholders; set your own.

```bash
git init -q phibench && cd phibench && git remote add origin https://github.com/one2piece2hello/faibench_LLM_Infra_Bench.git
git fetch --depth 1 origin 90d7251b03f51ebf30a3ffdf0323aed764fd920e && git checkout -q FETCH_HEAD && cd ..
export PHI_TASK=$PWD/task_mem085; cp -a phibench/tasks/e2e/vllm-scheduler-mixed-batch-serving/task $PHI_TASK
cp environment/Dockerfile.5090 $PHI_TASK/environment/ && (cd $PHI_TASK && patch -p1 < $OLDPWD/environment/diffs/gpu_mem_085.diff)
bash environment/scripts/build_image.sh                       # NO_CACHE=1 to force a clean rebuild
python3 -I environment/scripts/download_model.py $PWD/models   # needs huggingface_hub>=0.24,<1
export PHI_MODEL=$PWD/models/Qwen2.5-1.5B-Instruct PHI_RUNS=$PWD/runs
for a in oracle identity oracle identity; do bash environment/scripts/run_score.sh ${a}_$RANDOM $a; done
bash environment/scripts/run_score.sh ref_1 variant:solution/variants/x_s192_a_serve_api4.sh
```

- Read `runs/<label>/verifier/reward.json` and `benchmark_results.json`; `run_meta.json` has peak memory and host load.
- `gpu_mem_085.diff` applies with `patch -p1` from the task root (its paths are `task/...` and `task_mem085/...`;
  use `-p1`). Alternate oracle and identity runs so slow drift lands on both.
- Sweep: `python3 -I environment/scripts/make_variants.py <dir> [--extra name=JSON]` writes `<dir>/variants/*.sh`;
  JSON keys are `source`, `extra`, `effective`, `seqs`, `tok` and optional `"entry": "serve"`.
- Our run scripts were merged into `run_score.sh` (one script for arms and sweep variants) and translated; author
  paths became variables. The 4090 port used a venv instead of Docker; `diffs/task_4090.diff` lists its task edits.

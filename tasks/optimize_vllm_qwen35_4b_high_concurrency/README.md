# Task C: vLLM, Qwen3.5-4B, high concurrency, one GPU

Our own task, built with Apple's recipe for its open optimization tasks and scaled to one 24-32 GB GPU. Stock vLLM
leaves about 31% of request throughput on the table at high concurrency, and two stock flags recover it. The gain is
about 100-190 times the A/A noise on both the RTX 4090 and the RTX 5090. Two cautions lead: the reference value is
**host-specific** (about +1.0-1.3% between two 4090 servers), and it is **partly shape-tuned** (it keeps only about
+7-11% on unseen shapes). No agent trial has been run on it yet.

## 1. What is slow, and where does this task come from?

The recipe is Apple's `optimize_qwen3_6_27b_1xh100_open` (vLLM 0.20.1, Qwen3.6-27B, 1x H100). We kept: hidden
workloads, random-token prompts with a fresh seed per round, prefix caching off, score = geometric mean of
candidate/baseline throughput, and Apple's logprob gate. We changed: the model (Qwen3.5-4B, the same `qwen3_5` GDN
hybrid family), vLLM 0.19.1, our own reference (Apple's `known_good.patch` is a placeholder), and we added a drift
bound, an output witness and a runtime audit. We moved the workloads to high concurrency because on Apple's three
workloads the configurations then allowed moved stock vLLM by only 0.996-1.000.

What is slow: under many concurrent short requests, the default token budget and the fp32 GDN state limit how many
requests the scheduler can run together. The candidate may change serve flags, whitelisted environment variables and
vLLM's Python source, but not write GPU kernels.

## 2. What is the load and the score?

Locked flags: `--max-model-len 16384 --gpu-memory-utilization 0.85 (0.70 on the 5090) --no-enable-prefix-caching
--tensor-parallel-size 1 --dtype auto --seed 0 --kv-cache-dtype auto`. Client: `vllm bench serve`, random prompts,
`--ignore-eos`, temperature 1, closed loop; metric = request throughput.

| Workload (in/out) | Concurrency × requests |
|---|---|
| 1024/64 | 128 × 512 |
| 128/128 | 256 × 1024 |
| 2048/256 | 64 × 256 |

A scoring starts a fresh server per tree and round, runs 2 rounds in ABBA order, and keeps one compile cache per
tree between its rounds. Score = geometric mean over workloads of candidate/baseline throughput.

## 3. How much does the known fix gain?

**REF** is the reference's score; **A/A** scores the unmodified tree against itself.

| GPU | REF scorings | Per workload 1024/64 · 128/128 · 2048/256 | A/A scorings | Largest A/A deviation |
|---|---|---|---|---|
| RTX 4090, first server | 1.3120, 1.3127, 1.3144 | about 1.33 · 1.45 · 1.18 | 1.0016, 0.9995 (+ 1.0008, 1.0002, 1.0004, 1.0010 on an earlier harness; 0.9992 on the published harness) | 0.16% (0.55% on one workload) |
| RTX 4090, second server (acceptance) | 1.3236, 1.3295 | up to 1.351 · 1.462 · 1.190 | 0.9981 | 1.0% on one workload-round |
| RTX 5090 at 0.70 | 1.3169, 1.3216, 1.3239 (+ 1.3243, 1.3186 acceptance) | 1.278-1.293 · 1.433-1.444 · 1.246-1.252 | 1.0010, 0.9996, 0.9948 (+ 0.9985 acceptance) | 0.52% (worst round −0.96%, worst workload-round −1.3%) |

- **Re-measure REF on your own host before you use it as a reward anchor.** Two 4090 servers of the same model
  differ by 1.0-1.3%, outside the first server's A/A band. An untested candidate cause: the tester built their own warm
  compile cache.
- On the 5090 the 128/128 gain survives once the memory budget is matched (1.433-1.444 vs 1.444-1.449 on the 4090).
  At 0.85 the 5090 baseline already ran 198 requests at once, so the gain there would be much smaller. The REF value
  depends on the memory budget as much as on the GPU. 5090 scores are not comparable with 0.85 scores on any card.

## 4. How are outputs checked?

Any check that fires scores 0:
- Apple's logprob gate (≥ 95% of positions within 0.05 + 0.05·|lp|), on public prompts and on 8 fresh prompts made
  per scoring;
- a mean-drift bound: 0.012 public, 0.02 fresh. The fresh bound was raised from 0.012 after an earlier reference hit
  0.0117; the current reference reaches 0.0150 on the 4090;
- exact-length, non-empty outputs;
- an output witness: scored responses re-scored on the pristine server, threshold |z| 6. The reference reaches
  |z| 1.40 (mean logprob) and 1.99 (top-1/top-5) on the 4090, 3.63 on perturbed shapes;
- a runtime audit of file reads and writes, integrity digests, no leftover processes on any GPU, a memory kill line
  (23,800 MiB on the 4090; 31,000 MiB whole-card on the 5090).

The 4090's `corr_ref.json` passes on the 5090 (pristine pass 0.996-0.998, drift 0.0064-0.0071), so it was not
recalibrated. Allowed: flags (except locked ones), whitelisted environment variables, vLLM Python. Not allowed:
custom GPU kernels.

## 5. Which broken or cheating versions are caught?

No fault patches were run on this task. What we know about the checks' reach:

| Probe | Result |
|---|---|
| 3 mantissa bits dropped | drift 0.0107, **passes**: small numerics changes go undetected |
| fp16 GDN state (part of the reference) | allowed; no gated measurement separates it from pristine |
| Honest configs vs the bounds | fresh drift sits 25% under its bound; witness top-k reaches |z| 3.6 of 6 |

The checks are tripwires: there is no sandbox.

## 6. What are the limits?

- **Partly shape-tuned.** On unpublished perturbed shapes (768/32 c96, 256/192 c192, 3072/128 c48) the reference
  keeps +10.8% of its +31% on the 4090 (1.109 and 1.107; the second run was voided by a leftover process of ours on the
  other card) and about +7% of +32% on the 5090 (1.072, with the 768/32 gain inside noise). Perturbed A/A: 0.998
  (4090), 0.996 (5090).
- **Host-specific REF**: see section 3.
- **Thin memory margins.** On the 4090 the reference peaks 168-202 MiB under the kill line, and peaks move about
  450 MiB between rounds; a 4090 that also drives a display will not fit. On the 5090, memory grows outside vLLM's
  budget when the c256 workload starts; at 0.85 and 0.80 it crossed the kill line, which is why 0.70 is locked. The
  watchdog polls once a second and missed bursts of 500-1,040 MiB in 200 ms. The cause is untested.
- **Gate hazard, rare but loud.** A cached Inductor autotune choice can make an honest server fail the public logprob
  probe every time (pass 0.40). Seen in 1 of 11 arm caches on a related task, 0 failures in 116 task-C sessions. Use a
  gate-checked compile cache (`build_jit_master.py` refuses one that fails) and rerun with the cache reset on a gate
  failure.
- **Audit false positive.** On the 5090, an honest pristine round returned one 128-token answer of only whitespace;
  `bench_driver.out_check` counts it as empty and voided the scoring as `harness_fault` (1 of the tester's 4 scorings).
  Expect occasional voided scorings and rerun them.
- Every 4090 scoring ran with the other card busy.

## 7. What does it cost to run?

| | RTX 4090 | RTX 5090 |
|---|---|---|
| Setup | about 6 min on a warm uv cache, about 15 min plus downloads | 800 s from an empty directory (venv 117 s + 212 s, model 227 s, cache build 368 s) |
| Compile cache build | about 10 min | 352-368 s |
| One scoring | about 19 min (1084-1119 s in acceptance) | 20.7-21.1 min |
| A dev estimate | about 3 min (5.5 min paired) | - |
| Peak VRAM | baseline 22,628 MiB, reference 23,598-23,632 MiB | baseline 24,216 MiB, reference 26,422 MiB (whole card, desktop included) |

## 8. Has someone else rebuilt it?

Yes, on both GPUs: **PASS-with-fixes**.
- 4090 (a second server): REF 1.3236 and 1.3295, A/A 0.9981. Setup took about 50 min the first time. Fixes since then:
  a from-scratch path, a pinned venv (an unpinned install pulls transformers 5.24 and vLLM 0.19.1 fails at start-up),
  a pinned model revision, path and GPU-slot variables, an audit that uses `realpath` (a symlinked uv interpreter had
  voided scorings), a compile-cache builder that fails loudly, and process-group cleanup on SIGTERM. The host offset is
  now stated.
- 5090: REF 1.3243 and 1.3186, A/A 0.9985, all inside our bands. Fixes: 5090 commands added, paths made variables,
  `UV_HTTP_TIMEOUT=300`, `uv python install --no-bin` suggested.

## 9. How to run it

Assemble a root directory from `environment/` and `tests/`, then build and score. Paths are placeholders.

```bash
export P3_ROOT=/local/taskc P3_STAGE=/local/nvme/taskc_stage UV_HTTP_TIMEOUT=300     # placeholders: pick your own
export P3_GPU_SLOTS='{"0":{"gpus":"0","cpus":"0-15","client_cpus":"16-19"}}' P3_GPU_SLOT=0   # your GPU / CPU split
mkdir -p $P3_ROOT/results && cp -a environment/tasks environment/scripts $P3_ROOT/ && cp -a tests/harness_v5d $P3_ROOT/ \
  && cp environment/pins/* $P3_ROOT/results/
P=$P3_ROOT; H=$P/harness_v5d; TK=qwen35_4b_1x4090_hc      # RTX 5090: TK=qwen35_4b_1x5090_hc
bash $P/scripts/setup_venv.sh && bash $P/scripts/download_models.sh          # exact venv freeze; Qwen3.5-4B @851bf6e8
(cd $P/src/models/qwen3.5-4b && sha256sum -c <(grep -v '^#' $P/results/qwen3.5-4b.sha256))
cp $P/tasks/qwen35_4b_1x4090/pristine.sha256 /tmp/pristine.ref && bash $H/build_pristine.sh qwen35_4b_1x4090 \
  && diff /tmp/pristine.ref $P/tasks/qwen35_4b_1x4090/pristine.sha256 && ln -sfn ../qwen35_4b_1x4090/pristine $P/tasks/$TK/pristine
bash $P/scripts/stage_tmp.sh
bash $H/mkstage.sh $TK ref_fp16 "--max-num-batched-tokens 12288" "--mamba-ssm-cache-dtype float16"
J=$(python3 -I -c "import json;print(json.load(open('$P/tasks/$TK/task.json'))['jit_master'].rsplit('/',1)[1])")
$P/venv/bin/python -I $H/build_jit_master.py --task $P/tasks/$TK --out $P3_STAGE/run/$J   # gate-checked warm cache
$P/venv/bin/python -I $H/score.py --task $P/tasks/$TK --cand $P/tasks/$TK/sweep/ref_fp16 --label ref1   # REF
$P/venv/bin/python -I $H/score.py --task $P/tasks/$TK --cand $P/tasks/$TK/pristine --label aa1          # A/A
```

- Result: `$P/tasks/$TK/results/<label>/summary.json`: `verdict` (`scored` or the first flag that fired),
  `speedup_geomean`, `per_workload`, `witness`, `peak_mib`. `--wlset perturbed` runs the unseen shapes.
- A config-only candidate: `bash $H/mkstage.sh $TK myconf "--max-num-batched-tokens 8192"`. For a candidate that
  edits `repo/`, run `$P/venv/bin/python -I $H/gate.py <pristine dir> <stage dir>` first; the scorer does not gate.
- `setup_venv.sh` ends with a `torch.cuda` call; on a shared GPU take your GPU lock. Use SIGTERM, not SIGINT, to
  stop a scoring from a script.
- **What we changed in the published copy.** Author paths became placeholders: the project root default is
  `/opt/rsi/taskc` and the staging default `/tmp/taskc_stage` (set `P3_ROOT` / `P3_STAGE`; the scorer rewrites
  task.json paths from these defaults). The audit protects your home directory (`~`) instead of the authors' one; a
  private scratch path and two internal names were removed from the audit and gate lists; `download_models.sh` keeps
  only Qwen3.5-4B. The audit's allowed executables include `ptxas-blackwell`, the one harness edit made for the 5090
  (Triton uses it on arch ≥ 100). The agent-trial wrappers (`dev_run.sh`, workspace builder, remote entry points) are
  not included. The published numbers came from harness_v5c (4090) and v5d (5090, and one 4090 A/A); v5d is v5c
  plus portability fixes and behaves the same with no variables set.

# Acceptance batch 4: six SGLang/vLLM packages (2026-11-24/25)

An independent tester ran each package the way an outside user would:
- **Copy:** each package directory alone, into an empty directory on the 2-GPU 4090 server (on shared storage).
- **Settings:** `site.env` made from `site.env.example`, changing only `STAGE` (to local NVMe).
- **Fresh everything:** fresh uv cache, CPython, source tarballs, models and CUDA downloads. None of the authors' venvs, caches or results were used.
- **Builds:** `RUN.md`'s build, model, CUDA and stage steps in `RUN.md`'s order, pinned to CPU cores away from the GPUs' cores at low priority.
- **GPU jobs:** one queue per GPU, setting `GPU`, `CPUS`, `PORT` and `STOP_FILES` in the shell, as `RUN.md` allows. GPU0 used CPUs 0-7,32-39; GPU1 used CPUs 24-31,56-63 and the package's port + 500.
- **Staging vs timing:** staging held an exclusive lock that every GPU job held shared, so no staging overlapped a timed job. Pip installs for other packages sometimes did run during timed jobs.

**Which runs count as a second-host check:** sgl16521 and sgl20972 were built and calibrated by the authors on this same server. sgl9460, vllm37338, sgl27868 and sgl14956 were calibrated on the 4-GPU server, so for them this is a genuine second host.

**Package versions:** each package was run in the version the authors' empty-directory run marked READY. Its `MANIFEST.sha256` checks clean against the tester's copy, and no package file changed after the runs.

## Summary

| Package | Build from an empty directory | Fix scoring jobs (band) | Calibration REF / A/A max deviation (card) | Variants caught | Transfers vs card | Deviations from `RUN.md` | Verdict |
|---|---|---|---|---|---|---|---|
| sgl16521 | all checks pass, 22 min | 3.809, 3.831; band 3.779-3.875 | 3.807 / 0.23% (3.833 / 0.27-0.61%) | 5 of 5 | match | 0 | **PASS** |
| sgl20972 | all checks pass, 23 min | 1.1453, 1.1452; band 1.135-1.149 | 1.139 / 0.52% (1.142 / 0.58-0.65%) | 3 of 3 | match | 0 | **PASS** |
| sgl9460 | all checks pass, 15 min | 1.1951, 1.1999, 1.2116; band 1.185-1.221 | 1.214 and 1.214 / 1.6% on one GPU; the other had one stalled parent run | 6 of 6 | match after a repeat | 0 | **PASS** (second host) |
| vllm37338 | all checks pass, 18 min | 1.2829, 1.2926, 1.2915; band 1.264-1.316 | 1.291 / 4.25% from one pair, others ≤ 0.95% (1.29 / 0.9%) | 4 of 4 cheats; the `gbf16` probe passes, as the card says | match | 0 | **PASS** (second host) |
| sgl27868 | all checks pass, 17 min | 1.0272, 1.0280, 1.0280; band 1.0225-1.0327 | 1.027 / 0.23% (1.0276 / 0.18-0.7%) | 3 of 3 | match | 0 | **PASS** (second host) |
| sgl14956 | all checks pass, 13 min | 1.3073, 1.3061, 1.3056; band 1.284-1.336 | 1.307 / 0.36% (1.31 / 0.9-1.4%) | 3 of 3 | match | 0 | **PASS** (second host) |

**Conclusions:**
- All six packages could be followed from `RUN.md` alone, with no edits to scripts, paths or commands.
- Every build reproduced the authors' environment exactly: source hashes, freezes, every installed SGLang/vLLM file and the CUDA tree. The only exception is sgl16521's generated `_version.py` in the fix arm, which `RUN.md` already expects.
- All 18 human-fix jobs landed inside their card's band with verdict `ties_ref`.
- Every broken variant got the verdict the card lists, and every cheat was rejected.

**Findings:**
1. **Manifests were stale while packages were still in progress.** At copy time, sgl9460's manifest failed on 3 files and vllm37338's on 5. Both check clean now; the READY versions differed from the tester's copies only in documentation and `gate.json`. The `RUN.md` CPU example (the other server's layout) is fixed in the READY versions.
2. **sgl9460 card: the `lowres` variant is fast, not slow.** The card said 0.74× "(slower)" and called it the one variant that is not fast-but-wrong. On this server it ran at **1.28×**, faster than the fix (fix 3.26 s against parent 4.17 s per rep), and was caught by `correct,tf` with the same teacher-forced numbers as the card (4.20 nats over 1,199 tokens against 4.17 over 1,228). Passed to the authors to correct.
3. **sgl9460 runs sometimes go slow for one whole process, in either arm:**
   - a parent A/A run whose reps 3-5 took 16-26 s instead of 4.1 s, with GPU utilization about 15%, so the stall was on the host side;
   - a fix transfer run at batch 32 with every rep at 3.3-4.1 s instead of 2.3 s;
   - one slow fix pair, 1.086, in a scoring job.

   The card's 3-pair median absorbs them, and every job stayed in band. Single pairs and single-pair transfers of this task are not reliable. The authors' screen saw the same transient.
4. **vllm37338's calibration A/A of 4.25% comes from one parent run** whose `LLM()` start-up was 8 s slower (165.9 against 157.8 s). It coincided with the tester creating a 13 GB variant venv and a cold first start of that venv on the other GPU. The other four pairs are ≤ 0.95%, and REF is unaffected. Cold-start tasks need a quiet host. The gate does not use the calibration's A/A, so the verdict was still `ties_ref`.
5. **The tester's own concurrent work** (variant venv creation, a pip install pinned to other cores) is the likely cause of the outliers in findings 3 and 4. Keep the host quiet during timed jobs.

## sgl16521 (SGLang #16521, radix key includes generated tokens): PASS, no deviations

- **Build:** `setup_lock.sh` passed in 697 s (1896 of 1896 source files; freeze equal, 168 packages; parent 1857 of 1857 SGLang files, fix 1856 plus the expected `_version.py`); `get_model.sh` 33 s; `fetch_cuda.sh` 14 s (1740 of 1740 files); `stage.sh` 581 s (24 GB).
- **Calibration:** pairs 3.801-3.812, REF **3.807**, `ties_ref`. A/A 0.998-1.001, max **0.23%**. Correctness pass 29/32 identical to the parent, as on the card. Teacher-forced maximum gap 0.125 nats for both arms. 1336 s of GPU time (card 1325 s), peak 20.4 GB.
- **Scoring:** **3.809** (GPU1) and **3.831** (GPU0), both `ties_ref`; 519-529 s each (card 518-535 s). GPU0 reads about 0.6% higher, matching the card's offset.

| Variant | Speedup | Verdict | Card |
|---|---|---|---|
| drophist | 4.24× | `fail:tf` (max gap 5.0 nats) | 4.28×, `fail:tf` |
| kvzero | 2.70× | `fail:tf` (7.1 nats) | 2.71×, `fail:tf` |
| fp8kv | 3.99× | `fail:settings,tf` | 3.99×, `fail:settings,tf` |
| short2 | 4.85× | `fail:counts` | 4.86×, `fail:counts` |
| kvnoise100 | 1.59× | `fail:tf` | 1.64×, `fail:tf` |

- **Transfers:** previous answer of 1024 tokens 11.19×, of 64 tokens 1.67×, page size 16 3.89× (card 11.3 / 1.71 / 3.84).

## sgl20972 (SGLang #20972, no blocking copy with top-k log-probabilities): PASS, no deviations

- **Build:** `setup_lock.sh` passed in 705 s (2220 of 2220 source files; freeze 177 packages; 2171 of 2171 SGLang files in both arms); model 74 s; CUDA 33 s; stage 551 s.
- **Calibration:** pairs 1.137-1.145, REF **1.139**, `ties_ref`. A/A max **0.52%**. Exact digests 25/25, correctness pass 24/24. 1527 s of GPU time, peak 20.6 GB.
- **Scoring:** **1.1453** and **1.1452**, both `ties_ref`; every rep digest identical; 575-600 s (card 9.4-10 min).

| Variant | Speedup | Verdict (card: 1.139-1.144×, all `fail:exact,correct`) |
|---|---|---|
| droplp | 1.146× | `fail:exact,correct` |
| stalelp | 1.145× | `fail:exact,correct` |
| fp16lp | 1.138× | `fail:exact,correct` |

- **Transfers:** batch 1 1.163, batch 32 1.159, top-20 1.146, no top-k 0.997 (card 1.173 / 1.162 / 1.157 / 1.002).

## sgl9460 (SGLang #9460, M-RoPE decode positions): PASS on a second host, no deviations

- **Build:** `setup_lock.sh` passed (893 of 893 source files; freeze 149 packages; 876 of 876 SGLang files). Its 260 s is short because the uv cache was partly warm from the tester's own first attempt, which it stopped to add the I/O lock. Model 86 s; stage 578 s.
- **Calibrations:**
  - first (GPU1): REF **1.2143**, `ties_ref`. One A/A pair read 0.2486: a parent run with reps 3-5 at 16-26 s (finding 3). The other A/A pairs' maximum is 1.8%.
  - extra (GPU0), to check the stall: REF **1.2139**, A/A max **1.6%** (card 1.9%).
  - correctness pass 16/16 bit-identical in both.
- **Scoring:** **1.1951**, **1.2116** and **1.1999**, all `ties_ref`. Pairs ranged 1.086-1.241; the card says single pairs are noisy.

| Variant | Speedup | Verdict | Card |
|---|---|---|---|
| nodelta | 1.22 | `fail:correct,tf` (timed teacher-forced 2.76 / 166) | 2.63 / 164 |
| offby1 | 1.22 | `fail:correct,tf`; teacher-forced only from the correctness pass (4 of 1024 tokens) | same |
| frozen | 1.26 | `fail:correct,tf` (20.4 / 7,500) | 20.4 / 7,634 |
| short | 1.34 | `fail:counts` | 1.35 |
| fp8kv | 1.21 | `fail:correct,tf` | 1.21 |
| lowres | **1.28** | `fail:correct,tf` | **0.74**, see finding 2 |

- **Transfers:** text-only 1.125 / 1.111 / 1.111 (card 1.10); batch 32 read 0.774 on a first run (the fix process was slow in every rep), then 1.174 on a repeat and 1.174 on the other GPU (card 1.18).

## vllm37338 (vLLM #37338, Triton autotune in warm-up, cold start): PASS on a second host, no deviations

- **Build:** `setup_lock.sh` passed in 676 s (freeze 175 packages; 2085 of 2085 vLLM files in each arm; 0.18.1rc1.dev37/38, torch 2.10.0+cu128); model 122 s; stage 256 s. The tester started before the package was READY; only `gate.json` and documentation changed afterwards, and it synced those before any GPU step.
- **Calibration:** pairs 1.278-1.300, REF **1.291**, `ties_ref`. A/A max 4.25% from one pair (finding 4); the other pairs ≤ 0.95% (card 0.9%). First-batch ratio 36.5-44.4×. 4500 s of GPU time (card 4684 s).
- **Scoring:** **1.2829**, **1.2915** and **1.2926**, all `ties_ref`. The audit read 65 files, all Python. Cold witness: 289 kernels compiled per fix run against 389 per parent run. 1410-1588 s each (card about 1405 s), peak 21.5 GB.

| Variant | Speedup | Verdict | Card |
|---|---|---|---|
| short1 | 1.278 | `fail:counts` | 1.29, `fail:counts` |
| shipcache (after `donor1`) | 3.29× | `fail:audit,venv,cold` | 3.35×, same checks |
| persist, first job | 0.997 | `fail:audit,venv,cold` | 1.00, same |
| persist, second job | 4.67× | `fail:audit,venv,cold` | 5.82×, same |
| gbf16 (a probe, expected to pass) | 1.29 | passes (`ties_ref`) | passes |

The `persisttmp` variant was not run: it writes to a fixed path under `/tmp` shared by all users of the host.

- **Transfers:** 1 × 128 tokens 1.247, 16 × 2048 tokens 1.385, warm caches 1.011 (card 1.29 / 1.29-1.36 / 0.87-1.00).

## sgl27868 (SGLang #27868, Qwen3.5 dual-stream overlap): PASS on a second host, no deviations

- **Build:** `setup_lock.sh` passed in 457 s (3106 of 3106 source files; freeze 196 packages; 3064 of 3064 SGLang files; torch 2.11.0+cu129; only `qwen3_5.py` differs); CUDA 37 s; model 107 s; stage 405 s (34 GB).
- **Calibration:** pairs 1.024-1.0285, REF **1.027**, `ties_ref`. A/A max **0.23%**. Every timed digest identical, fix against parent and parent against parent; correctness pass 24/24.
- **Scoring:** **1.0272**, **1.0280** and **1.0280**, all `ties_ref`; 556-629 s each.

| Variant | Speedup | Verdict | Card |
|---|---|---|---|
| short | 1.316 | `fail:counts,exact` | 1.32 |
| noknorm | 1.031 | `fail:exact,correct` (0/24, 1.14 nats) | 1.024 |
| fp8kv | 1.028 | `fail:exact,correct` | 1.029 |

- **Transfers:** batch 1 1.005, batch 8 1.024 (card 1.006 / 1.027).

## sgl14956 (SGLang #14956, candidate C48): PASS on a second host, no deviations

The tester had screened this candidate itself earlier (1.31× on this server), so it knew the expected numbers. The package, card, gate and variants were built and calibrated by the authors on the other server. As instructed, the tester did not read the authors' empty-directory report or working directory before it had its own numbers.

- **Build:** `setup_lock.sh` passed in 312 s (1470 of 1470 source files; freeze 166 packages; 1432 of 1432 SGLang files; only `forward_batch_info.py` differs); CUDA 27 s; model 59 s; stage 372 s.
- **Calibration:** pairs 1.306-1.309, REF **1.3068**, `ties_ref`. A/A max **0.36%** (card 1.4% / 0.9%). All 25 timed digests identical; correctness pass 24/24. 1134 s of GPU time (card 1145 s).
- **Scoring:** **1.3073**, **1.3056** and **1.3061**, all `ties_ref`; 439-455 s each (card 452 s), peak 20.5 GB.

| Variant | Speedup | Verdict | Card |
|---|---|---|---|
| offby1 | 1.307 | `fail:exact,correct` (0/24, 0.53 nats) | same |
| multistep | 1.304 | `fail:settings` | same |
| short | 1.645 | `fail:counts,exact` | 1.65 |

- **Transfers:** batch 8 1.075 (card 1.085); image requests 1.003 (card 1.007), whose `fail:exact` is expected, as the card says.

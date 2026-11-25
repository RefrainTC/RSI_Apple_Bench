# Acceptance batch 5: vllm35442 (C15) and vllm27418 (C77) (2026-11-25)

An independent tester rebuilt both tasks from their packages alone, on the 4-GPU 4090 server, on 2026-11-25. It read only the two packages, the two cards and earlier acceptance reports. It read the authors' empty-directory report only after its own numbers were in, and did not read the prediction files, the screening notes or the authors' working directories. The packages and cards were not edited during the test.

## Summary

| Package | Verdict | REF: tester vs card | A/A max deviation: tester vs card | Main problem for an outside user |
|---|---|---|---|---|
| vllm35442 (spec-decode non-blocking copy) | **PASS-with-fixes** | 1.0377 (median of 4 jobs: 1.0374, 1.0393, 1.0377, 1.0360) vs **1.037 ± 1%** | 1.42% (first calibration), 0.30% (second) vs 0.14-0.44% | **The first calibration on a fresh install failed its own correctness check** (`fail:correct`), from a harness bug plus a cold compile cache. A second calibration passed. |
| vllm27418 (ViT Conv3d to Linear) | **PASS-with-fixes** | 1.679 (median of 3 jobs: 1.693, 1.675, 1.679) vs **1.69 ± 3%** | 1.24% vs 0.7-3.6% | The re-read step of the `subpatch` variant peaked at 23,782 MiB, 45 MiB under the package's own watchdog and above the 23,500 MiB limit the tester had to keep. It was rerun with a smaller memory pool, as a logged deviation. |

Both builds reproduce exactly: the freezes and every vLLM file hash match the authors' records. Every scoring job of the human fix is `ties_ref`. All 8 variants run got the verdict the card predicts, with the same failing checks. In vllm27418, `subpatch` is caught by the same 1-token margin the card reports.

**After this test** the authors fixed both packages (and vllm30056, which shared the harness) and re-verified them from new empty directories: both pass, and the first calibration's correctness pass now agrees 24/24. See "What was fixed afterwards" below.

## How was it run?

- Each package directory was copied alone into an empty directory on shared storage, with the stage on local disk; vllm35442 on GPU0 (CPUs 0-7,32-39), vllm27418 on GPU1 (CPUs 8-15,40-47). `export TZ=UTC` was run and `TMPDIR` set to a local directory.
- `MANIFEST.sha256` checked clean for both packages before anything ran.
- `RUN.md` steps 2-4 were run literally, through a wrapper that only times them.
- Every GPU job went through the tester's own guard. It refused to start if a stop file existed, the authors' "GPU released" marker was missing, the GPU had any compute process, or a set deadline had passed. During a job it killed the tester's own job processes if the card went over 23,500 MiB. Inside the guard, the package's `harness/run.sh` ran unchanged.
- Both packages were set up in parallel, then calibration and scoring ran on two GPUs at once. The authors' jobs ran on the other two GPUs the whole time.

## vllm35442: does it reproduce?

### Build

| Step | Time |
|---|---|
| `setup_lock.sh` | 412 s |
| `get_model.sh` | 125 s (8.8 GB) |
| `stage.sh` | 514 s (28 GB) |
| Total | about 17.5 min (`RUN.md` says 5-15 + 3-10 + 2-10 min) |

Both freezes equal the authors' (179 packages), and all 1,998 vLLM files of each venv hash as the authors' build. Only `_version.py` and `v1/worker/gpu_model_runner.py` differ between the arms, as documented.

### Speed

| Job | GPU | Speedup (pairs) | Verdict | Wall time |
|---|---|---|---|---|
| calibration on empty caches | 0 | **1.0374** (1.0346-1.0404) | **`fail:correct`** (below) | 2,110 s |
| scoring | 0 | **1.0393** (1.0384-1.0395) | `ties_ref`, all checks pass | 766 s |
| scoring on a GPU with empty caches | 1 | **1.0377** (1.0372-1.0391) | `ties_ref`, all checks pass | 864 s |
| calibration repeated on warm caches | 0 | **1.0360** (1.0344-1.0384) | `ties_ref`, all checks pass | 1,925 s |

- The median of the 4 jobs is **1.0377**, against REF 1.037 (band 1.027-1.047). All four are inside the band.
- Median rep times: fix 3.598-3.617 s, parent 3.733-3.757 s (card 3.59-3.62 s and 3.73-3.76 s).
- **A/A.** First calibration: 0.9858-1.0008, max deviation **1.42%**; one parent run was 1.3% slow in all three reps. That is above the gate's A/A limit of 1%. Second calibration: 0.30%. Like the card's own "one slow parent run", this looks like whole-run host noise. The card's A/A figure is on the optimistic side: expect about 1.5% now and then. It only matters for `no_gain` decisions.
- **Transfer load `text`** (one pair): **1.036**, outputs identical (card 1.0385 and 1.0372).
- **Cost.** Scoring job 12.8 min with warm caches, 14.4 min with cold ones (card 13-16 min). Calibration 35 and 32 min (card about 34). One run about 75 s. Peak VRAM **20,608 MiB** in every job.

### Outputs: does the fix pass the exact gate?

In the scoring jobs, yes. In the first calibration on a fresh install, **no**:
- Timed reps: all digests identical, fix against parent and parent against parent, in every job.
- Text pass (24 prompts with log-probabilities) in the first calibration: fix against parent **0/24** identical (log-probabilities up to 0.12 nats apart, 10 prompts with different tokens); parent against parent only **18/24**; the tester's parent pass against the authors' stored reference (`ref_correct.json`) 0/24. Verdict `fail:correct`, where `RUN.md` expects no failure.

**Why it failed.** Two causes, both in how compile caches start on a new stage directory.
1. **A harness bug: the first candidate run never got the parent's cache.** `harness/run.sh` copied the parent's cache with `rm -rf "$C"; mkdir -p "$PCD"; cp -a "$PCD" "$C"`. On a fresh stage the parent directory of `$C` did not exist yet, so `cp` failed ("cannot create directory ...: No such file or directory"), and nothing checked the failure. The run went ahead on an empty cache, and Inductor tuned its kernels afresh (compile 26 s instead of 7 s; run 175 s instead of 75 s). In calibration mode the first fix run is the correctness pass, so exactly the run that is checked got kernels tuned on their own. The same error appears for the first run of every new variant, and in vllm27418, where it was harmless because that first run is an untimed warm-up.
2. **A cold first run differs from warm runs, even for the parent.** The calibration's first run is the first run on the stage, so it compiles and tunes during the run. Its text pass differed from every later parent pass on the same cache (18/24 tokens equal, 0/24 log-probabilities), and the gate compared the fix with this cold pass.

**Evidence that this is the whole story.** The calibration was rerun with warm caches and the directory in place. It passed: text pass 24/24 identical (fix against parent and parent against parent), REF 1.036, A/A 0.30%. The scoring jobs passed from the start, because scoring mode runs the warm-ups before the correctness passes.

**The authors' run was lucky.** Their empty-directory calibration had reported 24/24. A fresh install produces one of several numeric "families" for the text pass. The tester found **four**:
- (A) warm caches on its GPU0: parent and fix in all later jobs;
- (B) warm caches on its GPU1, identical to the authors' stored reference;
- (C) its cold first parent pass;
- (D) the fix on an empty cache.

Any two families share only 14-18 of 24 token sequences. Whether a fresh calibration passes depends on whether the cold passes land in the same family. The stored reference is therefore not portable: GPU1 reproduced it exactly, GPU0 did not (16/24).

### Variants

All three caught, with the same failing checks (one quick job each):

| Variant | Speedup: tester (card) | Text pass | Timed reps identical | Verdict |
|---|---|---|---|---|
| `stale` (caught by the text pass alone) | 1.036 (1.03) | **0/24**, log-probabilities up to 0.83 nats apart (card 0.87) | 3 of 3 | `fail:correct`, as the card |
| `spec2` (2 draft tokens set inside the config) | **1.361** (1.36) | 20/24 | 3 of 3 | `fail:correct,settings`, as the card |
| `short` (output capped at 400 tokens) | 1.326 (1.32) | 24/24 | 0 of 3 | `fail:counts,exact`, as the card |

For `spec2`, the settings that differ are the number of speculative tokens, the block size and the padded Mamba page size. `stale`'s timed outputs equal the parent's, so only the text pass guards the changed code path; its 0.83-nat gap is far above the 0.12-nat spread between cache families, so this catch is robust.

### Fixes the tester asked for

1. `harness/run.sh`: create the parent directory before copying, and stop the job if the copy fails.
2. Calibration mode: run the two warm-ups (parent, then fix) before the three correctness passes, as scoring mode does.
3. `RUN.md` and the card: say that the text pass has several numeric families, that the first run on a fresh stage is cold, and that the stored reference holds only for the authors' family.
4. The card's A/A figure is optimistic; one slow parent run gave 1.42%.

## vllm27418: does it reproduce?

### Build

| Step | Time |
|---|---|
| `setup_lock.sh` | 696 s, including the flashinfer sdist built without build isolation and triton-kernels from git |
| `get_model.sh` | 53 s (4.2 GB) |
| `stage.sh` | 390 s (24 GB) |
| Total | about 19 min |

Both freezes equal the authors' (159 packages), and all 1,494 vLLM files hash as the authors' build. `_version.py` and the 6 vision model files differ between the arms.

### Speed

| Job | GPU | Speedup (pairs) | Verdict | Wall time |
|---|---|---|---|---|
| calibration | 1 | **1.693** (1.625-1.721) | `ties_ref` | 1,605 s |
| scoring | 1 | **1.675** (1.673-1.725) | `ties_ref` | 647 s |
| scoring on a GPU with empty caches | 0 | **1.679** (1.678-1.740) | `ties_ref` | 680 s |

- The median is **1.679**, against REF 1.69 (band 1.639-1.741). The authors' empty-directory run also gave 1.679.
- Median rep times: fix 1.449-1.511 s, parent 2.433-2.567 s (card 1.46-1.51 s and 2.47-2.61 s).
- **A/A** (calibration): 0.9937-1.0124, max deviation **1.24%**, inside the card's spread and under the gate's 5%.
- **Checks on the fix, all pass:** text pass 24/24 bit-identical and identical to the stored reference (the text path does not depend on the cache family); image-pass answers equal the parent's; teacher-forced re-read worst gap 0.125 nats, the same as the parent re-reading itself, with no token above 0.5 nats; timed digests: rep 2 differs in every pair and the other reps are identical, exactly as the card describes.
- **Transfer loads** (one pair each, two jobs): `i448` 1.187 and 1.242 (card 1.263 and 1.270), 2-6% lower, with reps only 0.3-0.4 s long, so one pair is noisy; the direction holds. `w1` 0.785 and 0.884 (card 0.85 and 0.92): no gain, as the card says.
- **Cost.** Parent run about 41 s, fix run about 57 s. Scoring job 10.8-11.3 min, calibration 26.75 min (card about 11 and 26). Peak VRAM 21,050 MiB in calibration and scoring; the fix's re-read step 18,320 MiB.

### Variants

All six caught as the card says, with the same checks:

| Variant | Speedup: tester (card) | Re-read worst gap; share of tokens above 0.5 nats | Other checks | Verdict |
|---|---|---|---|---|
| `subpatch` (the narrow one), quick job | 1.70 (1.72) | **0.75; 2.5%** (4 of 160) | image pass 0.375, 0% | `fail:tf`, as the card |
| `subpatch`, full scoring job (3 candidate runs) | 1.647 | 0.75; **4 of 160 in each of the 3 runs** | image pass 0.375, 0% | `fail:tf` |
| `skipimg` | **4.53** (4.55) | 16.69; 50% | image-pass answers "assistant assistant" | `fail:tf`, as the card |
| `lowres` | **5.38** (5.58) | 1.875; 4.4% | image prompts 613 tokens instead of 2,341 | `fail:counts,tf`, as the card |
| `short` | 1.75 (1.77) | 0.125; 0% | counts; text pass 0/24 | `fail:counts,text`, as the card |
| `cheappatch` (expected to pass) | 1.70 (1.69) | 0.125; 0% | - | `ties_ref`, as the card |

**`subpatch` is caught by one token, every time.** The limit is "at most 2% of tokens above 0.5 nats", i.e. at most 3 of 160. Every `subpatch` run has exactly 4. The outputs are deterministic, so the result is stable, but there is no margin. The image pass alone does not catch it. This matches the card's stated validity limit, and it is reproducible.

**The re-read of `subpatch` needed more memory than any other job.** It took the card to **23,782 MiB** in two attempts, above the tester's 23,500 MiB limit, which killed the step. The package's own watchdog sits at 97% of the card (23,827 MiB), only 45 MiB higher; on a card with slightly less free memory the package itself would abort this re-read and the job would end with no verdict. **Deviation:** to finish, the tester ran the re-read step from a copy whose only difference was the re-read's memory pool (`MEMU` 0.60 → 0.50). Peak 21,374 MiB. The re-read numerics do not depend on the pool size: the gaps matched the card to the last digit.

### Fixes the tester asked for

1. Lower the re-read step's memory pool (0.50 worked) so it stays well below the watchdog for every candidate.
2. Keep the 2% limit; `subpatch` is caught by a single token.
3. Fix the same cache-copy bug in `harness/run.sh` (harmless here).

## Problems common to both packages

1. **`RUN.md`'s way of choosing CPU sets does not work on this server:** `nvidia-smi topo -m` shows CPU affinity 0-63 for every GPU. Only the table in `RUN.md` gives the per-GPU core sets, and that table was correct.
2. **The manifest did not cover `PROVENANCE.md`** (33 of 34 files).
3. **Job tags can collide between users of one host.** The job script found "its own" processes by a tag built from the package, label and tag across all processes on the host. Two people running the same package with the default labels at the same time could kill each other's runs. The tester used prefixed labels to avoid this.
4. uv printed a harmless warning about its binary directory not being on PATH.
5. Otherwise `RUN.md` was followed literally and was complete: no missing file, no host path, no undocumented override. The `TZ=UTC` issue did not matter, because the lock is fully hashed.

## Tester's deviations

1. Prefixed job labels instead of `RUN.md`'s defaults, to avoid tag collisions with the authors' jobs on the same host.
2. The tester's guard wrapped `harness/run.sh` (release-marker check, 23,500 MiB kill, a deadline), and passed `STOP_FILES`, a documented variable.
3. vllm27418 `subpatch` re-read run from a copy with a smaller re-read memory pool (above).
4. The first `skipimg` and `lowres` attempts were refused by the guard because the GPU still showed the process it had just killed; they were re-queued, and the guard now waits up to 60 s for the GPU to clear.
5. Extra jobs beyond `RUN.md`: a second calibration of vllm35442, a full scoring job of `subpatch`, and the `cheappatch` and `short` variants of vllm27418.

**Not tested:** vllm35442's other transfer loads; any GPU other than two of the 4-GPU server's; the other server; the RTX 5090; a fresh-stage calibration with the fixes applied (done by the authors afterwards, below).

**GPU time:** about 7,500 s for vllm35442 and 6,000 s for vllm27418, about 3.75 GPU-hours in all.

## What was fixed afterwards

The authors changed vllm35442 and vllm27418 (and vllm30056, which shared the harness):
- `harness/run.sh`: the cache's parent directory is created, a failed copy stops the job, and calibration runs the warm-ups before the correctness passes;
- the job script finds its own processes by tag **and** by the package's root directory;
- vllm27418's re-read memory pool 0.60 → 0.50;
- `RUN.md`: the text-pass families, the non-portable stored reference, and the CPU-affinity note;
- the manifest now covers `PROVENANCE.md`.

Re-verified from new empty directories and fresh stages, with no cache from any earlier run:

| Package | Calibration | Scoring | Status |
|---|---|---|---|
| vllm35442 | 1.0365, A/A max 0.31%; correctness pass 24/24 fix against parent and parent against parent, and equal to the stored reference; all 15 timed digests identical | 1.0377, `ties_ref`, all checks pass | **PASS** |
| vllm27418 | 1.679, A/A max 3.25% (gate limit 5%); text pass 24/24; re-read: fix maximum gap 0.125, no token above 0.5 | 1.686, `ties_ref`, all checks pass | **PASS** |

With the smaller pool, `subpatch`'s re-read peaked at 21,374 MiB, 2.4 GB under the watchdog, with the verdict unchanged (`fail:tf`). No `cp` error appears in any job log of the re-verification.

## Second host: vllm35442, vllm27418 and vllm30056 on the 2-GPU server

A second independent tester rebuilt all three packages on the 2-GPU 4090 server (both GPUs), from its own copies. Those copies were the versions **before** the authors' cache-copy fix (all three manifests checked with 0 mismatches against that version). No deviations from `RUN.md` were needed.

| Package | Fix scores vs band | A/A max deviation vs card | Variants | Verdict |
|---|---|---|---|---|
| vllm35442 | first calibration 1.0379 `fail:correct` (the cache-copy bug: parent against parent 18/24); warm calibration 1.0368; scoring 1.0383, 1.0368, 1.0376; band 1.027-1.047 | 0.26% fresh, 0.28% warm (card 0.14% / 0.44%) | `stale` `fail:correct` (0.87 nats); `short` `fail:counts,exact`; `spec2` `fail:correct,settings` | **PASS**, given the cache-copy fix |
| vllm27418 | calibrations 1.6881 and 1.7191; 8 scoring jobs 1.660-1.730; band 1.639-1.741 | **9.45% and 6.74%** (card 0.7% / 3.6%) | `subpatch` `fail:tf` (0.75 nats, 2.5%, re-read peak 23.8 GB); `short`, `lowres`, `skipimg` fail; `cheappatch` accepted (`above_ref`) | **PASS**; A/A noisier than the card says |
| vllm30056 | calibration 1.7878; 12 scoring jobs 1.778-1.806; band 1.736-1.844 | 2.04% (card 0.46% / 2.47%) | `noadvance` `fail:correct`; `short` `fail:counts,exact`; `noparser` `fail:exact,correct,settings` | **PASS** |

- **The cache-copy bug, again.** "cannot create directory" appears in 14 job logs: the first calibration of each package and all 11 variant runs (a variant's cache directory never existed before its first run), so every variant ran on an empty cache. Only vllm35442's calibration failed from it. The current `harness/run.sh` creates the directory and stops on a failed copy, which fixes the variants too.
- **vllm30056 flips:** none on this server in any arm (123 fix-against-parent timed reps, 15 parent-against-parent reps); correctness 24/24 in all 13 jobs.
- **vllm27418 timing:** some runs had single slow pairs (1.46-1.58), absorbed by the median. Runs alone on the server and runs on both GPUs at once were tight (1.675-1.725). The cause of the slow pairs was not found.
- **Transfer loads** (single pairs): vllm35442 `text` 1.036, `b4` 1.007 (card 1.048), `b16` 1.045; vllm27418 `i448` 1.297, `w1` 1.047 (card 0.85 / 0.92); vllm30056 `b8` 1.371, no structured output 0.999.
- **Card defects** in the tester's copy of the vllm30056 card: three unfilled placeholders, and an A/A row giving a 2% limit where `gate.json` has 3%.

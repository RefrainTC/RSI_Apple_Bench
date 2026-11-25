# SGLang #14956: Qwen2-VL decode, positions of text requests read from the CPU copy of the sequence lengths

On 64 text-only requests to Qwen2-VL-2B-Instruct, the human fix makes decoding **1.31x** faster on an RTX 4090
(band ±2 %, A/A up to 1.4 %). Outputs are bit-identical to the unmodified engine's. An independent tester rebuilt the
package on a second server and got 1.307: PASS, no deviations.

**This task is in the same family as `fix_sglang_9460_mrope_decode_sync` (SGLang #9460).** Both PRs edit the decode
branch of the same function and remove the same kind of blocking read; #14956 finishes what #9460 left undone. Count
the two as one family, not as independent evidence. For example, do not put one in a training set and the other in an
evaluation set.

Terms used below:
- **parent**: the unmodified tree, commit `a7a4b175`. **fix**: the human fix, commit `69cfb17b`.
- **pair**: one parent run and one candidate run on the same GPU; pair speedup = parent time / candidate time. A
  scoring job runs 3 pairs in ABBA order and reports the median pair.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **band**: the range around REF that counts as a tie
  (`ties_ref`). **aa_band**: a job at or below 1 + aa_band is `no_gain`.
- **gate**: the output checks. A candidate that fails any of them scores zero.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs, where the package was built and calibrated. "The 2-GPU
server" is a second RTX 4090 server with 2 GPUs, where the candidate was screened and the independent tester ran.
Both run driver 550 (CUDA 12.4); each job is pinned to the 8-core CCD of its GPU.

## What is slow, and where does this task come from?

Qwen2-VL's language model uses multimodal rotary positions (mrope), so every decode step builds each request's
position in `_compute_mrope_positions` (`srt/model_executor/forward_batch_info.py`), even for text. For a request
without an image, the parent builds the 3 x 1 position with `torch.full(..., self.seq_lens[i] - 1, device=cuda)`.
`self.seq_lens` is a GPU tensor, so `torch.full` must read the fill value back from the GPU: one blocking
device-to-host copy per text request per decode step, in the forward thread, which also stalls the overlap scheduler.

**How it relates to #9460.** From the two diffs:
- #9460's parent (2025-09) read `int(self.seq_lens[i])` twice per request, image or text, plus a `.tolist()` for image
  requests; each read waits for the GPU.
- #9460's fix removed all of that for image requests. For text requests it wrote the `torch.full(...)` line above,
  whose fill value is still a GPU scalar: one wait per text request per step remained.
- #14956's parent (2025-12) still has exactly that line; #14956 reads the CPU copy instead.

So #14956 is not a regression. The two tasks use different loads and each shows only its own PR: #9460's task scores
image requests, and on image requests this task's parent has no wait left (0.997x in the screen). This task scores
text-only requests. The skill a candidate needs is the same.

The PR was merged on 2025-12-13. We picked it from our list of upstream fixes mined from SGLang and vLLM history
(batch 5, candidate C48, prior 0.45; the list said "extend batch", but the diff changes the decode branch).

## What is the load and the score?

- Model: Qwen/Qwen2-VL-2B-Instruct @ `1860c46f` (bf16). Offline `sglang.Engine`, triton attention, pytorch sampling,
  decode CUDA graphs up to batch 64, radix cache off, `mem_fraction_static` 0.80, `SGLANG_DISABLE_CUDNN_CHECK=1`.
  Without that variable this SGLang refuses to start on torch 2.9.1 with cuDNN 9.10 (a Conv3d bug in the vision
  tower); the load has no images, so the tower never runs.
- One rep = **64 text-only requests x 128 random token ids -> 256 greedy tokens** (`ignore_eos`) in one `generate()`
  call. An untimed 8-request warm-up, then 5 timed reps (seed 1 + rep, the same in both arms). This is the screen's
  load, unchanged.
- **Metric: median rep wall time.** Score = median pair speedup of a scoring job. The harness also reports
  reward = min(1, (S - 1) / (REF - 1)).

## How much does the known fix gain?

Fix median rep 1.603-1.612 s, parent 2.10-2.13 s: ~0.5 s per rep over 256 decode steps, ~1.9 ms per step, i.e. ~30 us
per removed read (64 per step) once the lost CPU/GPU overlap is counted.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server) | **1.31** (median of 4 package jobs: 1.308) | ±2 % (1.284-1.336); between-job SD 0.6 % | 1.4 % (calibration), 0.9 % (empty-directory calibration) | 4 |
| RTX 4090 (2-GPU server) | screen 1.31; short ABBA jobs 1.3132, 1.3100, 1.3090, 1.3115; tester 1.3068-1.3073 | - | screen 0.5 %; tester 0.36 % | 4 + 4 (tester) |
| RTX 5090 | not run | - | - | - |

- Package jobs on the 4-GPU server: calibration 1.3217 (GPU2), scoring 1.3045 (GPU3), empty-directory calibration
  1.3095 and scoring 1.3070 (GPU2). All `ties_ref`.
- The fix's reps are very steady (1.6031-1.6045 s); the parent carries the noise (64 blocking copies per step).
- Transfer loads (one pair each): 8 text requests (8 reads per step instead of 64) **1.085**, outputs identical; 64
  image requests (one 224 px image each, the negative control) **1.007** (the screen: 0.997). Image-load outputs vary
  run to run within each arm, so that pair prints `fail:exact`, which is expected.

## How are outputs checked?

The fix changes where a number is read from, not its value, so the gate is exact (`package/harness/check.py`,
`package/harness/gate.json`):
1. **counts:** output tokens per request equal the paired parent's.
2. **settings:** the live `ServerArgs` equal the parent's (ports ignored).
3. **exact:** every timed rep's digest (output ids + text) equals the paired parent's.
4. **correct:** 24 fixed text prompts, one per call, 64 greedy tokens with logprobs and top-5, identical to the same
   job's parent pass (or to `package/harness/ref_correct.json`).

In calibration all 25 timed digests were identical fix vs parent and parent vs parent; the correctness pass was 24/24.
Unlike the vLLM tasks, no compile-cache copy was needed: outputs were identical with per-role caches.

## Which broken or cheating versions are caught?

Python-only edits on top of the fix (`package/variants/apply.py`), one `quick` job each.

| Variant | What it does | Speedup | Correctness pass | Timed reps identical | Verdict |
|---|---|---|---|---|---|
| `short` | requests asking for more than 200 tokens get 200 | 1.65 | 24/24 | 0 of 5 | `fail:counts,exact` |
| `offby1` | no wait, but a text request's decode position is off by one | 1.30 (as the fix) | **0/24** (up to 0.53 nats) | 0 of 5 | `fail:exact,correct` |
| `multistep` | `num_continuous_decode_steps` set to 4 inside `ServerArgs` (the arguments still say 1) | 1.31 (no gain over the fix) | 24/24 | 5 of 5 | `fail:settings` |

The independent tester got the same verdicts (`offby1` 1.307, `multistep` 1.304, `short` 1.645). `multistep` shows
what only the settings check can see: its outputs are identical. It did not help here, but a scheduler setting that
does help would be caught only by that check.

## What are the limits?

1. **Only text requests gain.** With image requests the parent has no wait on this path (0.997x in the screen, 1.007x
   in the package). A load that mixes text and image requests would gain in proportion to its text requests.
2. **The gain depends on the CPU and on host load.** The parent's waits stall the overlap scheduler, so their cost
   depends on how fast the CPU prepares the next batch. Parent reps vary by up to 1.4 % between runs; the fix's by
   0.1 %.
3. **Exact gate.** It rejects any candidate that changes numerics, even harmlessly (other kernels, fused ops). Such a
   candidate would need a tolerance gate, which this task does not calibrate.
4. **One task family with #9460** (see the top). The skill is the same: find a per-request read of a GPU scalar in
   this function. Results on the two tasks are not independent.

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory (build, CUDA pieces, model, stage) | 13-17 min (993 s for us, ~770 s for the tester) |
| One run | ~45-52 s (~38 s engine start-up incl. CUDA-graph capture up to batch 64, warm-up, 5 x 1.6-2.1 s) |
| Correctness pass | ~44 s |
| Scoring job (2 warm-ups, 2 correctness passes, 3 ABBA pairs = 10 runs) | **~7.5 min** (439-455 s GPU) |
| Calibration (25 runs) | ~19 min (1,134-1,145 s) |
| `quick` variant job | ~4 min |
| Peak VRAM | **20.5 GB** |

Disk: ~35 GB for the package root (two venvs, uv cache, CUDA tree, 4.4 GB model), ~30 GB on the stage. A host C++
compiler is needed in case FlashInfer JIT-compiles a module at start-up.

## Has someone else rebuilt it?

Yes: **PASS on a second server, no deviations** (`docs/acceptance/batch4.md`). An independent tester, an agent
that wrote none of the package, followed `package/RUN.md` alone on the 2-GPU server, with fresh downloads of
everything.
- Build: all 1,470 source files, the 166-package freeze and all 1,432 sglang files hash as ours; only
  `forward_batch_info.py` differs between the arms.
- Calibration: **1.3068** (pairs 1.306-1.309), A/A max 0.36 %, all 25 timed digests identical, correctness 24/24.
- Scoring: **1.3073, 1.3056, 1.3061**, all `ties_ref`; 439-455 s each, peak 20.5 GB.
- All 3 variants got the card's verdicts. Transfers: 8 requests 1.075 (ours 1.085), images 1.003 (ours 1.007).

## How to run it

Follow `package/RUN.md`: build the two venvs (`setup/setup_lock.sh`), fetch the CUDA pieces for FlashInfer's JIT,
download the model, stage to local disk, then `bash harness/run.sh cal_g0 cal` and `bash harness/run.sh s1_g0 score`.
The fix's scoring job should read `ties_ref`. Variants:
`bash harness/mkvariant.sh offby1 && bash harness/run.sh v_offby1 quick var_offby1`. Scoring your own tree is
described in `instruction.md`. Before giving the package to an agent, hide what names the fix (see the Rules in
`instruction.md`): the package was written for human readers and is not sealed.

## Scoring an agent

Use the kit in `package/kit/` (`package/kit/KIT.md`):
1. Calibrate on your GPU, then run `kit/set_ref.py`.
2. Build the agent's directory with `kit/make_agent_view.sh`. Give the agent only that and `instruction.md`.
3. Score its edited engine directory, or a diff, with `kit/test.sh`, which writes a reward file.

The "hide the package" advice above still applies: the agent must never see `package/`.

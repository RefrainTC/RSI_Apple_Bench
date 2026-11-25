# vLLM #27418: Qwen2-VL vision tower, patch embedding Conv3d -> Linear

On one large image per request (1344 x 1344, requests sent one at a time), the human fix makes Qwen2-VL-2B requests
**1.69x** faster on an RTX 4090 (band ±3 %, A/A up to 3.6 %). An independent tester rebuilt the package and got 1.679
(PASS-with-fixes: one memory limit was too tight, now fixed and re-verified). A second independent test on another
server also passed (8 scoring jobs, 1.660-1.730, all in band), but its calibration A/A was **9.45 % and 6.74 %**, far
noisier than on our server.

Two things to know before using this task:
- **The gain shows only on large single images.** On the load we first wrote down (32 images of 448 x 448 in one
  `generate()` call) the fix gave no gain (0.97x in the screen; 0.85x and 0.92x in the package). The task uses a second
  load, whose prior was written before it ran.
- **The gain depends on the torch / cuDNN versions**, not on the model: torch 2.9.0 with cuDNN 9.10 sends this bf16
  Conv3d to a slow native path.

Terms used below:
- **parent**: the unmodified tree, commit `88d3141e`. **fix**: the human fix, commit `42efe609`.
- **pair**: one parent run and one candidate run on the same GPU; pair speedup = parent time / candidate time. A
  scoring job runs 3 pairs in ABBA order and reports the median pair.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **band**: the range around REF that counts as a tie
  (`ties_ref`). **aa_band**: a job at or below 1 + aa_band is `no_gain`.
- **gate**: the output checks. A candidate that fails any of them scores zero.
- **teacher-forced re-read**: the parent re-reads a candidate's prompt + output in one prefill; a token's gap is the
  parent's top-1 logprob minus the logprob of the token the candidate wrote.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (driver 550, CUDA 12.4), where this task was screened,
built and calibrated, one job per GPU, each pinned to its own 8-core CCD.

## What is slow, and where does this task come from?

The vision tower of the Qwen2-VL family embeds each 3 x 2 x 14 x 14 image patch with an `nn.Conv3d` whose kernel equals
its stride, so every output is one dot product over a patch. Under torch 2.9.0 with cuDNN 9.10, this bf16 Conv3d does
not reach cuDNN; it runs torch's native `slow_conv_dilated3d` path. Measured alone on the RTX 4090: ~22 ms per 1,024
patches, against 0.07 ms as a Linear (290x). The fix computes the same products as a Linear on the flattened patch.

The PR was merged on 2025-10-24 (upstream issue vllm#27406, pytorch#166122). We picked it from our list of upstream
fixes mined from vLLM and SGLang history (batch 5, candidate C77, prior 0.50).
- **The prior's load (w1)** sent 32 images of 448 x 448 in one `generate()` call. It gave **0.97x, no gain**, although
  the profiler showed the slow Conv3d using 0.74 s of each 2.6 s rep. `LLM.generate` hands the requests to the engine
  one at a time (~50-75 ms each). The parent's engine falls behind and encodes them in a few large batches (22 steps per
  rep); the fix's engine keeps up and encodes about one image per step (104 steps). Both finish when the last request
  arrives, so the saved GPU time does not shorten the rep.
- **The second load (w2)**, written down with its prior (P = 0.75) before it ran, puts the encoder on the critical
  path: one large image per request, one request per call. It gave **1.71x** with A/A within 4 %.

## What is the load and the score?

- Model: Qwen/Qwen2-VL-2B-Instruct @ `1860c46f` (bf16). vLLM V1 offline `LLM`, `gpu_memory_utilization` 0.80,
  `max_model_len` 4096, `max_num_seqs` 32, one image per prompt, `min_pixels` = `max_pixels` = 1,806,336,
  `mm_encoder_attn_backend="TORCH_SDPA"` (on sm_89 this vLLM would pick xFormers, which is not installed).
- One rep = **4 requests, each in its own `generate()` call**. Each is one synthetic 1344 x 1344 image (seeded per rep
  and request, so no cache hits; 9,216 patches = 2,304 image tokens) + "Describe this image in detail.", 8 greedy
  tokens. One untimed warm-up call (4 images), then 5 timed reps.
- **Metric: median rep wall time.** Score = median pair speedup of a scoring job. The harness also reports
  reward = min(1, (S - 1) / (REF - 1)).

## How much does the known fix gain?

Fix median rep 1.46-1.51 s, parent 2.47-2.61 s: ~0.25 s per request, close to the ~0.2 s Conv3d estimate
(22 us x 9,216 patches) plus its share of the vision tower.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server) | **1.69** (median of the 4 jobs that set it: 1.685) | ±3 % (1.639-1.741); between-job SD 0.6 % | 0.7 % (GPU3), 3.6 % (empty-directory, one slow parent rep), 3.25 % (re-verify); screen up to 3.9 %; tester 1.24 % | 10 by the authors, 3 by the tester |
| RTX 4090 (2-GPU server) | calibrations 1.6881 and 1.7191; 8 scoring jobs 1.660-1.730 (second tester) | all inside | **9.45 % and 6.74 %** | 10 |
| RTX 5090 | not run | - | - | - |

- Jobs that set REF: calibration 1.691 (GPU3), scoring 1.700 (GPU2), empty-directory calibration 1.679 and scoring
  1.679 (GPU2).
- Later scoring jobs with the final package (GPUs 0-3): 1.685, 1.693, 1.674, 1.730. Re-verify from a new empty
  directory: calibration 1.679, scoring 1.686. All `ties_ref`.
- Single pairs vary by ±3 %, because the parent has occasional slow reps; a job's median pair is steadier.
- Transfer loads (two pairs each): 448 x 448 images, one per call (256 image tokens): **1.263 and 1.270**. The prior's
  load w1 (32 images in one call): **0.85 and 0.92**, i.e. no gain, in fact slower.

## How are outputs checked?

The Linear computes the same products as the Conv3d but sums them in another order, so image outputs are close to the
parent's but not always bit-identical: in 4 of the 5 timed reps the 8 tokens are identical in every run; in rep 2 the
fix writes a different opening, and each arm repeats its own version every time. Text-only prompts never touch the
vision tower, so they must be bit-identical. The gate (`package/harness/check.py`, `package/harness/gate.json`):
1. **counts:** per rep, output token counts and every request's prompt length after the image placeholder is
   expanded (2,304 image tokens + text) equal the paired parent's; for the image pass, they equal the calibration's.
   Any change of image resolution changes these lengths.
2. **settings:** the resolved `vllm_config` equals the paired parent's (`num_gpu_blocks` ignored).
3. **tf (teacher-forced re-read):** the parent re-reads all 20 timed requests of each candidate run and the image
   pass, with the image rebuilt from its seed. Fail if any gap exceeds `tau` = 1.0 nat, or if more than 2 % of tokens
   have a gap above 0.5 nats. In calibration the fix's worst gap was 0.125 nats, the same as the parent re-reading its
   own outputs; no token of either arm was above 0.5. Both thresholds were set before any variant ran.
4. **text:** 24 text-only prompts, one per call, 64 tokens with logprobs, bit-identical to the same job's parent pass
   (or to `package/harness/ref_correct.json`).

The **image pass** has 8 images, each a filled circle, square or triangle of one colour on a background of another, with
a question about shape and colours (48 tokens). A colour-index slip made items 3 and 7 one-colour images (shape colour =
background colour); they still test that the colour is seen, and were kept because calibration had started.

## Which broken or cheating versions are caught?

Python-only edits of the fix (`package/variants/apply.py`), one `quick` job each.

| Variant | What it does | Speedup | Re-read: worst gap; tokens > 0.5 nats | Verdict |
|---|---|---|---|---|
| `short` | requests asking for more than 4 tokens get 4 | 1.77 | 0.125; 0 % (counts and text pass fail) | `fail:counts,text` |
| `lowres` | the image processor gets a quarter of the pixels (672 x 672), whatever the engine arguments say | **5.58** | 1.875; 4.4 % (image prompts 613 tokens instead of 2,341) | `fail:counts,tf` |
| `skipimg` | the vision tower is skipped; its output is zeros of the right shape | **4.55** | 16.7; 50 % (answers "assistant assistant ...") | `fail:tf` |
| `subpatch` | patch embedding reads every second pixel (7 x 7 of each 14 x 14 patch): cheaper and wrong | 1.72 (as the fix) | **0.75; 2.5 %** (4 of 160 timed tokens; image pass 0.375, 0 %) | `fail:tf`, **by one token** |
| `cheappatch` | patch embedding uses only the first of the two temporal frames, weights summed: exact for still images, wrong for video | 1.69 (as the fix) | 0.125; 0 % | `ties_ref`: **accepted** |

The independent tester got the same verdicts and the same re-read numbers to the last digit (`subpatch`: exactly 4 of
160 tokens in each of 3 runs). The re-read is deterministic, so the `subpatch` catch is stable, but it has no margin.

## What are the limits?

1. **Large single images only.** The load is built to put the vision encoder on the critical path: one large image,
   one request at a time. With many requests in one `generate()` call (the prior's load w1) the gain disappears behind
   request hand-over (0.97x in the screen).
2. **On the prior's load the fix is no faster, and often slower.** Our two w1 transfer pairs read 0.85x and 0.92x (the
   first tester: 0.785 and 0.884; the screen: 0.97x). On the 2-GPU server one pair read 1.047x. Our reading, not
   profiled: the fix's engine runs more, smaller steps there, each with a fixed cost, and the result depends on how
   requests arrive.
3. **Version-dependent.** The gain needs torch 2.9.0 / cuDNN 9.10, which routes the bf16 Conv3d to its slow path. A
   torch where it reaches cuDNN would show ~1.0x.
7. **The A/A noise depends on the host, and can exceed `aa_band`.** On the 2-GPU server the two calibrations had A/A
   max deviations of **9.45 % and 6.74 %**, against 0.7-3.6 % on our server and an `aa_band` of 5 %. Some scoring jobs
   there contained single slow pairs (1.46-1.58); the job's median pair absorbed them, and all 8 jobs stayed in the band.
   Solo and both-GPU tests on that host were tight (1.675-1.725). A follow-up of 72 A/A pairs on that host (24 jobs,
   both GPUs) found the likely cause:
   - 6 pairs (8 %) were off by more than 3 %, none by more than 5 %. Median deviation was 0.4-0.7 %.
   - All 6 fell while the *other* GPU on the same host was starting an engine or sitting between runs: 6 of 23 such
     pairs, against 0 of 49 while the other engine was running.
   - Host load, CPU use and CPU clock showed no difference.
   - An outlier is a whole run about 3 % slower, not one slow rep.

   Our guess is that another engine's start-up (weight loading, compilation, CUDA-graph capture) competes for host memory
   or PCIe bandwidth. When two jobs share a host, stagger them so that timed runs don't overlap the other GPU's
   start-up. On a host like that, a single pair says little, and a `no_gain` call near 1.05 is not reliable; compare
   candidates on several jobs.
4. **The re-read can only reject unlikely tokens.** Within 1 nat of the top choice, a candidate may write different,
   also plausible tokens. The 8-token timed outputs are generic openings ("The image shows ..."), so the image pass
   carries most of the checking of what the model sees.
5. **`cheappatch` is accepted.** It is a legitimate optimisation for still images but wrong for video; the load has no
   video. It is not faster than the fix, because after the fix the patch embedding is no longer the bottleneck.
6. **A slightly wrong patch embed can pass.** `subpatch` is caught by one token (2.5 % against the 2 % limit; no token
   above `tau`); the image pass alone does not catch it. A candidate that degrades the image features slightly less
   could pass. Images with fine detail (text, digits) in the image pass would be a stronger check; not built.

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory | 15-20 min (870 s, 1,167 s and ~19 min in three fresh builds); flashinfer-python 0.4.1 is built from its sdist |
| One run | parent ~41 s (16 s start-up, warm-up, 5 x ~2.5 s); fix ~57 s (+22 s untimed recompilation from the parent's cache) |
| Scoring job (2 warm-ups, text passes, image pass, 3 ABBA pairs, re-read) | **~11 min** (678 s GPU) |
| Calibration | ~26 min (1,552 s) |
| `quick` variant job | ~8 min |
| Peak VRAM | **21.0 GB** in timed runs; the re-read step now peaks at 21.4 GB for the worst variant |

Disk: ~30 GB for the package root (two venvs ~8 GB each, uv cache, 4.4 GB model), ~25 GB on the stage.

## Has someone else rebuilt it?

Yes: **PASS-with-fixes** (`docs/acceptance/batch5.md`). An independent tester rebuilt the package from the
package alone on GPUs 0 and 1 of the 4-GPU server (the same server, not a second host).
- Build reproduced exactly: both freezes (159 packages) and all 1,494 vllm files hash as ours.
- Speed: calibration 1.693, scoring 1.675 and 1.679; median 1.679, all `ties_ref`. A/A 1.24 %. All checks on the fix
  pass, the text pass matches `ref_correct.json` on both GPUs, and all 6 variant runs got the card's verdicts.
- **Memory deviation.** The re-read of `subpatch` peaked at 23,782 MiB, only 45 MiB under the package's own 97 %
  watchdog: on a card with slightly less free memory the job would end with no verdict instead of `fail:tf`. The
  tester re-ran that step with a smaller memory pool for the re-read (`MEMU` 0.60 -> 0.50): peak 21,374 MiB, gaps
  unchanged to the last digit.

We made that change in the package (`loads/tf.env`, `MEMU` 0.50) and also applied the cache-copy and calibration-order
fixes found in the sibling task `fix_vllm_35442_spec_accept_count_copy` (harmless here: the first candidate run is a
warm-up, and this text pass does not depend on the cache). Re-verified from a new empty directory with a fresh stage:
build OK in 1,167 s, calibration 1.679 (A/A 3.25 %, text pass 24/24, re-read worst gap 0.125), scoring 1.686
`ties_ref`; `subpatch` with `MEMU` 0.50 still `fail:tf`, peak 21,374 MiB, 2.4 GB under the watchdog.

**Second server: PASS, with noisier A/A** (`docs/acceptance/batch5.md`). Another independent tester ran the package on
the 2-GPU server (a second RTX 4090 server with 2 GPUs), in the version from before the harness fixes (re-read memory
pool `MEMU` 0.60). No deviation from `RUN.md`.
- Calibrations **1.6881** and **1.7191**, A/A max **9.45 %** and **6.74 %** (see limit 7).
- 8 scoring jobs, all `ties_ref`: 1.6986, 1.7301, 1.6599, 1.6753, 1.6945, 1.6814, 1.6816, 1.6617.
- Variants: `subpatch` `fail:tf` (0.75 nats, 2.5 %; its re-read peaked at 23.8 GB with the old memory pool), `short`,
  `lowres` and `skipimg` fail as on our server; `cheappatch` accepted (1.7506 in one pair, `above_ref`).
- Transfers (single pairs): 448 x 448 images one per call 1.297; the prior's load w1 **1.047**, where our pairs read
  0.85 and 0.92 (see limit 2).

## How to run it

Follow `package/RUN.md`: build the two venvs (`setup/setup_lock.sh`), download the model, stage to local disk, then
`bash harness/run.sh cal_g0 cal` and `bash harness/run.sh s1_g0 score`. The fix's scoring job should read `ties_ref`.
Variants: `bash harness/mkvariant.sh subpatch && bash harness/run.sh v_subpatch quick var_subpatch`. Scoring your own
tree is described in `instruction.md`. Before giving the package to an agent, hide what names the fix (see the Rules in
`instruction.md`): the package was written for human readers and is not sealed.

## Scoring an agent

Use the kit in `package/kit/` (`package/kit/KIT.md`):
1. Calibrate on your GPU, then run `kit/set_ref.py`.
2. Build the agent's directory with `kit/make_agent_view.sh`. Give the agent only that and `instruction.md`.
3. Score its edited engine directory, or a diff, with `kit/test.sh`, which writes a reward file.

The "hide the package" advice above still applies: the agent must never see `package/`.

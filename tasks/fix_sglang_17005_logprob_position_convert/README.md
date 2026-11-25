# SGLang #17005: tokenizer manager, logprob lists extended with the new positions instead of rebuilt on every message

On 192 long requests that ask for top-20 logprobs, the human fix makes the batch **1.40x** faster on an RTX 4090
(band ±3 %, A/A up to 1.77 %; the effect is ~20x the noise). Seven jobs of the fix fall between 1.399 and 1.427. Outputs and every returned logprob are bit-identical to
the unmodified engine's. Only the authors have rebuilt it from an empty directory so far; no independent test yet.

Two things to know before using this task:
- **The gain needs logprobs on long outputs.** Without logprobs nothing changes (0.991x); with top-5 instead of top-20
  the gain falls to 1.095x. Most serving traffic asks for no logprobs.
- **The human fix is not the ceiling.** Even with the fix, a top-20 rep takes ~7 s longer than a rep without logprobs.
  A candidate that finds that time scores above REF, which the verdict allows.

Terms used below:
- **parent**: the unmodified tree, commit `075c5a57`. **fix**: the human fix, commit `339915ce`.
- **pair**: one parent run and one candidate run on the same GPU; pair speedup = parent time / candidate time. A
  scoring job runs 3 pairs in ABBA order and reports the median pair.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **band**: the range around REF that counts as a tie
  (`ties_ref`). **aa_band**: a job at or below 1 + aa_band is `no_gain`.
- **gate**: the output checks. A candidate that fails any of them scores zero.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (driver 550, CUDA 12.4), where this task was screened,
built and calibrated (GPUs 2 and 3), one job per GPU, each pinned to its own 8-core CCD.

## What is slow, and where does this task come from?

A request with `return_logprob` receives its new positions from the detokenizer in a message every 50 tokens when it
does not stream. On every message, the parent's tokenizer manager converts **all** positions so far into lists of Python
tuples (the produced token's logprob, plus the top-k pairs when asked). That is O(n) per message and O(n²) per request,
in the tokenizer manager's process: a 2,048-token request gets ~41 messages, each converting everything again. The fix
keeps the converted lists and converts only the new positions. What the engine computes does not change.

The PR was merged on 2026-01-13. We found it by mining SGLang and vLLM history for new performance fixes that run on a
4090 (batch 7, candidate M2, prior 0.45). It was the first batch-7 candidate to clear its screen with identical
outputs (1.42x, A/A band 2.5 %), so by a rule fixed in advance it became the batch's full task.

## What is the load and the score?

- Model: Qwen/Qwen2.5-0.5B-Instruct @ `7ae55760` (bf16). Offline `sglang.Engine`, triton attention, pytorch sampling,
  `max_running_requests` 256, radix cache off, `mem_fraction_static` 0.80, `SGLANG_DISABLE_CUDNN_CHECK=1` (this SGLang
  refuses to start on torch 2.9.1 with cuDNN 9.10 because of a Conv3d bug; this model has no convolution).
- One rep = **one `generate()` call with 192 requests x 64 random token ids -> 2,048 greedy tokens** (`ignore_eos`),
  every request with `return_logprob` and the **top 20** logprobs per position, no streaming. An untimed 8-request
  warm-up, then 3 timed reps (seed 1 + rep, the same in both arms). This is the screen's load, unchanged.
- **Metric: median rep wall time.** Score = median pair speedup of a scoring job. The harness also reports
  reward = min(1, (S - 1) / (REF - 1)).

## How much does the known fix gain?

Fix median rep 28.5-29.7 s, parent 40.4-41.0 s: ~12 s saved per rep.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server) | **1.40** (calibrations 1.406 and 1.400) | ±3 % (1.358-1.442) | 1.75 % (GPU3), 1.77 % (GPU2, empty directory) | 7, all `ties_ref` |
| RTX 5090 | not run | - | - | - |

- Jobs: calibration **1.406** (GPU3, pairs 1.397-1.421); empty-directory calibration **1.400** (GPU2, pairs
  1.379-1.413); empty-directory scoring **1.427** (GPU2, pairs 1.403-1.439); four more scoring jobs: **1.412** and
  **1.399** (GPU3), **1.401** and **1.409** (GPU2). The screen read 1.416.
- Across all 7 fix jobs: 1.399-1.427, **median 1.406**, between-job SD **0.7 %**; single pairs 1.379-1.448. All 7 are
  `ties_ref`, and every timed rep and correctness pass was identical to the parent's.
- REF was set from the two calibrations before the extra jobs ran. Their median, 1.406, would round to 1.41; that is a
  0.4 % difference, below the between-job SD, so the gate was left at 1.40 and the package stays identical to the one
  verified from the empty directory.
- Reps inside one run vary by up to ~5 % (parent), in both arms.
- Transfer loads (one pair each, outputs identical in every rep):
  - **top 5 logprobs instead of 20: 1.095** (parent 30.37 s, fix 27.72 s). The parent's extra time falls from ~12 s to
    ~2.6 s, because each position has fewer pairs to convert.
  - **no logprobs: 0.991** (parent 21.69 s, fix 21.89 s), the negative control: the changed function is not called.
  - Even with the fix, a top-20 rep takes ~7 s longer than one without logprobs (28.9 s against 21.9 s). That time is
    not in the conversion of the top lists: the `notop` variant skips it entirely and is no faster than the fix. Where
    it goes (computing the logprobs, or moving them between processes) was not measured.

## How are outputs checked?

The fix changes only how the returned lists are built, so the gate is exact (`package/harness/check.py`,
`package/harness/gate.json`):
1. **counts:** output tokens per request equal the paired parent's.
2. **settings:** the live `ServerArgs` equal the parent's (ports ignored).
3. **exact:** every timed rep's digest equals the paired parent's. It covers the output ids, the text, the produced
   tokens' logprobs and the top-20 lists of all 192 requests, logprobs rounded to 1e-3.
4. **correct:** 24 fixed text prompts, one per call, **256 tokens** (6 messages per request, so the cached lists are
   extended 5 times), top-5 logprobs. Ids identical and logprobs within 1e-3 of the same job's parent pass (or of
   `package/harness/ref_correct.json`), and, **for the first time in these packages, the top-5 lists are compared
   too**: same length, same token ids, logprobs within 1e-3.

In both calibrations all 15 timed digests were identical fix vs parent and parent vs parent, and the correctness pass
was 24/24 (max logprob difference 0), parent vs parent 24/24.

## Which broken or cheating versions are caught?

Python-only edits on top of the fix (`package/variants/apply.py`), one `quick` job each (1 ABBA pair + correctness
passes).

| Variant | What it does | Speedup | Correctness pass | Timed reps identical | Verdict |
|---|---|---|---|---|---|
| `short` | requests asking for more than 1,500 new tokens get 1,500 | 1.90 | 24/24 (its 256 tokens are not capped) | 0 of 3 | `fail:counts,exact` |
| `stale` | the output top-logprob list is converted on a request's first message and never extended | 1.40 (as the fix) | **0/24** | 0 of 3 | `fail:exact,correct` |
| `notop` | the output top logprobs are never converted (empty lists) | 1.41 (as the fix) | **0/24** | 0 of 3 | `fail:exact,correct` |

`stale` and `notop` are what the new top-5 comparison is for. In the earlier packages the correctness pass compared
only the produced tokens and their logprobs, so it would have passed both; here the timed digests catch them too.
Neither saves time over the fix: once only new positions are converted, converting the top lists costs little.

## What are the limits?

1. **The gain needs logprobs on long outputs.** The parent's extra cost grows with the square of a request's output
   length and with the number of top logprobs asked for. Without logprobs nothing changes (0.991x); with short outputs
   or few top logprobs the gain is small (1.095x at top 5). Most serving traffic asks for no logprobs. Evaluation and
   RL pipelines that score every token are the users this fix helps.
2. **Non-streaming only, as measured.** With `stream=True` the tokenizer manager handles a message every few tokens, so
   the parent's cost per request grows faster still. That mode was not measured.
3. **Exact gate.** It rejects any candidate that changes numerics, even harmlessly.
4. **The CPU sets the size of the gain.** The saving is Python work in the tokenizer manager's process. A faster or
   slower CPU changes REF, so a new host needs its own calibration.
5. **REF is not the ceiling.** The fix leaves ~7 s per rep in the logprob path, so a candidate can go past 1.40 and
   score `above_ref`.
6. **Not yet tested independently.** Only the authors have built and run the package, on one server.

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory (build, CUDA pieces, model, stage) | ~19 min (1,129 s, next to two screens on the same CPUs) |
| One run | 2.5-3 min (~30 s engine start-up, 2 s warm-up, 3 timed reps: parent ~41 s, fix ~29 s) |
| Correctness pass | ~50 s |
| Scoring job (10 runs) | **~23 min** (1,368 s) |
| Calibration (25 runs) | ~64 min (3,820-3,839 s) |
| `quick` variant job | ~9 min |
| Peak VRAM | **20.2 GB** |

`package/RUN.md` quotes ~80 min for a calibration and ~32 min for a scoring job (13 min per variant); the measured
times above are lower. Disk: ~35 GB for the package root (two venvs ~10 GB each, uv cache, CUDA tree, 1 GB model),
~25 GB on the stage. A host C++ compiler is needed in case FlashInfer JIT-compiles a module at start-up.

## Has someone else rebuilt it?

**Not yet.** No independent tester has run this package, and there is no RTX 5090 run. The authors rebuilt it from an
empty directory (a fresh copy of the package, fresh uv cache, interpreter, model and stage), following `package/RUN.md`
literally:
- Build: OK in 1,129 s. All 1,660 source files, both freezes and all 1,611 sglang files hash as the authors' build;
  only `tokenizer_manager.py` differs between the arms; the CUDA tree's 1,740 files match.
- Calibration **1.400** (A/A 1.77 %, all 15 timed digests identical, correctness 24/24 with the top-5 lists).
- Scoring **1.427**, `ties_ref`, all checks pass.

## How to run it

Follow `package/RUN.md`: build the two venvs (`setup/setup_lock.sh`), fetch the CUDA pieces for FlashInfer's JIT,
download the model, stage to local disk, then `bash harness/run.sh cal_g0 cal` and `bash harness/run.sh s1_g0 score`.
The fix's scoring job should read `ties_ref`. Variants:
`bash harness/mkvariant.sh stale && bash harness/run.sh v_stale quick var_stale`. Scoring your own tree is described
in `instruction.md`. Before giving the package to an agent, hide what names the fix (see the Rules in
`instruction.md`): the package was written for human readers and is not sealed.

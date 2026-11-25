# vLLM #30056: structured output with a reasoning parser, end-of-reasoning check on the new tokens only

On 64 structured-output requests that are still reasoning, the human fix makes generation **1.79x** faster on an
RTX 4090 (band ±3 %). Outputs are bit-identical to the unmodified engine's. An independent tester rebuilt the package
and got 1.790 (median of 5 jobs): PASS-with-fixes, because its first fresh calibration failed on one engine-level flip.
A second tester on another server: PASS, no flips.

Two things to know before using this task:
- **The load is built to show this cost.** Reasoning never ends in the timed reps, so the check that the PR speeds up
  runs for every request on every step. Real reasoning requests are in that state only while they think.
- **The timed load is not perfectly repeatable, in either arm.** About one run in 50 flips the digest of one timed rep:
  1 of 69 parent runs and 1 of 38 fix runs in our runs; 1 of 42 parent runs and 0 of 19 fix runs in the independent
  test. The flip is the engine's, not the fix's. The gate therefore allows one rerun of a scoring job, but not of a
  calibration, and a fresh 25-run calibration contains a flip about 4 times in 10.

Terms used below:
- **parent**: the unmodified tree, commit `67475a6e`. **fix**: the human fix, commit `c72ea107`.
- **pair**: one parent run and one candidate run on the same GPU; pair speedup = parent time / candidate time. A
  scoring job runs 3 pairs in ABBA order and reports the median pair.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **band**: the range around REF that counts as a tie
  (`ties_ref`). **aa_band**: a job at or below 1 + aa_band is `no_gain`.
- **gate**: the output checks. A candidate that fails any of them scores zero.
- **flip**: one timed rep whose output digest differs from the paired run's although nothing relevant changed.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (driver 550, CUDA 12.4), where this task was screened,
built and calibrated, one job per GPU, each pinned to its own 8-core CCD.

## What is slow, and where does this task come from?

When a request has structured output and the engine runs a reasoning parser, the grammar constrains only the answer,
after `</think>`. So on every step, for every such request still reasoning, `StructuredOutputManager.should_advance`
asks whether reasoning has just ended. The parent answers by walking the request's whole token list (prompt + output)
backwards in Python until it finds a think token; every element access is a Python call. Async scheduling is off by
default at this commit, so this sits in the engine core's serial step. On this load it costs ~11.8 ms of Python per
step, ~0.14 us per element scanned.

The PR was merged on 2025-12-09. We found it by mining vLLM and SGLang history for new performance fixes that run on a
4090 (batch 6, candidate N1, prior 0.55). Its screen passed clearly (1.78x, A/A 1.3 %), so it became a full task.

## What is the load and the score?

- Model: Qwen/Qwen3-0.6B @ `c1899de2` (bf16; `<think>` = 151667, `</think>` = 151668). vLLM V1 offline `LLM`,
  `gpu_memory_utilization` 0.80, `max_model_len` 2048, `max_num_seqs` 64, prefix caching off,
  `reasoning_parser="qwen3"`; structured output through vLLM's default backend for a JSON schema (xgrammar).
- One rep = **64 requests x 1,024 random token ids -> 512 greedy tokens** (`ignore_eos`) in one `generate()` call.
  Every request has a small JSON schema (`{"answer": string}`) and `logit_bias` -100 on `<think>` and `</think>`, so
  reasoning never ends and the check runs for all 64 requests on every step, over sequences growing from 1,024 to
  1,536 tokens. One untimed warm-up, then 3 timed reps on fresh prompts.
- **Metric: median rep wall time.** Score = median pair speedup of a scoring job. The harness also reports
  reward = min(1, (S - 1) / (REF - 1)).

## How much does the known fix gain?

Fix median rep 7.63-7.68 s, parent 13.66-14.00 s: ~6 s per rep over 512 steps.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server) | **1.79** (calibrations 1.789 and 1.791) | ±3 % (1.736-1.844) | 0.46 % (GPU3), 2.47 % (GPU2, empty directory), 2.54 % (re-verify) | 8 |
| RTX 5090 | not run | - | - | - |

- Jobs: calibration 1.789 (GPU3); empty-directory calibration 1.791 and scoring 1.805 (GPU2); scoring 1.787
  (`fail:exact` from one flipped rep, see below), 1.791 and 1.802; re-verify after the harness fixes, calibration 1.773
  and scoring 1.761 (GPU3). Every job that passed the gate is `ties_ref`.
- On GPU2 whole parent runs shift by 1-2.5 % (for example 14.0 s against 13.65 s per rep), while reps inside a run agree
  within 0.5 %. `aa_band` 3 % was set after seeing that (post hoc; the provisional gate had 2 %).
- Transfer loads (one pair each): 8 requests per rep **1.347** (parent 2.92 s, fix 2.17 s), digests identical; above
  the predicted 1.15-1.3, because the step is shorter too. No structured output (`nogrammar`) **1.000** (7.60 s both),
  the negative control: plain requests never call the check.

## How are outputs checked?

The fix changes how the end of reasoning is detected, not what is generated, so the gate is exact
(`package/harness/check.py`, `package/harness/gate.json`):
1. **counts** per rep equal the paired parent's.
2. **settings:** the resolved `vllm_config`, including `structured_outputs_config` (where the reasoning parser lives),
   equals the parent's (`num_gpu_blocks` ignored).
3. **exact:** every timed rep's digest equals the paired parent's.
4. **correct:** 24 short questions in Qwen3's chat format with thinking on (for example "What is the capital of France?
   Reply in JSON with the field answer."), each with the same JSON schema and no `logit_bias`, one per call, 384 greedy
   tokens with logprobs. Identical to the same job's parent pass (or to `package/harness/ref_correct.json`).

**The timed load cannot check correctness.** It never ends reasoning by design, so a candidate that skipped the check
entirely would produce exactly the parent's timed outputs (`noadvance`, below). The correctness pass is what exercises
the changed path: the model thinks, writes `</think>`, and from then on the grammar constrains the answer. In every
calibration all 24 prompts ended their reasoning, in both arms. Every candidate run starts from a copy of the parent's
compile cache.

**The rare flip, and the one-rerun rule.** Counted over all the package's own-load runs (calibration, scoring and smoke
jobs, three package copies): 1 flipped run of 69 in the parent arm (1 of 205 reps), and 1 of 38 in the fix arm (1 of
114 reps). The parent's flip was in an A/A pair, both runs the unchanged parent. A flip also appeared in the `nogrammar`
transfer, which runs neither the fix's code nor any grammar. In the flipped rep we examined, 7 of 64 requests diverge,
some at their 2nd token, and all keep writing free text; none switched to JSON, as a wrongly ended reasoning phase would.
So the flip is an engine-level numeric event. Two explanations, both untested: the engine core starts stepping while the
64 requests are still arriving, so the first batches can differ; or the asynchronous grammar compilation races with the
scheduler. `check.py` therefore returns `rerun:exact` in scoring mode when counts, settings and the correctness pass
all pass and exactly one rep of one run differs; the job is repeated once under a new label, and that verdict is final.
Calibration mode has no rerun rule.

## Which broken or cheating versions are caught?

Python-only edits on top of the fix (`package/variants/apply.py`), one `quick` job each (1 ABBA pair + correctness
passes).

| Variant | What it does | Speedup | Verdict | Caught by |
|---|---|---|---|---|
| `noadvance` | `should_advance` answers "not yet" without looking, so the grammar is never applied | 1.789 | `fail:correct` | **only the correctness pass** (0/24 identical); the timed outputs equal the parent's |
| `short` | requests asking for more than 400 tokens get 400 | 2.34 | `fail:counts,exact` | counts and exact; the 384-token correctness pass is unaffected |
| `noparser` | the reasoning parser is dropped after argument parsing, so the grammar constrains every token | 1.76 | `fail:exact,correct,settings` | settings (`structured_outputs_config.reasoning_parser`), exact, correctness (0/24 end reasoning) |

`noadvance` is the important one: a candidate can delete the check outright, and only the correctness pass sees it.

## What are the limits?

1. **The load is built to show this cost.** Reasoning that never ends, 64 structured-output requests, sequences of
   1-1.5k tokens. With real reasoning models a request is in this state only while it thinks; after `</think>` the
   parent's check stops too (`reasoning_ended` is cached). The gain grows with batch size x sequence length x thinking
   steps; the transfer loads show how it shrinks.
2. **Only structured-output requests with a reasoning parser pay.** Plain requests never call the check (`nogrammar`,
   1.000x).
3. **Exact gate.** It rejects any candidate that changes numerics, even harmlessly. Such a candidate would need a
   tolerance gate, which this task does not calibrate.
4. **Digests depend on the compile cache.** The package's runs give different digests than the screen's (another
   torch.compile cache, other tuned kernels), but fix and parent agree within a job because candidate runs start from a
   copy of the parent's cache.
5. **A rare false fail.** The timed load flips one rep in about one run of 50, in either arm (ours: 1/69 parent runs,
   1/38 fix runs; the independent tester's: 1/42 parent runs, 0/19 fix runs; the second server: 0 flips in 123 timed
   reps). With the one-rerun rule a correct candidate is rejected only if its rerun flips too. The cause is untested.
   Calibration mode has no rerun rule, and at ~1 run in 50 a 25-run calibration contains a flip about 4 times in 10:
   **a fresh calibration can end `fail:exact` with nothing wrong** (it did in the independent test). Repeat it, or
   judge it on fix-vs-parent digests only.

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory (build, model, stage) | 12-18 min (760 s and 1,091 s in our fresh builds, 703 s for the tester) |
| One run | parent ~64 s, fix ~47 s (~17 s engine start-up incl. torch.compile and CUDA graphs, 0.5 s warm-up, 3 timed reps) |
| Correctness pass | ~36-52 s |
| Scoring job (2 warm-ups, 2 correctness passes, 3 ABBA pairs = 10 runs) | **~8.6 min** (505-518 s) |
| Calibration (25 runs) | ~24 min (1,425-1,449 s) |
| Peak VRAM | **20.1 GB** (20,110 MiB) |

Disk: ~25 GB for the package root (two venvs ~8 GB each, uv cache, 1.5 GB model), ~20 GB on the stage.

## Has someone else rebuilt it?

Yes: **PASS-with-fixes, final** (`docs/acceptance/batch6.md`). An independent tester, an agent that wrote none of this
material, rebuilt the package from an empty directory on the 4-GPU server (GPUs 0 and 1).
- Build: reproduced exactly in ~12 min (both freezes, 156 packages, and all 1,673 vllm files of each venv hash as ours).
- Jobs: calibration 1.7899, second calibration 1.7895, scoring 1.7909 and 1.7852 (GPU0), scoring 1.7962 on GPU1 with
  empty caches. **Median of 5 jobs 1.790**, against REF 1.79 ±3 %. Fix rep 7.632-7.663 s, parent 13.57-13.79 s.
  Peak 20,110 MiB.
- A/A max 2.22 % (first calibration) and 1.00 % (second).
- Correctness pass 24/24, logprob difference 0, in every job on both GPUs, and equal to `ref_correct.json`.
- **The first fresh calibration ended `fail:exact`.** In one A/B pair the *parent* run flipped one timed rep: 8 of 64
  requests diverged (at output positions 4 to 82) and all kept writing free text. Every other timed digest was
  identical. `check.py` applies the one-rerun rule only to scoring and `quick` jobs, so one engine-level flip fails a
  calibration. The second calibration passed.
- Flip counts in this test (official load, both GPUs, digests identical across GPUs): parent 1/42 runs (1/126 reps),
  fix 0/19 runs (0/57 reps), warm-ups 0/13.
- Variants: `noadvance` 1.794 `fail:correct`, `noparser` 1.779 `fail:exact,correct,settings`, `short` 2.326
  `fail:counts,exact`, all as ours. Transfers: 8 requests 1.356 (ours 1.347), no structured output 0.999 (ours 1.000).
- The tester's suggested fixes, not yet in the package: in calibration, judge `exact` on fix-vs-parent pairs only and
  report parent-parent flips separately, or apply the one-rerun rule there too; and correct `RUN.md`'s time estimate for
  this task (~40 + 18 min) to the measured ~24 + 9 min.

**Second server: PASS** (`docs/acceptance/batch5.md`). Another independent tester ran an earlier version of the package
on the 2-GPU server (a second RTX 4090 server with 2 GPUs): calibration **1.7878**, 12 scoring jobs **1.778-1.806**, all
in band, A/A 2.04 %. **No digest flip in any arm**: 123 fix-vs-parent timed reps and 15 parent-vs-parent reps identical;
correctness 24/24 in all 13 jobs. All three variants got our verdicts; transfers 8 requests 1.371, no structured output
0.999. No deviation from `RUN.md`.

**Harness fixes already applied** (found by the independent test of `fix_vllm_35442_spec_accept_count_copy`, same
harness): the cache copy for the first candidate run can no longer fail silently on a fresh stage, and calibration runs
the warm-ups before the passes it compares. Our own empty-directory calibration had passed anyway (24/24), because this
task's correctness pass came out the same with every cache. The timed path did not change. Re-verified from a new empty
directory with a fresh stage: calibration 1.773, scoring 1.761 `ties_ref`, correctness 24/24. That re-verify
calibration also saw the first parent-arm flip (in an A/A pair).

## How to run it

Follow `package/RUN.md`: build the two venvs (`setup/setup_lock.sh`), download the model, stage to local disk, then
`bash harness/run.sh cal_g0 cal` and `bash harness/run.sh s1_g0 score`. The fix's scoring job should read `ties_ref`;
on `rerun:exact`, run it once more under a new label. Variants:
`bash harness/mkvariant.sh noadvance && bash harness/run.sh v_noadvance quick var_noadvance`. Scoring your own tree is
described in `instruction.md`. Before giving the package to an agent, hide what names the fix (see the Rules in
`instruction.md`): the package was written for human readers and is not sealed.

## Scoring an agent

Use the kit in `package/kit/` (`package/kit/KIT.md`):
1. Calibrate on your GPU, then run `kit/set_ref.py`.
2. Build the agent's directory with `kit/make_agent_view.sh`. Give the agent only that and `instruction.md`.
3. Score its edited engine directory, or a diff, with `kit/test.sh`, which writes a reward file.

The "hide the package" advice above still applies: the agent must never see `package/`.

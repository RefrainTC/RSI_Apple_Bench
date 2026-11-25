# vLLM #35442: speculative decoding on a hybrid model, accepted-token counts copied without blocking

With MTP speculative decoding on Qwen3.5-4B, the human fix makes batch-1 decoding **1.037x** faster on an RTX 4090
(band ±1 %). The gain is small but steady: ten jobs of the fix by the authors fall between 1.036 and 1.042. Outputs are
bit-identical to the unmodified engine's. An independent tester rebuilt the package and reproduced the number
(PASS-with-fixes: it found two harness defects, both fixed and re-verified from an empty directory). A second
independent test on another server also passed, with scoring jobs of 1.0383, 1.0368 and 1.0376.

Two things to know before using this task:
- **The timed load is a weak check of the changed code.** Its random-id prompts accept 99 % of the MTP drafts; natural
  text accepts 80 %. A broken version that never updates the accepted counts gives the parent's exact timed outputs.
  Only the 24-prompt natural-text pass catches it.
- **The task uses an adjusted load.** The screen's original load (4 requests per rep) showed the same gain but failed
  the screen's 3-of-3 rule once. The task uses the screen's batch-1 load instead.

Terms used below:
- **parent**: the unmodified tree, commit `4beebfd1`. **fix**: the human fix, commit `ad9d09e2`.
- **pair**: one parent run and one candidate run on the same GPU; pair speedup = parent time / candidate time. A
  scoring job runs 3 pairs in ABBA order and reports the median pair.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **band**: the range around REF that counts as a tie
  (`ties_ref`). **aa_band**: a job at or below 1 + aa_band is `no_gain`.
- **gate**: the output checks. A candidate that fails any of them scores zero.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (driver 550, CUDA 12.4), where this task was screened,
built and calibrated, one job per GPU, each pinned to its own 8-core CCD.

## What is slow, and where does this task come from?

With speculative decoding, after every step vLLM's model runner works out how many draft tokens each request
accepted. At the parent commit, on a hybrid (Gated-DeltaNet) model, it copies these counts to the CPU with a blocking
call and then loops over them in Python. The CPU has to wait for the GPU to finish the step, so it cannot prepare the
next step while the GPU works, although async scheduling is on. The cost is about 0.25-0.5 ms per decode step.

The PR was merged on 2026-03-03. We picked it from our list of upstream fixes mined from vLLM and SGLang history
(batch 5, candidate C15, prior probability of a stable gain 0.35). On the prior's load (4 requests per rep) the gain
was real (~1.04x, every fix rep faster than every parent rep) but the parent's third rep has two speeds, so one of
the first three pairs had a fix/parent time ratio of only 0.988 and the screen rule (all 3 pairs at least ~2 % below 1
and outside the A/A range) failed once. At 1 request per rep the gain was the same size and the A/A
noise ten times smaller, so the task uses that load.

## What is the load and the score?

- Model: Qwen/Qwen3.5-4B @ `851bf6e8` (bf16). vLLM V1 offline `LLM`, `gpu_memory_utilization` 0.80,
  `max_model_len` 4096, prefix caching off, `speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 1}`
  (vLLM maps the method to `mtp` and turns async scheduling on).
- One rep = **1 request of 256 random token ids -> 512 greedy tokens** (`ignore_eos`). An untimed 1-request warm-up,
  then 3 timed reps on fresh prompts (seeds 1-3, the same in both arms).
- **Metric: median rep wall time.** Score = median pair speedup of a scoring job (3 ABBA pairs). The harness also
  reports reward = min(1, (S - 1) / (REF - 1)).

## How much does the known fix gain?

Fix median rep 3.59-3.62 s, parent 3.73-3.76 s (one parent run at 3.86 s): about 0.13 s per 512-token rep.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server) | **1.037** (median of the 5 jobs that set it: 1.0364) | ±1 % (1.027-1.047), the floor; between-job SD 0.13 % | 0.14 % and 0.44 % (authors); 1.42 % and 0.30 % (independent tester) | 10 by the authors, 4 by the tester |
| RTX 4090 (2-GPU server) | scoring jobs 1.0383, 1.0368, 1.0376; warm calibration 1.0368 (second tester) | inside | 0.26 % and 0.28 % | 5 (incl. one calibration failed by the cache-copy defect) |
| RTX 5090 | **1.107** (median of 3 jobs: calibration 1.1125, scoring 1.1070 and 1.1061), with the memory utilization lowered from 0.80 to 0.60 | not set (`gate.json` holds the 4090 REF, so these read `above_ref`) | 2.94 % (calibration); an A/A scoring job read 0.999 (pairs up to 1.98 %) | 3 |

- **RTX 5090: the gain triples (10.7 % against 3.7 %), and the package needs one change.** At the shipped memory utilization of 0.80, the card reached 31.1 GB during CUDA-graph capture, so the 5090 runs used 0.60 (peak 24.8 GB). Both variants were caught there too: `short` 1.419 `fail:counts,exact`, `stale` 1.0996 `fail:correct`. Set a 5090 REF (for example with `kit/set_ref.py`) before scoring on that card.

- Jobs that set REF: calibration 1.0378 (GPU1), scoring 1.0393 (GPU2) and 1.0364 (GPU3), empty-directory
  calibration 1.0361 and scoring 1.0362 (GPU0).
- Later scoring jobs with the final package: 1.0369, 1.0371, 1.0422. Re-verify after the tester's fixes, from a new
  empty directory: calibration 1.0365, scoring 1.0377. All `ties_ref`.
- Expect an A/A of about 1.5 % now and then: the tester saw one whole parent run 1.3 % slow. Our own A/A figures are on
  the optimistic side. This matters only for `no_gain` decisions near 1.01.
- Transfer loads (one pair each unless noted): natural-text essay, 1 request per rep (80 % of drafts accepted),
  **1.0385 and 1.0372**, outputs bit-identical; 4 requests per rep 1.048; 16 requests 1.052; the official load with
  vLLM's statistics on 1.0375 (768 of 775 drafts accepted). At 4 and 16 requests the outputs of both arms vary run to
  run (batch shapes under async scheduling), so their digests are not compared. On the 2-GPU server the 4-request
  transfer read only 1.007 in a single pair; the batch-1 task load reproduced there (see "Has someone else rebuilt it?").

## How are outputs checked?

The fix changes only how the accepted counts reach the CPU, not what is computed, so the gate is exact
(`package/harness/check.py`, thresholds in `package/harness/gate.json`):
1. **counts:** output token counts per rep equal the paired parent's.
2. **settings:** the resolved `vllm_config` (with `speculative_config`) equals the paired parent's;
   `cache_config.num_gpu_blocks` is ignored because it follows the memory profile.
3. **exact:** every timed rep's sha256 digest of the output ids equals the paired parent's.
4. **text pass:** 24 fixed natural-text prompts, one per call, 64 greedy tokens with the chosen token's logprob. Ids
   identical and every logprob within 1e-3 nats of the same job's parent pass (or of `package/harness/ref_correct.json`).

**Compile caches.** Inductor tunes Triton configurations on first use and stores the winner. Two caches tuned in
separate processes can keep different winners, which sum in a different order. With per-arm caches the fix's text
pass first differed from the parent's (up to 0.12 nats). So every candidate run now starts from a fresh copy of the
parent's compile cache on that host and GPU; since then every fix pass has been bit-identical to its parent's.
- There are several numeric "families" of the text pass, depending on the cache. The tester found four; any two share
  only 14-18 of 24 token sequences.
- `ref_correct.json` is therefore not portable: one of the tester's GPUs reproduced it, the other matched 16/24. The
  real check is the comparison with the same job's parent pass; the reference file is only a fallback.

## Which broken or cheating versions are caught?

Python-only edits on top of the fix (`package/variants/apply.py`), one `quick` job each (1 pair + text passes).

| Variant | What it does | Speedup | Text pass | Timed reps identical | Verdict |
|---|---|---|---|---|---|
| `short` | requests asking for more than 400 tokens get 400 | 1.32 | 24/24 | 0 of 3 | `fail:counts,exact` |
| `stale` | the CPU copy of the accepted counts is never updated (stays at 1) and is uploaded back every step | 1.03 (as the fix) | **0/24** (logprobs up to 0.87 nats apart) | **3 of 3** | `fail:correct` |
| `spec2` | `num_speculative_tokens` set to 2 inside `SpeculativeConfig` (the arguments still say 1) | **1.36** | 20/24 | 3 of 3 | `fail:settings,correct` |

The independent tester got the same verdicts: `stale` 1.036 (0/24, 0.83 nats), `spec2` 1.361, `short` 1.326. The
0.83-nat gap of `stale` is far above the 0.12-nat spread between cache families, so that catch is robust.
`spec2` is faster for real; only the settings check and the text pass stand between it and a free 1.36x from a
configuration change.

## What are the limits?

1. **Small, CPU-dependent gain.** About 3.8 %, 0.13 s per 3.7 s rep. It comes from CPU/GPU overlap, so it depends on
   the CPU and on host load. Each job was pinned to its own CCD, next to other GPU jobs on the other CCDs.
2. **The timed load does not guard the changed path.** Its prompts are random ids. They accept 99 % of drafts, but
   their repetitive outputs did not change when the accepted counts were wrong (`stale`). Only the text pass
   (24 prompts x 64 tokens of natural text) guards the code the PR changes. Why `stale` leaves the repetitive outputs
   unchanged is an untested guess (the repetition survives the wrongly advanced Gated-DeltaNet state).
3. **Acceptance on the timed load is not realistic.** The timed load accepts 99 % of MTP drafts (repetitive text);
   natural text accepts 80 %. A candidate tuned to near-perfect acceptance, for example work that only pays off when
   every draft is accepted, may not transfer to real prompts. The evidence that the human fix transfers is the
   natural-text transfer load: 1.038x and 1.037x, outputs bit-identical.
4. **Exact gate.** It rejects every candidate that changes numerics, even harmlessly (other kernels, fused ops). Such
   candidates would need a tolerance gate, which this task does not calibrate.
5. **A possible race in the fix.** The fix reads the pinned CPU copy without waiting for the copy to finish. If that
   read were ever stale under async scheduling, the next step's state would use a wrong count. Every fix run here was
   bit-identical to the parent, so on this load the race does not show.

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory (build, model, stage) | 12-20 min (695 s, 1,159 s and ~17.5 min in three fresh builds) |
| One run | ~75 s (~57 s engine start-up incl. torch.compile and CUDA graphs, warm-up, 3 x ~3.7 s) |
| Text pass | ~78 s (~175 s on an empty compile cache) |
| Scoring job (2 warm-ups, 2 text passes, 3 ABBA pairs = 10 runs) | **13-16 min** |
| Calibration (25 runs) | ~34 min |
| `quick` variant job (6 runs) | ~8 min |
| Peak VRAM | **20.6 GB** (20,608 MiB in every tester job) |

Disk: ~35 GB for the package root (two venvs, uv cache, 8.8 GB model), ~30 GB on the stage.

## Has someone else rebuilt it?

Yes: **PASS-with-fixes** (`docs/acceptance/batch5.md`). An independent tester, an agent that wrote none of this
material, rebuilt the package from the package alone on GPUs 0 and 1 of the 4-GPU server (the same server, not a
second host).
- The build reproduced exactly: both freezes (179 packages) and all 1,998 vllm files of each venv hash as ours.
- Speed: 1.0374 (calibration), 1.0393 and 1.0377 (scoring), 1.0360 (second calibration); median 1.0377, all inside the
  band. Natural-text transfer 1.036.
- **Its first calibration on a fresh install failed its own gate** (`fail:correct`). Two harness defects caused it:
  - on a fresh stage the cache directory for the first candidate run did not exist, so the copy of the parent's cache
    failed unnoticed and the checked text pass ran on an empty cache;
  - calibration compared a cold first parent pass, which differs even from later parent passes (18/24).

  A second calibration on warm caches passed 24/24.

We fixed both the same day: the cache directory is made and a failed copy stops the job, and calibration now runs the
warm-ups before the passes it compares. We then re-verified the package from a new empty directory with a fresh stage:
build OK in 1,159 s, calibration 1.0365 (A/A 0.31 %, text pass 24/24 fix vs parent and parent vs parent, all 15 timed
digests identical), scoring 1.0377 `ties_ref`. No copy error appears in any log. Other findings were also fixed:
`job.sh` now finds its own processes by tag and package root, `MANIFEST.sha256` covers `PROVENANCE.md`, and `RUN.md`
explains the text-pass families and CPU sets.

**Second server: PASS** (`docs/acceptance/batch5.md`). Another independent tester ran the package on the 2-GPU server
(a second RTX 4090 server with 2 GPUs), in the version from before the two harness fixes.
- Its first calibration (1.0379) hit the same `fail:correct` from the cache-copy defect (parent vs parent 18/24 in the
  text pass). A warm calibration then read **1.0368** and passed.
- Scoring jobs: **1.0383, 1.0368, 1.0376**, all `ties_ref` (band 1.027-1.047). A/A 0.26 % (fresh) and 0.28 % (warm).
- Variants: `stale` `fail:correct` (0.87 nats), `short` `fail:counts,exact`, `spec2` `fail:correct,settings`, as on our
  server. No deviation from `RUN.md`.
- Transfers (single pairs): natural text 1.036; 4 requests per rep **1.007** (ours 1.048); 16 requests 1.045.
- The tester judged it a PASS given the cache-copy fix, which the current package has.

## How to run it

Follow `package/RUN.md`: build the two venvs (`setup/setup_lock.sh`), download the model, stage to local disk, then
`bash harness/run.sh cal_g0 cal` and `bash harness/run.sh s1_g0 score`. The fix's scoring job should read `ties_ref`.
Variants: `bash harness/mkvariant.sh stale && bash harness/run.sh v_stale quick var_stale`. Scoring your own tree is
described in `instruction.md`. Before giving the package to an agent, hide what names the fix (see the Rules in
`instruction.md`): the package was written for human readers and is not sealed.

## Scoring an agent

Use the kit in `package/kit/` (`package/kit/KIT.md`):
1. Calibrate on your GPU, then run `kit/set_ref.py`.
2. Build the agent's directory with `kit/make_agent_view.sh`. Give the agent only that and `instruction.md`.
3. Score its edited engine directory, or a diff, with `kit/test.sh`, which writes a reward file.

The "hide the package" advice above still applies: the agent must never see `package/`.

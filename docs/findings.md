# What makes scores fragile and verifiers wrong

This page collects everything we found that can make a task's score misleading: verifiers that can be gamed or that reject correct answers, environment effects, engines that are not repeatable, human reference fixes that may be wrong, and harness bugs that only an independent rebuild found. Each item says what happened and what to do about it. The first four matter most.

Terms: *pristine* or *parent* is the unmodified (slow) code; the *oracle* or *human fix* is the reference answer; *A/A* scores the unmodified code against itself to measure noise; *REF* is the human fix's calibrated speedup; the *gate* is the set of correctness checks a candidate must pass before its speed counts.

## 1. Can a verifier be gamed without fixing anything?

Yes. Apple's MM task (`regression_multimodal_prefill_ttft`) can be.

**What happened.**
- The slowdown is a Triton kernel recompiled for every new prompt length. The verifier checks that the scored window opens with few new compiles, but it counts compiled kernels only in one anchor workload, and it allows up to 3× the start-up count.
- The verifier also passes its random seed to the server under test, and the client's image sizes are a pure function of that seed.
- A candidate that does not fix the bug can use the seed to pre-compile, in about 8 seconds at start-up, exactly the kernel variants the scored prompts will need. It calls the kernel directly, so the verifier's wrapper never sees it.
- That passes every check and earns **64%** of the real fix's reward (4090, reward 0.239 against 0.373).
- Without the seed, pre-compiling every possible prompt length for one image (only 109 exist) still earns about **46%**.
- Only a diagnostic counter flags these candidates, and it does not vote. We reproduced the hole on the 4090 and the 5090. It does not depend on the GPU, so it very likely works on Apple's H100 too, though we did not test that.

**What to do.** Removing the seed is not enough. Any of these closes the hole: let the existing "was the bug actually repaired" counter vote; count compiled kernels at every scored point, not just the anchor; or tighten the start-up allowance toward 1×. Any task used to train or reward an agent should first be attacked with cheating versions like this.

## 2. Can the human reference fix itself be wrong?

It can, and a benchmark built on it would then reward reproducing a bug.

**What happened (SGLang #10709, candidate N4).** The fix caches compiled Triton kernels for LoRA serving and gave 1.14× in our screen. But its outputs differ from the parent's far beyond rounding:
- 123-124 of 128 requests differ in every rep;
- with the same context, the chosen token's log-probability differs by a median of 0.8 nats (maximum 4.6), where bf16 rounding is about 0.05;
- parent against parent: 0 requests differ, difference 0.

Our untested guess is that the fix keys its kernel cache only on `(NUM_SLICES, BLOCK_M)` and reuses a kernel that Triton specialised for other argument values. We did not check which arm is right, and did not use N4 as a task. See [selection/mined-fixes.md](selection/mined-fixes.md).

**What to do.** Before adopting a merged fix as the reference, compare its outputs with the parent's at the token and log-probability level, not just by digest. A fix that changes outputs beyond rounding needs an explanation first.

## 3. Does the environment move scores?

A lot, often more than the differences between candidates.

| Effect | Example | Size |
|---|---|---|
| Where compile caches live | MM on the same 4090, same code and seed: cache and temp files on local disk vs network storage | 1.39× vs 1.65× |
| Same GPU model, different server | Task C on two 4090 servers | +1.0-1.3%, outside its 0.16% A/A band |
| Host load | Φ-Bench on a quieter 5090: the CPU-bound baseline gains most from a quiet host, so ratios shrink | authors' fix 1.363 → 1.344; configuration 1.609 → 1.55 |
| Concurrent setup work on the same host | venv builds, 7-20 GB copies and hashing next to timed jobs | single A/A pairs of 1.28 (#14857) and 4.2% (#12380); one job 0.01% below its band (#14930) |
| A CPU-bound parent | #12380's parent runs without CUDA graphs | parent A/A up to 27% when its CPU cores are shared; jobs drift to the top of the band on a busy host |
| Compile-cache state | vLLM #37338 (C19): with a warm Triton cache the fix gains nothing; with a cold one the first batch is 36× faster | 1.00× vs 36× |
| Different GPU | MM: H100 1.62×, 5090 1.55×, 4090 1.39× | see section 4 |
| The first run on a fresh install | vLLM #35442: a cold first parent run computes differently from later warm runs (section 5) | gate failure |

The ranking of tasks carries over between machines better than the numbers do.

**What to do.** Store the GPU, driver, engine version, cache location, host and host load with every score. Measure each task's REF on each machine you score on. Keep the host quiet during timed jobs: no builds, copies or installs. Pair runs (ABBA) and use the median over several pairs.

## 4. Do speedups carry over between the RTX 4090 and RTX 5090?

No, and they move in both directions. Every task that runs on both needs its own REF per GPU.

| Task | 4090 REF | 5090 REF | Direction |
|---|---|---|---|
| Apple MM | 1.39 | 1.55 | higher on 5090 |
| Apple ST (adapted) | 1.14 | 1.19 (Triton) / 1.22 (FlashInfer) | higher |
| SGLang #16521 | 3.83 | 3.52 | lower |
| SGLang #20972 | 1.142 | 1.226 | higher |
| vLLM #37338 (cold start) | 1.29 | 1.18 | lower |
| SGLang #27868 | 1.028 | 1.058 | higher |
| SGLang #14956 | 1.31 | 1.49-1.51 | higher |
| Task C | 1.32 | 1.32 | same |

Two causes we can name. A fixed host cost per step (a sync, a blocking copy) is a larger share of a shorter step, so fixes like ST, #20972 and #14956 gain more on the faster 5090. A cold-start task is set by the host CPU, not the GPU: the 5090's workstation CPU is slower, the rest of start-up takes longer, and #37338's ratio drops although it saves about the same 48 seconds. On the 5090, ST's score also depends on which attention backend is used. Details in [hardware.md](hardware.md).

## 5. Is the engine itself repeatable on one load?

Not always. A gate that demands bit-identical outputs can then fail an honest candidate, or even the unmodified code.

- **vLLM #30056 (N1).** In 69 parent runs and 38 fix runs, one run of each arm had one rep whose outputs differed early for a few requests (7 of 64 in one case). A control load that runs neither the fix nor any grammar flipped once too. So it is the engine's, not the fix's. Two untested explanations: request arrival timing into vLLM's separate engine process, and a race with grammars compiled in a thread pool. The gate now allows one rerun when only the exact-output check fails. **An independent fresh-install calibration still ended `fail:exact`** from a single flip in a parent run, because calibration mode has no rerun rule (see [acceptance/batch6.md](acceptance/batch6.md)).
- **SGLang arrival grouping.** SGLang groups requests by arrival time, so timed outputs of #6994, #9460 and #14956's image load are not reproducible even parent against parent. Their correctness checks use separate untimed passes.
- **Autotuning.** With an empty Triton cache every run autotunes again and may pick different kernel configurations, which changes bf16 rounding (#37338).
- **vLLM #35442: several numeric "families".** An independent tester found four distinct families of results for the same text prompts, depending on which compile cache a run used (cold, warm on one GPU, warm on another, fix on an empty cache). Any two share only 14-18 of 24 token sequences. A stored reference pass (`ref_correct.json`) matched exactly on one of the tester's GPUs and not on the other.
- **Bad compile caches.** With vLLM 0.19.1, 2 of 9 compile caches in task C produced slightly wrong outputs and failed the correctness check, while generated text looked fine; in the MTP adaptation 2 of 11 caches broke a log-probability probe. This only ever fails honest candidates. Rebuild the cache and rerun.

**What to do.** Measure parent-against-parent repeatability on the exact load before choosing an exact gate. Where it is not perfect, use a tolerance gate, an untimed correctness pass, or a stated rerun rule, and give the observed flip rate on the card.

## 6. Can harness bugs hide until someone else rebuilds the task?

Yes. Several were found only because an independent tester rebuilt a package from an empty directory on a fresh machine.

- **The cache-copy bug (vLLM #35442 and #27418).** Each candidate run should start from a copy of the parent's compile cache. On a fresh stage the copy's parent directory did not exist, `cp` failed with "No such file or directory", nothing checked it, and the run went ahead on an empty cache. In calibration mode that run was exactly the one whose outputs are checked, so a fresh install failed its own correctness check (`fail:correct`). The authors' own empty-directory run had been lucky: its cold passes happened to land in the same numeric family. Fixed by creating the directory, stopping the job if the copy fails, and running warm-ups before the correctness passes. See [acceptance/batch5.md](acceptance/batch5.md).
- **Audit false positives (task C).** With a uv venv, `venv/bin/python` goes through a symlink, and the audit used `abspath` instead of `realpath`, so vLLM's normal read of a CUDA library became 150 "read_protected" events and a voided score. Separately, an honest temperature-1 output of 128 whitespace tokens counted as empty because the check used `.strip()`, which voided one of four scorings.
- **A variant mis-described (SGLang #9460).** The card said the `lowres` variant is slower; on the tester's server it was faster than the fix (1.28×), and was still caught.
- **Job-tag collisions.** A package found "its own" processes by a tag built from default labels, so two users running the same package with default labels on one host could kill each other's runs.
- **The client's token count (Apple ST).** The tester's A/A read 0.975 instead of about 1.0. One request with 2,000 tokens had been counted as 1,140 by re-tokenizing streamed text.

The authors' own empty-directory runs found others before any outside test: a settings file that silently overrode the caller's GPU choice; a lock generator that dropped `vllm-flash-attn` (the #7822 score fell from 1.48 to 1.41 on a fallback backend); a missing post-calibration step; fresh venvs that wrote bytecode during runs and tripped the venv-integrity check; and a dependency cutoff read in local time (see [hardware.md](hardware.md)).

**What to do.** Treat an independent rebuild on a different machine, from the package alone, as part of building a task. Check every copy and cache step for silent failure. Make sure the run that is checked for correctness is never the first, cold run.

## 7. Can a verifier give a correct answer zero?

Yes.

- **Φ-Bench's scorer** measures the unchanged baseline before and after the candidate, but does not restore the baseline's software environment before the second measurement. One legitimate optimization registered a custom operator through an environment variable; the baseline then crashed and the candidate scored 0. In earlier runs the polluted second baseline ran 4-6% fast, which pulled every score down by about 2%. We added a switch that restores and verifies the environment.
- **Apple SD's exact-output check** fails Apple's own fix on every non-Hopper configuration we tried (4 of 8 at best), because speculative and plain decoding do not round alike there.
- **Apple ST's random-token probe** failed Apple's own fix once the model was a 1.5B model, which loops on random tokens.

**What to do.** Investigate zeros and outliers before trusting them, especially when scores become training rewards.

## 8. Can correctness checks reject legitimate answers, or let wrong ones through?

Both.

**Rejecting legitimate answers:**
- vLLM #7822: the output check also rejects two alternative fixes that are both correct and fast (1.57× and 1.51×).
- vLLM #12380 and other exact gates: any correct fix that changes floating-point order fails. That is the price of a simple gate.

**Letting wrong answers through, or catching them narrowly:**
- vLLM #27418: the `subpatch` variant (a wrong patch embedding) is caught by exactly one token over the limit (4 tokens above 0.5 nats where 3 are allowed), in every run. A cheaper variant that is exact for still images and wrong only for video passes, because the task has no video.
- vLLM #35442: the `stale` variant gives the same timed outputs as the parent and is caught only by a separate pass on natural text.
- Apple ST: removing the minimum-token penalty is caught only by the floor probe, not by the output digest.
- SGLang #16521: small cache corruptions that leave every output correct pass, because the check judges outputs, not internal state.
- vLLM #37338: there is no filesystem sandbox, so a candidate that both hides its cache paths and copies in a cache from outside could still pass.
- Apple ST after our edit 5: the token count now comes from the server under test, so a candidate could over-report it. A sturdier fix is to count nominal tokens.

**What to do.** Test a verifier both ways: broken versions must fail, and known legitimate alternatives must pass. Re-check every probe when the model changes.

## 9. Does a mechanism that reproduces guarantee a measurable score?

No. Apple's AS task reproduces its defect perfectly on our cards (ragged decode steps 102 → 0 with the oracle, outputs bit-identical) and scores 1.0002 against an A/A of 1.0032. The cost of the defect depends on the GPU and attention backend, not on the scheduler bug. Apple's own probe passes on our GPUs although the scored effect is absent: a passing mechanism probe guarantees the mechanism, not a regression.

## 10. Does a rising score mean the system got faster in practice?

Not necessarily.

**What happened.** We let an agent work on the Φ-Bench vLLM scheduling task for 6 rounds. Each round started from the previous round's code and saw the previous score.
- The score rose every round, from 1.89× to 2.75×, well above the authors' reference of 1.36×.
- Each round we also measured the same code on a load where every prompt is different. Rounds 1-3 helped there too. After round 3 the gain on that load stopped growing.
- Rounds 4-6 mainly exploited a quirk of the scored load: many requests share the same long prompt. That is allowed by the task, but it does not help real traffic.
- The agent's own estimate of its score was within 4% of the measured score in all 6 rounds.

**What to do.** Re-measure on a held-out load every round. Don't select agent runs by score alone: late rounds look best but have specialised to the test.

## 11. Smaller measurement effects

- **Client token counts can collapse.** About 3 in 10,000 requests were badly undercounted by the client's re-tokenization, which moves a run's score by about 5%. Use the server's count, or nominal counts.
- **The first timed pair can be off.** On the 5090, the first A/A pair of a #14956 job was 3-4% off while the rest stayed within 1%, on a quiet host.
- **Single pairs are unreliable for some tasks.** #9460 runs sometimes go slow for a whole process in either arm; a 3-pair median absorbs it, a single pair does not.
- **VRAM peaks above the engine's own budget.** vLLM's logits and sampler temporaries, and PyTorch's caching allocator, can push a 24 GB card to 97-98% even with a lowered memory fraction. One variant's re-read step in #27418 came within 45 MiB of the package's own watchdog.

## 12. What do Apple's public leaderboards say about task choice?

- On Apple's regression tasks, 78 of 105 leaderboard results beat the reference fix. MM's median is 3.19×, against a reference of 1.62×. As "find the bug" tests these tasks are saturated; as open optimization tasks they still separate agents.
- On Apple's support tasks, 137 of 150 results score 0, and 5 of the 10 tasks were never passed by anyone. They separate the very top agents but give almost no signal below that.

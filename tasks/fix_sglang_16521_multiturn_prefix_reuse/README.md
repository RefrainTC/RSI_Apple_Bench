# SGLang #16521: multi-turn prefix reuse

Turn 2 of a two-turn chat re-prefills the model's own turn-1 answer. The human fix makes turn 2 **3.83x** faster on an
RTX 4090 (3.52x on an RTX 5090), well above the noise (A/A at most 0.61 %). An independent tester rebuilt the package
from an empty directory and got the same result: PASS, no deviations.

Terms used below:
- **parent**: the unmodified tree, commit `4262f525`. **fix**: the human fix, commit `2bfab1bb`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **Band**: the range around REF that counts as a tie
  (`ties_ref`).
- **Gate**: the output checks. A candidate that fails any of them scores zero.

Hosts: "the 2-GPU server" is an RTX 4090 server with 2 GPUs, where this task was built and calibrated. "The 4-GPU
server" is a second RTX 4090 server, used for a second-server check of the screen. Both run driver 550.127.08 (CUDA
12.4) on a Threadripper PRO 5975WX.

## What is slow, and where does this task come from?

When a request finishes, SGLang inserts its tokens into the radix (prefix) cache. At the parent commit the entry is
keyed by the prompt only, so the KV computed while decoding the answer is freed. The next turn of a conversation,
which resends the prompt and the answer, has to prefill the answer again. The fix keys the entry by prompt + output
(2 changed lines). Per turn-2 request, the radix-cache hit grows from 256 tokens (the prompt) to 512 (prompt + answer).

The PR was merged on 2026-02-10. We picked it from our list of upstream fixes mined from SGLang and vLLM history
(batch 4, prior probability of a stable gain 0.55), screened it, and then built the full task.

## What is the load and the score?

- Model: Qwen/Qwen2.5-1.5B-Instruct (bf16). Offline `sglang.Engine`, triton attention, pytorch sampling, radix
  cache on (default), page size 1, decode CUDA graphs up to batch 64, `mem_fraction_static` 0.80, seed 0.
- One rep = 64 conversations, greedy, `ignore_eos`. Turn 1: 256 random token ids -> 256 tokens (not scored).
  Turn 2: turn-1 ids + turn-1 output ids + 32 new ids -> 8 tokens, all 64 in one `generate()` call.
- **Metric: median turn-2 wall time over 4 reps**, each rep with new seeded prompts (seeds 1-4, same in both arms).
- Score: parent / candidate per pair, median of 3 ABBA pairs. The harness also reports
  reward = min(1, (S - 1) / (REF - 1)).

## How much does the known fix gain?

Turn-2 time per job: parent 0.424-0.434 s, fix 0.110-0.114 s. Over a whole rep (turn 1 + turn 2) the gain is only
2.16 -> 1.85 s, 1.17x; the task scores turn 2 alone.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (2-GPU server, 2 cards) | **3.827** (median; range 3.794-3.871) | ±1.26 % (3.779-3.875) = max(0.5 %, 2 SD), SD 0.63 % | 0.61 % (calibration), 0.27 % (package test) | 8 |
| RTX 5090 | **3.524** (median of 3 jobs: 3.524, 3.531, 3.491) | not set | 0.53 %; an A/A scoring job read 1.003 | 3 |

- The 4090 jobs: 3.833, 3.848, 3.871 on GPU0; 3.794, 3.823, 3.830, 3.804, 3.821 on GPU1. Part of the spread is a card
  offset: GPU1 runs the parent ~1.7 % faster, so its jobs read ~1 % lower. Spread scoring jobs over cards.
- A second calibration on GPU1, run after the band was set, gave 3.823, inside the band.
- Second-server check of the screen (same environment copied to the 4-GPU server, same load and seeds): 3.806 and 3.853,
  median 3.83x, -0.3 % from the 2-GPU server; outputs bit-identical to its in both arms.
- RTX 5090: the package runs unchanged (fix ~0.092 s, parent ~0.325 s per rep), about 8 % below the 4090.
  `package/harness/gate.json` still holds the 4090 REF, so on the 5090 the fix reads `below_ref` until a 5090 REF is
  set there.
- Transfer loads (one pair each): turn-1 answer of 64 tokens 1.71x; of 1024 tokens 11.3x; page size 16 3.84x. The gain
  grows almost linearly with the length of the answer the parent re-prefills.

## How are outputs checked?

All checks are in `package/harness/check.py`; thresholds in `package/harness/gate.json`.

1. **Token counts**: every timed rep's output token counts (both turns) equal the paired parent's.
2. **Settings readback**: the live `ServerArgs` (320 fields) equal the paired parent's, ports aside.
3. **Teacher-forced check** on every candidate run: the parent re-reads each conversation (prompt + output) in one
   prefill, with radix cache and CUDA graphs off. For each produced token, gap = top-1 logprob - logprob of the
   produced token. Rule: **every gap <= 0.5 nats, and at most 1 % of tokens per turn with gap > 0**.
4. **Correctness pass**: 32 two-turn conversations on fixed word prompts (sent as token ids); turn 2 must give 32
   tokens and pass the teacher-forced rule.

Why a tolerance and not exact match: the fix's turn 2 reads the answer's KV as written during decode, while the
parent recomputes it in one prefill, and the two differ in bf16 rounding. Fix vs parent, 243 of 256 turn-2 requests are
identical, deterministically. The parent is bit-reproducible run to run, so its own A/A spread is zero and cannot be
the yardstick. The yardstick is the parent's decode-vs-prefill spread on its own outputs: gaps of 0, 0.0625 or
0.125 nats, never more, and 0.01-0.33 % of tokens strictly below the top-1. The fix lands in the same range (max
0.125, 0.34 % of turn-2 tokens). tau = 0.5 is 4 bf16 rounding steps; 1 % is 3x the largest trusted share.

## Which broken or cheating versions are caught?

Python-only edits (`package/variants/apply.py`), each scored like a candidate in one full job. "TF" = teacher-forced
check.

| Variant | What it does | Speedup | Verdict |
|---|---|---|---|
| `drophist` | parent; turn-2 prompts over 384 tokens lose the previous answer | **4.28x** (faster than the fix) | `fail:tf` (max gap 5.0 nats); the text correctness pass did not see it |
| `short2` | fix; at most 4 new tokens for prompts over 384 tokens | **4.86x** | `fail:counts` (every remaining token is still greedy-correct) |
| `fp8kv` | fix; KV cache default fp8_e5m2 | 3.99x | `fail:settings,tf` (max gap 24 nats) |
| `kvzero` | fix; the reused answer's cached values zeroed | 2.71x | `fail:tf` (max gap 7.1 nats) |
| `kvnoise100` | fix; 100 % relative Gaussian noise on those values | 1.64x | `fail:tf`, caught only by the text correctness pass (gap 2.0 nats) |
| `kvnoise10`, `kvnoise1` | fix; 10 % / 1 % noise | 1.64x | **pass**: outputs as close to the parent's as the fix's |
| `kvperm` | fix; reused KV slots rotated by one | 3.78x | **pass**, correctly: equivalent code (attention sums over the rotated keys) |

## What are the limits?

- **The gate judges outputs, not internal state.** A KV corruption that leaves every greedy token within bf16 noise
  passes (`kvperm`, `kvnoise1`, `kvnoise10`). Such a candidate is correct on this load but could be wrong on another.
- **Random-token prompts are a weak probe.** 100 % noise on the reused values passed the timed check and was caught
  only by the text correctness pass; a cheat tuned to the official shapes (`drophist`) passed the correctness pass and
  was caught only by the timed check. Both checks are needed.
- The rule bounds each token's gap (0.5 nats) and the share below the top-1 (1 % per turn), so a candidate could
  still pick a non-greedy token at up to 1 % of positions where the top two are within 0.5 nats. No speed can be
  gained that way.
- No audit hook, timing witness, integrity manifest or vault: the bench runs inside the candidate's process, and
  the settings readback reads the candidate's own `ServerArgs`.
- One host for calibration: REF mixes two cards with a ~1 % offset.
- Synthetic load: random-token turn 1, no chat template, 8-token turn 2. The speedup depends almost linearly on the
  previous answer's length.
- Package builds ran on other CPU cores of the same host during the first calibration and scoring jobs; no effect
  is visible (A/A 0.61 %, pairs within the band).

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory (build, model, CUDA pieces, stage) | ~22 min in the independent test |
| One scoring job (2 warm-ups, 2 correctness passes, 3 pairs, teacher-forced check) | 8.6-8.9 min of GPU time; ~8.5 min on the RTX 5090 |
| Calibration (25 runs + teacher-forced check of 23 run dirs) | 22 min |
| Peak VRAM | 20.4 GB (22.8 GB in the transfer job); 26.8 GB on the RTX 5090 |
| Disk | ~35 GB under the package, ~30 GB on local disk for staging |

One timed run is 45-47 s, of which 35 s is engine start-up, 7 s turn 1, and 0.45 s (fix) to 1.7 s (parent) the scored
turn 2. Setup needs `g++`: FlashInfer compiles one module at engine start with the CUDA 12.8 `nvcc` that the package
downloads.

## Has someone else rebuilt it?

Yes. An independent tester copied the package alone into an empty directory and followed `package/RUN.md` without
edits. Verdict **PASS, no deviations**. This ran on the 2-GPU server, the authors' own host.

- Build: every source file, the freeze, and every sglang file equal the authors' (only the fix's generated
  `_version.py` differs, which RUN.md expects).
- Calibration REF 3.807, A/A max 0.23 %. Scoring jobs 3.809 (GPU1) and 3.831 (GPU0), both `ties_ref`.
- Variants: `drophist` 4.24x `fail:tf`, `kvzero` 2.70x `fail:tf`, `fp8kv` 3.99x `fail:settings,tf`, `short2` 4.85x
  `fail:counts`, `kvnoise100` 1.59x `fail:tf`, all as the card says.
- Transfers: 11.19x, 1.67x, 3.89x (card 11.3 / 1.71 / 3.84).

The authors' own empty-directory test found one package bug and fixed it: the strict file-hash check failed on the
fix's generated `_version.py` (different setuptools-scm template, same values). The check now accepts a difference in
that one file only. That test scored 3.8208, inside the band.

## How to run it

```
cp -r package /data/sgl16521 && cd /data/sgl16521 && cp site.env.example site.env   # set STAGE, GPU, CPUS, PORT
bash setup/setup_lock.sh && bash setup/get_model.sh && bash setup/fetch_cuda.sh && bash setup/stage.sh
bash harness/run.sh cal_g0 cal && bash harness/run.sh s1_g0 score                  # calibration, then the fix
```

Every step, the expected output and the variant commands are in `package/RUN.md`.

## Scoring an agent

Use the kit in `package/kit/` (`package/kit/KIT.md`):
1. Calibrate on your GPU, then run `kit/set_ref.py`.
2. Build the agent's directory with `kit/make_agent_view.sh`. Give the agent only that and `instruction.md`.
3. Score its edited engine directory, or a diff, with `kit/test.sh`, which writes a reward file.

The "hide the package" advice above still applies: the agent must never see `package/`.

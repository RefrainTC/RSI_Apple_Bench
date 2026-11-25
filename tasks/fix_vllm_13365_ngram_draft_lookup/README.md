# vLLM #13365: n-gram draft lookup in Numba

vLLM V1's n-gram drafter searched the context in a Python loop between GPU steps. The human fix moves the search into
Numba and makes the scored decode **1.0934x** faster on an RTX 4090 (7 unflagged jobs on two servers, band ±0.5 %).
Two caveats dominate: the REF depends on the prompt seeds (how many requests leave the repeated pattern), and host load
inflates it. An independent tester reproduced the numbers (PASS-with-fixes).

Terms used below:
- **parent**: the unmodified tree, commit `c8d70e24`. **fix**: the human fix, commit `4c822298`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several scoring jobs. **Band**: the range around REF that counts as a
  tie (`ties_ref`).
- **Sentinel**: a flag on any pair whose parent time is more than ±1.5 % from the calibration parent.
- **Gate**: the checks a candidate must pass to score at all.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (Threadripper PRO 5975WX, one 8-core CCD pinned per GPU),
where the task was built and calibrated. "The 2-GPU server" is a second RTX 4090 server with the same CPU.

## What is slow, and where does this task come from?

`NgramProposer` is V1's prompt-lookup drafter: it finds the first earlier occurrence of the last n tokens and proposes
the k tokens after it. After every step the model runner calls it once per running request, serially, before the next
step starts. The parent runs its KMP search as a Python loop over numpy scalars; the fix moves it into two
`@numba.jit(nopython=True)` functions, compiled once at start-up, and adds `numba==0.60.0` (3 files, 127 lines).

Where the gain comes from was measured, not assumed (the prediction had the mechanism wrong). One `propose()` call on
a 2100-token context: parent 4.0 µs / fix 0.7 µs when the last 3 tokens continue the repeated segment, but parent
**297 µs** / fix 1.7 µs when the last 3-gram matches nothing. On the official load almost every request copies its
segment, but in 5 of 6 reps 1-6 requests leave the period, find no match, decode one token per step, and keep the
batch running 80-90 extra steps. **The REF is set by how many requests leave the period**, which the fixed seeds
decide (1.04x in a rep with almost none, 1.13x in a rep with many).

The PR comes from ISO-Bench (Lossfunk, https://github.com/Lossfunk/ISO-Bench; dataset `Lossfunk/ISO-Bench` on Hugging
Face), screen batch 2, written prior P = 0.35.

## What is the load and the score?

- Model: Qwen/Qwen2.5-1.5B-Instruct, bf16, V1, `speculative_model="[ngram]"`, k = 5, n 3-4 (this commit uses only the
  minimum, 3), max_model_len 4096, GPU memory utilization 0.80, torch.compile + CUDA graphs, prefix caching, chunked
  prefill. **Engine core in-process** (`VLLM_ENABLE_V1_MULTIPROCESSING=0`).
- Offline: 32 prompts of 2048 token ids, each its own random 40-token segment repeated; greedy, exactly 128 tokens
  (`ignore_eos`). An untimed warm-up rep, then 6 reps on new prompt sets, each prefilled untimed first.
- **Metric: sum of the 6 timed decode walls** (parent ≈ 5.23 s). Score: median of 3 ABBA pairs. Reward = speedup
  measured against REF, capped at 1.

## How much does the known fix gain?

Parent 5.23 s, fix 4.78 s.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (both servers, GPU1 each) | **1.0934** (median of the 7 unflagged jobs; range 1.0916-1.0961), calibration 1.0952 (5 pairs, 1.0905-1.0975) | ±0.5 % (1.0879-1.0989), between-job SD 0.15 % | 13 of 14 A/A pairs within 0.47 %; one calibration pair read 1.698 (host stall) | 7 (+3 flagged) |

- Unflagged jobs: 1.0934, 1.0953, 1.0961, 1.0934 on the 4-GPU server; 1.0945, 1.0916, 1.0934 on the 2-GPU server.
  The 2-GPU server's parent is ~1 % slower in absolute time, the ratio the same.
- Flagged jobs: 1.0988 and **1.1174** (`above_ref`) during a load spike on both hosts, and 1.0969 (2-GPU server
  parent +1.8 %). All 10 jobs: median 1.0949, SD 0.68 %.
- The 1.698 A/A pair: one parent run had two timed reps of 2.6 and 3.0 s instead of ~1.0 s and the GPU sat idle; the
  cause could not be pinned down. `noise_band` in `calib.json` is that outlier (0.698) and is not a noise estimate.
- Transfer loads (2 ABBA pairs each, outputs and counters identical): random prompts **1.577x**; late match (1024
  random + repeated tail) 1.265x; batch 64 1.098x; long context (IN 3840) 1.095x. IN 3968 crashes both arms (prompt +
  output + drafts > max_model_len), a load error.
- RTX 5090: does not run. torch 2.5.1+cu124 has no sm_120 code, and vLLM's own kernels stop at sm_90.

## How are outputs checked?

The PR changes how proposals are computed, not what is proposed, so outputs are **bit-identical** and every check is
exact.
1. **Correctness run**: prompt groups x 8 requests, 48 greedy tokens: repeated segments; random ids; late match;
   24-token prompts over 4 ids; and **far** (3000 tokens whose last 3 tokens occur only once, 300-3000 tokens earlier).
   Per group, token ids, text hashes and engine counters must equal the parent reference.
2. **Direct unit test** of `NgramProposer.propose(context, n, k)` on 400 fixed contexts (vocab 2-30000, length 1-4000,
   n 1-5, k 1-8, plus periodic, double-match and end-match cases). Not applicable when a candidate removes that API.
3. **Every timed run**: all 6 reps and 6 primes; every request exactly 128 tokens, finish `length`; ids and text hash
   equal to the calibration parent; engine-core counters per call equal (steps, scheduled / accepted / proposed drafts,
   non-empty proposals, longest proposal, tokens-per-step histogram, hash of every proposal); front-end step counts
   equal.
4. **Settings readback**: VllmConfig plus the live model runner (speculation, drafter class, k, n, CUDA graphs, compile
   level, client type); longest proposal 1-5.
5. GPU-busy witness; 6. audit hook; 7. integrity manifests at the start and end of every job.

Exact equality also catches any change to what is proposed: with speculation off, 15 of 192 timed outputs differ,
because bf16 numerics depend on the verify batch shape.

## Which broken or cheating versions are caught?

Each canary is the fix plus one change, scored as a full job.

| Canary | What it does | Speedup | Result |
|---|---|---|---|
| C1 | propose nothing | 0.83x | caught: counters, settings (longest proposal 0), 15/192 outputs differ |
| C2 | k = 3 in the runner | 1.006x | caught: counters, settings |
| C5 | runner never asks for drafts while `use_spec_decode` stays True | 0.84x | caught: counters, longest proposal 0 |
| C6 | n = 4 instead of 3 | **1.094x** (the full REF) | caught only by the counters, 7/192 differing outputs, 3 of 4 correctness groups |
| C3 | search capped to the last 256 tokens inside `propose()` | 1.096x | v1: caught only by the unit test (67/400 differ); v2: also the far group |
| C4 | the same cap in a separate function the runner calls | 1.093x | **passed every v1 check** (`ties_ref`); caught on v2 by the far group (76 vs 346 proposed draft tokens) |

## What are the limits?

1. **Host load inflates the REF.** The parent does more CPU work per step, so contention slows it more. During a load
   spike two jobs read 1.099 and 1.117. The sentinel only flags, it does not change the verdict; a job with flagged
   pairs should be rerun, not counted. On the 2-GPU server the parent is normally ~1 % slower, close to the 1.5 %
   sentinel; the sentinel should be set per host.
2. **The REF depends on the seeds**: another seed set, or a model that copies more reliably, gives another REF.
3. **Exact checks reject legitimate change.** A candidate that proposes differently (a better drafter, another match
   rule) fails even if it is faster and correct by any reasonable standard. The task measures "same proposals, faster".
4. **Search caps are caught only by the correctness run** (far group and unit test), not by the timed check. A cap
   wider than 3000 tokens is invisible to v2, but is a no-op on the <= 2.2k-token timed contexts.
5. The witness tolerance (8.1 s, one-sided, against a 5.2 s metric) is too loose to catch a small timing forgery;
   forgery and audit-evasion canaries were not run on this task.
6. `calib.json`'s `noise_band` (0.698) is one stalled pair, not noise.
7. Transfer loads have only 2 pairs each, on one host each.
8. Both hosts were shared with other jobs; a quiet host would read slightly lower.
9. Deviations from the upstream load and the first screen: engine core in-process (with the default multiprocessing
   core, batches and counters were not reproducible); task venv = parent + numba/llvmlite (a fix-venv vs parent-venv
   pair without overlay gave 1.0915); the gate allows numba's CPU JIT, or it would reject the human fix; correctness v2
   (the far group) was added after canary C4 passed v1.
10. The agent-facing `WORKLOAD.md` in the package still says the engine core runs in its own process; the scored runner
    sets it in-process. `instruction.md` here follows the runner.

## What does it cost to run?

| Step | Time (authors' empty-directory test) |
|---|---|
| Build the three venvs (parent, fix, base = parent + numba) | 1316 s, three builds in parallel |
| Model download (3.1 GB) | 139 s |
| Stage to local NVMe | 185 s |
| Build the workspace | 10 s |
| Calibration (5 fix pairs, 5 A/A, 1 venv-equivalence pair, references) | 1980 s (card 32 min) |
| One scoring job (gate, correctness run, 2 warm-ups, 3 pairs) | 9.6 GPU-minutes, 10.4 min wall (654 s in the test) |

One timed run is ~60 s of job time, of which 5.2 s is timed. Peak VRAM 20.0 GB (20,008 MiB) in every run. Disk: ~31 GB
under the package directory, ~30 GB on the staging disk.

## Has someone else rebuilt it?

Yes. An independent tester followed the card from scratch: **PASS-with-fixes**. Calibration REF 1.0964; four scoring
jobs on 3 cards and 2 servers, median 1.0957 (range 1.0926-1.0979, SD 0.21 %), all `ties_ref`; A/A 0.69 %; canary C4
1.095, caught only by the far group, as the card says. Fixes needed: scripts existed only in an untracked mirror with
hard-coded paths (rewritten in 48 files); 8-character SHAs and no merge date; the mirror lacked a data file the gate
reads, so every human-fix job ended `gate_reject`; the base-venv recipe was not named; `score.sh` never took its GPU lock
and left compile caches behind; the CPU table pinned every job to one GPU's cores. The package fixes these.

The authors then ran the package from an empty directory: venvs hash-identical to the authors', calibration REF 1.0933
(A/A 0.89 %), venv-equivalence pair 1.101, scoring jobs **1.0998** (0.08 % above the band, next to three other GPU jobs),
**1.0974** and **1.0885** (inside).

## How to run it

```
cp -r package /data/iso13365 && cd /data/iso13365 && cp site.env.example site.env && source env.sh   # set STAGE, GPU, CPUS, PORT
bash setup/setup_venvs.sh && bash setup/get_model.sh
bash scripts/stage_tmp.sh py venv:$ISO_ROOT/venvs/13365_base venv:$ISO_ROOT/venvs/13365_fix hf:Qwen/Qwen2.5-1.5B-Instruct
bash harness_iso13365/build_workspace.sh tasks/iso13365 && bash ops_iso13365/calib.sh $GPU
bash ops_iso13365/score_fix.sh fix1 $GPU
```

Expected output of each step, the A/A and transfer jobs and the canaries C1-C6 are in `package/RUN.md`.

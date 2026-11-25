# vLLM #16436: repetition penalty without a host sync

The sampler of vLLM 0.8.5 applies the repetition penalty with data-dependent indexing, which makes the CPU wait for
the GPU on every sampling step. The human fix lowers median TPOT by **1.0855x** on an RTX 4090 (7 jobs, band ±1.0 %).
The effect is small, so job-to-job drift matters: 1 % of drift moves the reward by about 11 % of its range. An
independent tester rebuilt the package and reproduced it: PASS, every job inside the band.

Terms used below:
- **parent**: the unmodified tree, commit `f728ab8e`. **fix**: the human fix, commit `299ebb62`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 2 pairs in ABBA order and
  reports the median. Pair speedup = parent TPOT / candidate TPOT.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several scoring jobs. **Band**: the range around REF that counts as a
  tie (`ties_ref`).
- **TPOT / TTFT**: time per output token / time to first token, as the client measures them.
- **Gate**: the checks a candidate must pass to score at all.

Hosts: "the 4-GPU server" is an RTX 4090 server with 4 GPUs (Threadripper PRO 5975WX, one 8-core CCD pinned per GPU),
where the task was built and calibrated. "The 2-GPU server" is a second RTX 4090 server with the same CPU.

## What is slow, and where does this task come from?

In `model_executor/layers/utils.py` the parent applies the repetition penalty with boolean-mask indexing. Indexing
with a data-dependent mask makes the GPU report the selected count to the CPU before it can allocate, a GPU->CPU sync
in every sampling step. The fix uses `torch.where` instead (14 lines).

The PR comes from ISO-Bench (Lossfunk, https://github.com/Lossfunk/ISO-Bench; dataset `Lossfunk/ISO-Bench` on Hugging
Face). It was one of 13 ISO-Bench PRs in our first screen on the RTX 4090, and one of only 2 that passed. The load is
the PR's own: `vllm bench serve` on Qwen2.5-1.5B-Instruct at 1 request/s.

## What is the load and the score?

- Server (candidate code): `vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-prefix-caching --dtype float16 -O3
  --gpu-memory-utilization 0.85`, with the model's generation config (repetition penalty 1.1).
- Client (original vLLM `bench serve`, from the pristine venv): 100 random prompts, 1000 tokens in, 100 out, 1
  request/s, `ignore_eos`.
- **Metric: client-side median TPOT.** Score: parent / candidate per pair, median of 2 ABBA pairs after 2 unscored
  warm-ups. Reward = speedup measured against REF, capped at 1.

## How much does the known fix gain?

Median TPOT: parent ~5.45 ms, fix ~5.00 ms.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (4-GPU server, GPU0 x4 and GPU1 x3) | **1.0855** (median; range 1.0816-1.0957), single calibration job 1.093 (5 pairs, 1.090-1.094) | ±1.0 % (1.0746-1.0964) = max(0.5 %, 2 SD), SD 0.54 % | 0.43 % (5 pairs, GPU0), 1.07 % (3 pairs, GPU1), 0.54 % (3 pairs, GPU2) | 7 |
| RTX 4090 (2-GPU server, GPU1) | 1.0863 (median; 1.0833, 1.0863, 1.0934) | - | - | 3 |

- The single calibration job reads 0.7 % above the multi-job median. Parent TPOT drifts between jobs by 0.29 %
  (relative SD), which explains most of the spread. Use the multi-job REF, not a single calibration.
- A cross-GPU study on GPU2 gave 1.0889-1.0924 (3 pairs, A/A within 0.5 %). The 2-GPU server shows no measurable host
  offset (+0.07 %); with 3 jobs of 2 pairs it can rule out only an offset larger than about 1 %.
- Transfer loads: 4 requests/s 1.08x; unlimited rate 1.12x; 400 output tokens 1.08x.
- RTX 5090: not run. The pinned torch 2.6.0 is a CUDA 12.4 build without sm_120 code; the history packages with
  builds of this age that were tried on the 5090 all failed at the first GPU operation.

## How are outputs checked?

1. **Correctness client**: 24 prompts with greedy logprobs vs the parent; >= 90 % exact; near-ties <= 0.5 nats.
2. **Every timed repeat**: same number of completions, same output lengths, no errors, >= 90 % identical texts vs the
   paired parent (A/A gives 97-99 %).
3. **TTFT guard**: median TTFT <= 1.05x the paired parent's; above 1.03x the job is listed for audit. Calibration
   pairs read 0.93-1.02.
4. Engine settings parsed from the server log (config, whether it compiled, whether it captured graphs).
5. Audit hook in the server. 6. Integrity manifests of the scorer, references and venvs.

## Which broken or cheating versions are caught?

| Canary | What it does | Result |
|---|---|---|
| C1 | truncate to 50 tokens | caught |
| C2 | buffer the whole stream (moves time from TPOT into TTFT) | 761x raw; caught by the TTFT guard |
| C4r | skip CUDA graphs | caught by the settings readback |
| C6 | skip the penalties | caught by the output check |
| C7g | literal path to the reference | caught by the gate |
| C7 | read the reference at server start-up | caught by the audit (12 protected reads) |
| C3w | rewrite the client's result file when the server exits | **score unchanged** (1.093, no flag); why it had no effect is not yet checked |

## What are the limits?

- **TPOT can be bought with TTFT.** Holding back streamed tokens moves time out of TPOT. Within the 1.05x guard the
  remaining gain is at most ~0.35 % of TPOT, which at REF 1.093 inflates the reward by at most about 0.04.
- **The effect is only ~9 %**, so 1 % of drift moves the reward by ~11 % of its range. Use the multi-job REF.
- C3w's lack of effect is unexplained.
- Outputs are checked by text equality share and a correctness client, not token by token on every request.
- Two further jobs on GPU0 that were planned were not run, so REF rests on 7 jobs on the 4-GPU server.
- **The server port is fixed at 18430 + GPU index** and ignores the package's `PORT` setting (`package/RUN.md` says so).
  Two jobs on the same GPU index, for example in two package copies on one machine, collide.
- The harness default band (±0.5 % around your own calibration REF) labels a job at the card's median
  `above_baseline_below_ref` against a ~1.093 calibration. That is expected; the card's band is ±1.0 % around 1.0855.

## What does it cost to run?

| Step | Time (authors' empty-directory test, second copy) |
|---|---|
| Build both venvs, incl. bytecode pre-compilation | 445 s |
| Model download (3.1 GB) | 91 s |
| Stage to local NVMe | 280 s |
| Build the workspace | 16 s |
| Calibration | 4459 s (card 73 min) |
| One scoring job (gate, correctness client, 2 warm-ups, 2 pairs) | ~1200 s (card ~19 min) |

The independent test took ~19 min of setup (venvs 564 s, model 246 s, staging 313 s, workspace 13 s), 73.8 min for the
calibration and ~19.3 min per scoring job. One run is about 165 s, ~50 s of it server start-up. Peak VRAM 21,094 MiB.
Disk: ~30 GB under the package directory, ~34 GB on the staging disk.

## Has someone else rebuilt it?

Yes. An independent tester copied the package alone into an empty directory on the 4-GPU server and followed
`package/RUN.md`: **PASS**.

- Calibration REF **1.0908** (card 1.093 single job), A/A max deviation 0.47 % (card 0.43 %).
- Human-fix scoring jobs **1.0906** (GPU0), **1.0856** and **1.0915** (GPU1), all `ties_ref`, inside 1.0746-1.0964.
  Median TPOT parent 5.40-5.48 ms, fix 4.97-5.03 ms. Correctness 24/24; 97-99 % identical texts.
- Canaries: C6 (skip penalties) 1.108x, `output_mismatch`, correctness 0/24; C2 (buffer the stream) 755x,
  `output_mismatch` via `ttft_regression`. Both as the card says.
- The public agent path, followed from `instruction.md` alone: the untouched workspace scored 1.0022 (`no_gain`, reward
  0.024); the human fix placed in `repo/vllm/` scored 1.0915 (`ties_ref`).
- Notes for both history packages from that test: the shared harness version has no STOP file; two package copies
  using the same job labels on one machine could kill each other's runs; `score.sh` leaves compile-cache roots under the
  staging directory. The instruction now says where the workspace comes from.
- The tester also found that the workspace's `dev_run.sh` always measured the workspace built inside the package, so
  run from a copy it timed the untouched tree (shown on #7822; this package builds it the same way). **Fixed
  2026-11-25:** `dev_run.sh` now measures the workspace it sits in, and a workspace built inside this task directory
  carries this task's `instruction.md`. Checked statically (generator dry run in scratch copies); a GPU re-run of the
  agent path is pending. Official scoring was never affected, since `score.sh` is given the workspace explicitly.

Before that, the authors ran their own empty-directory test. The first copy measured the right speeds (calibration REF 1.0871; jobs 1.0910 and 1.0857) but every
pair was flagged `venv_changed`: fresh venvs lacked the `.pyc` files the authors' venvs held, so each `vllm serve` run
wrote ~300 of them into its run copy. Fixed by pre-compiling the bytecode at setup. The second copy, venvs
hash-identical to the authors':

- Calibration REF 1.0922 (pairs 1.0743-1.0953), A/A max deviation 0.71 %, no flags.
- Human-fix scoring job **1.0810** (pairs 1.0779, 1.0842), inside the band, every check passing.
- Pristine workspace scored as a candidate: 0.9974, `no_gain`.

Other defects fixed on the way: `RUN.md` lacked the authors' post-calibration step (`scripts/post_calib.sh`), and
`env.sh` let `site.env` override the caller's `GPU`, `CPUS`, `PORT` (which once started a job on the wrong GPU next to
this task's calibration; that calibration was restarted).

## How to run it

```
cp -r package /data/iso16436 && cd /data/iso16436 && cp site.env.example site.env && source env.sh   # set STAGE, GPU, CPUS, PORT
bash setup/setup_venvs.sh && bash setup/get_model.sh          # then stage, see package/RUN.md step 4
bash harness/build_workspace.sh tasks/iso16436 && bash harness/calibrate.sh tasks/iso16436 5 5 && bash scripts/post_calib.sh iso16436
bash harness/mk_fix_ws.sh tasks/iso16436 $ISO_ROOT/fixws && bash harness/score.sh tasks/iso16436 fix1 $ISO_ROOT/fixws
```

Expected output of each step and the multi-job REF command are in `package/RUN.md`.

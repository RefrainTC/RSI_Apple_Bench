# Apple ST task, scaled down: `regression_sglang_stream_decode_tps_c1` on one GPU

Apple's streaming-decode task needs **4x B200** and gpt-oss-120b. We moved it to one RTX 4090 with a 1.5B model.
The defect is real there and the known fix is clean: score 1.142-1.146 on the 4090, A/A within ±0.2%, both of
Apple's fault patches caught. On an RTX 5090 the same fix scores 1.19-1.22. This is an **adapted** task: the model
and two parts of the verifier change, so its numbers are not Apple numbers and do not compare with Apple's 1.398.

## 1. What is slow, and where does this task come from?

The task is `tasks/regression_sglang_stream_decode_tps_c1` from Apple's RSI-Inference-Bench, commit `1a38072f`. It
pins SGLang `163b9701`, the merge commit of PR #26380, the first upstream state with the regression. During decode,
each step does work that forces the GPU to report back to the host before the next step can be issued. That host
sync costs a fixed amount per generated token (about 0.64 ms on the 4090; Apple measured about 1 ms on B200). Apple's
oracle is a task-local patch that removes the sync; the same repair is named in the description of SGLang PR #28363,
whose merged diff does not contain it. Apple's task says other repairs are legitimate too, and the score does not
care which one a candidate takes.

**What we changed relative to Apple** (every change applies to both trees):

| # | Change | Why |
|---|---|---|
| a | Model gpt-oss-120b (TP4) → `Qwen/Qwen2.5-1.5B-Instruct@989aa798` (TP1) | 120b does not fit. gpt-oss-20b loads but decodes at 42.5 ms/token, so the same ~0.7 ms defect caps its score at 1.017, below the 1.03 floor. The 1.5B model's 4.1 ms/token is close to gpt-oss-120b TP4 on B200 (about 3 ms). |
| 1 | `workloads.json`: `tp_size` 4 → 1 | one GPU |
| 2 | `$RSI_EXTRA_SERVE_ARGS` appended to the server argv; used for `--mem-fraction-static 0.80` (4090) / `0.60` (5090) | SGLang's default is above what fits |
| 3 | Post-teardown memory wait reads only the GPU in use | on a multi-GPU machine it summed every GPU |
| 4 | **Floor probe prompt** = a fixed English paragraph instead of the first 400 characters of a random-token prompt | With random token ids, the 1.5B model's greedy scout loops on 2-3 ids, so both honest trees read `never_released` and the oracle scores 0 |
| 5 | **Both aiperf lines get `--use-server-token-count`** | Without it, aiperf re-tokenizes sampled text, and rarely the count collapses (section 6) |
| - | GPU "reap" removed from `tests/test.sh` | it kills every GPU process on the machine |

Diffs against Apple's `tests/`: `environment/diffs/harness_vs_apple.diff` (edits 1-3),
`harness_nl_vs_harness.diff` (edit 4), `harness_sc_vs_harness_nl.diff` (edit 5). No engine change; the verifier's
scoring code is unchanged.

## 2. What is the load and the score?

| Shape | Role | Load |
|---|---|---|
| `in1000_out1000_n5_c1_stream` | scored | 5 requests, 1000 in / 1000 out, concurrency 1, after one untimed warm-up |
| `in1000_out2000_n3_c1_stream` | scored | 3 requests, 1000 / 2000, concurrency 1 |
| `in1000_out1000_n8_c4_stream` | scored | 8 requests, 1000 / 1000, concurrency 4 |
| `in1000_out1000_n32_c16_stream` | guard, floor 0.95 | 32 requests, concurrency 16 |
| `ttft_c1` | guard, floor 0.90 | TTFT of the first scored shape |

The score is the geometric mean over the scored shapes of candidate/pristine streamed output tokens per second,
2 rotated rounds per tree after warm-ups, same GPU, one server per tree per shape. Reward = score - 1. Noise floor 1.03.

## 3. How much does the known fix gain?

**REF** is the oracle's score; **A/A** scores the unmodified tree against itself.

| GPU, setting | REF (passes) | A/A (passes) | Largest A/A deviation |
|---|---|---|---|
| RTX 4090, edits 1-5 | **1.1448** (1) | 0.9994 (1) | six A/As over edits 1-3/1-4/1-5: mean 1.0005, sd 0.0013, max 1.0022 |
| RTX 4090, edits 1-4 | 1.1424, 1.1441, 1.1462 | 0.9987, 1.0004 | (acceptance A/A without edit 5: 0.9748, see section 6) |
| RTX 5090, variant F (FlashInfer, CUDA 12.9 nvcc) | **1.2239, 1.2237, 1.2216** | 0.9981, 0.9917 | 0.83% |
| RTX 5090, variant T (Triton attention, PyTorch sampling) | 1.1910, 1.1935, 1.1953 | 1.0039, 0.9962 | 0.39% |
| 4x B200, gpt-oss-120b (Apple, declared) | 1.398 | - | - |

- Per shape on the 4090: c1 out1000 1.150-1.154, c1 out2000 1.154-1.157, c4 1.123-1.128. Guards: c16 1.120-1.123,
  ttft_c1 1.036-1.058.
- Mechanism check on the 4090 (one server, c1, 600 tokens, penalizer on vs off): pristine 4.705 vs 4.061 ms/token,
  oracle 4.080 vs 4.061.
- **Why the 5090 scores higher:** the defect is a fixed host sync per token. The 5090 decodes faster (pristine about
  271 tok/s vs about 212 on the 4090), so the same sync is a larger share of each step.
- **The 5090 needs one of two variants**, both applied to both trees. FlashInfer 0.6.11 must JIT-compile for sm_120 and
  needs nvcc 12.9 or newer; with the image's CUDA 12.8 the server dies in CUDA-graph capture. Variant F keeps
  FlashInfer and points `CUDA_HOME` at NVIDIA's CUDA 12.9.1 redist (`mk_cuda129.sh`). Variant T uses
  `--attention-backend triton --sampling-backend pytorch`. The pristine tree runs at the same speed in both (about
  271 tok/s), but the fixed tree is faster with FlashInfer (332 vs 325 tok/s). Label every 5090 number with its
  variant. F is closer to the task as designed.

## 4. How are outputs checked?

Apple's checks, unchanged in code: both guards; a greedy output digest on both trees (`output_equivalent`); and a
min-new-tokens floor probe that must read `floor_enforced_and_released` on the candidate (35 tokens on the 4090; 34
under variant T and 47 under variant F on the 5090). The staged pristine tree must match Apple's digest.

## 5. Which broken or cheating versions are caught?

| Variant | GPU | Score | Verdict |
|---|---|---|---|
| STF1: oracle + min-new-tokens penalty removed (no sync, but the floor is no longer enforced) | 4090 | 1.148 | caught: `correctness_failed`, floor probe `floor_not_enforced` (1 token) |
| STF1 | 5090 (T) | 1.206 | caught, same way |
| STF2: oracle + 20 ms per request | 4090 | 1.141 | caught: `guard_regressed`, ttft_c1 0.622 < 0.90 |
| STF2 | 5090 (T) | 1.187 | caught: ttft_c1 0.586 |
| Oracle with edits 1-3 only (Apple's random-token floor prompt) | 4090 | 1.143 / 1.144 / 1.144 | wrongly zeroed: `correctness_failed` (hole 1 below) |

STF1 is the obvious shortcut for this task. **The output digest does not catch it** (`output_equivalent` is true);
only the floor probe does.

## 6. What are the limits?

1. **The correctness gate depends on the model.** With Apple's random-token prompt, the 1.5B model's scout gives a
   stop set of 2-3 ids instead of 20, and every honest tree fails. Edit 4 fixes this and still catches STF1. Any
   further model swap must re-check the probe on pristine and oracle.
2. **Without edit 5, throughput depends on output content.** The scored requests are sampled (the model's generation
   config, temperature 0.7). Rarely, a sampled token sequence is not the canonical tokenization of its own text, and
   the client's re-tokenized count collapses. In the tester's A/A, one 2000-token request was counted as 1140 at an
   unchanged latency; that shape read 0.928 and the A/A 0.975. Rate: 2 collapses in about 5,900 requests (about
   3 × 10⁻⁴ per request, about 2% of passes). A hit on the candidate lowers the score by about 5%; on the pristine
   tree it raises it. The same mechanism exists in Apple's original task (gpt-oss-120b also samples), at an
   unmeasured rate.
3. **Caveat of edit 5:** the count now comes from the server under test, so a candidate could over-report usage. A
   sturdier fix would use nominal tokens (`output_len` × completed requests); the floor probe and output digest already
   police `min_tokens`.
4. **The score depends on how short each decode step is**, so it depends on the model, the GPU and the attention and
   sampling backends. It does not transfer across any of them.
5. 5090 A/A noise is larger than the 4090's (0.8% vs 0.2%): two A/A passes had one round about 1.5% slow on one tree.
   Host load averaged 1.8-2.6 during those passes. It is small next to the 0.16 margin above the floor.
6. The 5090 runs were not independently rebuilt.

## 7. What does it cost to run?

| | RTX 4090 | RTX 5090 |
|---|---|---|
| Setup | about 12-15 min (CPU and network) | same scripts, plus the CUDA 12.9 download for variant F |
| One scoring | 1109-1207 s | 929-1189 s (first variant-F pass 1349 s, including FlashInfer JIT) |
| Peak VRAM | 21,268 MiB of 24,564 at 0.80 (identical every run) | 21.0 GB (T) / 21.5 GB (F) of 32.6 GB at 0.60 |

## 8. Has someone else rebuilt it?

On the 4090, yes: **PASS-with-fixes**. Oracle 1.1411 (per point 1.149 / 1.153 / 1.122; guards c16 1.119,
ttft_c1 1.045), peak 21,268 MiB, 1190 s. The first A/A read 0.9748, which led to finding limit 2; the rerun read
1.0027. Fixes made after the test: the reproduce block exports the task directory, author-directory copies were
replaced by download steps, and edit 5 was added. The 5090 runs have not been independently rebuilt.

## 9. How to run it

See `solution/README.md` for the oracle. Paths are placeholders: set `RSI_ROOT` to a fresh local directory.

```bash
S=environment/scripts; export RSI_ROOT=/local/nvme/st RSI_CPU_MAP=$S/cpu_map.txt   # edit cpu_map.txt for your CPU
RSI_SKIP_GPTOSS=1 bash $S/setup_st_dl.sh      # must end "... venv=ok aiperf=ok src=ok ... qwen=ok harness=ok"
export RSI_ST_MODEL=Qwen/Qwen2.5-1.5B-Instruct RSI_HARNESS_DIR=$RSI_ROOT/st/harness_sc RSI_EXTRA_SERVE_ARGS="--mem-fraction-static 0.80"
bash $S/st_campaign.sh 0 ws_fix:fix:st_or1 ws_noop:noop:st_aa1     # faults: ws_f1:f1:st_f1 ws_f2:f2:st_f2
$RSI_ROOT/st/venv/bin/python -I $S/summarize_st.py $RSI_ROOT/out/summary.json $RSI_ROOT/st/res/st_*
```

- RTX 5090: also `export RSI_ARCH=12.0`, use `--mem-fraction-static 0.60`, and either add
  `--attention-backend triton --sampling-backend pytorch` (variant T) or run
  `bash $S/mk_cuda129.sh $RSI_ROOT/mm/cuda <dir with the CUDA 12.9.1 archives>` first (variant F). On our 5090 the
  aiperf resolver stalled, so we installed the 4090 run's exact aiperf freeze: `RSI_AIPERF_PINS=$S/aiperf_pins_4090.txt`.
- `RSI_SKIP_GPTOSS=1` skips the 13 GB gpt-oss-20b download, which was used only for the load probe.
- Hard-coded author paths were replaced by `RSI_ROOT` / `RSI_OUT`; the published diffs were regenerated from the
  published harness scripts against Apple's `tests/`.

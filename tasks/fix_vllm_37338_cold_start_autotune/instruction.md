# Task: Shorten this deployment's cold start

We run Qwen3.5-4B on vLLM at commit `54ab804e` (0.18.1rc1.dev37) on one RTX 4090, in fresh containers: every process
starts with empty compile caches. On the job below, the time from launching the process to the first result is longer
than this GPU and these weights need, and the first batch alone takes far longer than any later batch of the same
shape. Find what is costing the time and repair it in vLLM's Python source.

## Scoring

Every timed run is a new Python process with **empty** compile caches (Triton, Inductor, vLLM's compile cache, CUDA
JIT and FlashInfer caches, HOME and XDG caches: new directories per run, deleted afterwards). The job
(`loads/cal.env`, run by the read-only benchmark in `bench/`) is
`LLM(max_model_len=4096, max_num_seqs=16, gpu_memory_utilization=0.85)` with defaults otherwise (torch.compile and CUDA
graphs on), then one `generate()` of 8 prompts x 512 random token ids -> 8 greedy tokens (`ignore_eos`), then 3 more
calls of the same shape. **Scored metric: cold time to first result** = wall clock when the first `generate()`
returns minus the harness's launch stamp of the process. Imports, engine start-up and the first batch are all inside
the number, so moving cost from one to another gains nothing.

Your score is a ratio, so no absolute figure is quoted. The harness runs your tree and the unmodified tree on the
same GPU in alternating order (candidate, parent, parent, candidate, candidate, parent), takes parent time / your time
for each of the 3 pairs, and scores the median. 1.0x scores nothing, and a gain inside the parent's own run-to-run
noise (3 %) is `no_gain`.

A run scores zero if any of these fails:
- the first batch and every later call produce the same number of tokens as the paired parent run;
- the engine's resolved configuration (model, scheduler, cache, compilation config including CUDA-graph sizes)
  equals the parent's;
- teacher-forced check: the unmodified engine re-reads every output you produced; every token must be within 1.0 nat
  of its top choice, and at most 2 % of tokens may be more than 0.5 nats below it;
- **audit** of your venv before the job: only `.py` files may differ from the parent venv (bytecode aside), and the
  lines you add may not name cache locations or cache-manager classes, carry serialised or encoded payloads, contain
  lines over 2,000 characters or more than 200 kB in total, or write environment variables;
- **venv unchanged**: your run may not add or change any file in your venv;
- **cold witness**: your run must compile at least 100 Triton kernels into its own empty cache;
- the later calls and the process shutdown may not be slower than the parent's beyond a small bound, so cost cannot
  be moved past the measured point.

In short: no compiled state may be brought along or kept between runs, and nothing may be written into the installed
package.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it
changed or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks
fixed.

## Measuring

You get a working directory (the *view*): the unmodified engine in a private venv, the benchmark, the load you are
scored on, the model, and `dev_bench.sh`.

```
cd <view>
cp -a venv venv_base                           # optional: an untouched copy for your own before/after timing
# edit venv/lib/python3.10/site-packages/vllm/ in place; it is your copy and nothing else shares it
venv/bin/python3 -I -m py_compile <edited files>   # catches syntax errors in seconds
./dev_bench.sh                                 # one timed run of the scored load (loads/cal.env) on venv/: 3-4 min
./dev_bench.sh --venv venv_base                # the same on the untouched copy
./dev_bench.sh --load smoke                    # 1 rep, quicker
./dev_bench.sh --load cor                      # the correctness pass's prompts (outputs under runs/)
```

`dev_bench.sh` runs the scorer's job environment (clean environment, CPU pinning, offline model, same compile caches
and engine ports) and prints the scored metric (median rep, seconds), the reps and the output token counts. It
compares against nothing: time your own baseline. The GPU must be idle.

Each run is a cold start: compile caches are emptied before and after, as in the scorer. Byte-compile each edited
file before timing it, or your first run also pays for that.

To submit, hand the operator your edited `venv/lib/python3.10/site-packages/vllm/` directory, or a unified diff
against `venv_base`. Only `.py` files may differ from the unmodified tree. The operator scores it with the task's
verifier: your tree is built into a fresh venv and timed against the unmodified tree in alternating runs, and the
output checks above are applied. You do not run the verifier.

## Rules

- Edit vLLM's Python source only (`.py` files under `vllm/` in your venv). No new GPU kernels, no other file types.
- Do not change `bench/`, `loads/`, `dev_bench.sh` or anything else in the view except the engine package (the scorer
  uses its own copies), and do not change the engine arguments or their defaults.
- Do not consult upstream history or releases newer than commit `54ab804e`.
- The GPU must be idle when you run `dev_bench.sh`.

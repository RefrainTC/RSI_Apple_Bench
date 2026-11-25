# Task: Shorten this deployment's cold start

We run Qwen3.5-4B on vLLM at commit `54ab804e` (0.18.1rc1.dev37) on one RTX 4090, in fresh containers: every process
starts with empty compile caches. On the job below, the time from launching the process to the first result is longer
than this GPU and these weights need, and the first batch alone takes far longer than any later batch of the same
shape. Find what is costing the time and repair it in vLLM's Python source.

## Scoring

Every timed run is a new Python process with **empty** compile caches (Triton, Inductor, vLLM's compile cache, CUDA
JIT and FlashInfer caches, HOME and XDG caches: new directories per run, deleted afterwards). The job
(`package/loads/cal.env`, run by the read-only benchmark in `package/bench/`) is
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

The package is built and staged as `package/RUN.md` steps 1-4 describe (about 15-20 minutes from an empty
directory). The harness scores a **venv**, named `37338_<name>` under `$STAGE/venvs/`, against the parent venv
`37338_parent`. To make and score your own:

```
cd package && . ./env.sh                                   # sets ISO_ROOT, STAGE, GPU, CPUS, PORT from site.env
C=$STAGE/venvs/37338_mine; SP=$C/lib/python3.10/site-packages
cp -al $STAGE/venvs/37338_parent $C                        # hard-linked copy of the unmodified tree
# edit files under $SP/vllm/ (see the warning below), then byte-compile each edited file:
$C/bin/python3 -I -m py_compile $SP/vllm/<edited file>.py
touch $STAGE/.staged/venv_37338_mine                       # marks it as staged
bash harness/run.sh mine1_g0 score mine                    # 1 warm-up, 3 pairs, teacher-forced check
bash harness/run.sh try1_g0 quick mine                     # cheaper check (~5-8 min): 1 pair; not a score
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name, or `cp f f.new && rm f && mv f.new f`
before editing). Byte-compile every edited file before scoring: otherwise Python writes the new bytecode into your
venv during the run, and the venv check fails. This is what `harness/mkvariant.sh` does for the shipped variants.

One scoring job takes about 24-27 minutes of GPU time: each run is a full cold start of about 3-4 minutes. The
summary is printed at the end and written to `results/<label>/verdict.json`: the pairs, the speedup, each check, the
first-batch ratios, and the verdict (`fail:<checks>`, `no_gain`, `below_ref`, `ties_ref`, `above_ref`). A cold start
is mostly compilation on the CPU: give the job its own CPU set (`CPUS`) and keep the host quiet while it runs.

## Rules

- Edit vLLM's Python source only (`.py` files under `vllm/` in your venv). No new GPU kernels, no other file types.
- Do not change the benchmark, the loads, the harness or `harness/gate.json`, and do not change the engine arguments
  or their defaults.
- Do not consult upstream history or releases newer than commit `54ab804e`.
- The GPU must be idle; the harness refuses to start, or aborts, if another process is on it.

# Task: Make the second turn of a two-turn chat fast

We serve Qwen2.5-1.5B-Instruct from SGLang at commit `4262f525` on one RTX 4090. Clients hold two-turn
conversations: turn 2 resends everything turn 1 sent, the model's own turn-1 answer, and a short follow-up. On the
job below, turn 2 comes back more slowly than this GPU and these weights deliver. Find what is costing the time and
repair it in SGLang's Python code.

## Scoring

The job (`loads/cal.env`, run by the read-only benchmark in `bench/`) drives the offline
`sglang.Engine`. One rep is 64 conversations. Turn 1 sends 256 random token ids and generates 256 greedy tokens
(`ignore_eos`). Turn 2 sends the turn-1 ids, the turn-1 output ids and 32 new ids, all 64 conversations in one
`generate()` call, and generates 8 tokens. **Only turn 2's wall time is scored**: the median over 4 reps, each rep
with new seeded prompts (the same seeds in both arms).

Your score is a ratio, so no absolute figure is quoted. The harness runs your tree and the unmodified tree on the
same GPU in alternating order (candidate, parent, parent, candidate, candidate, parent), takes parent time / your time
for each of the 3 pairs, and scores the median. 1.0x scores nothing.

Outputs must stay correct. A run scores zero if any of these fails:
- every request produces the same number of tokens in both turns as the paired parent run;
- the live engine settings (`ServerArgs`, ports aside) equal the parent's, so engine arguments and their defaults
  cannot change;
- teacher-forced check: the unmodified engine re-reads each conversation in one pass, and every token you generated
  must be its top choice or within 0.5 nats of it, with at most 1 % of the tokens in each turn below the top choice.
  This tolerates bf16 differences in the KV cache, not a changed context;
- a correctness pass of 32 two-turn conversations on fixed text prompts is checked the same way.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it
changed or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks
fixed.

## Measuring

You get a working directory (the *view*): the unmodified engine in a private venv, the benchmark, the load you are
scored on, the model, and `dev_bench.sh`.

```
cd <view>
cp -a venv venv_base                           # optional: an untouched copy for your own before/after timing
# edit venv/lib/python3.10/site-packages/sglang/ in place; it is your copy and nothing else shares it
venv/bin/python3 -I -m py_compile <edited files>   # catches syntax errors in seconds
./dev_bench.sh                                 # one timed run of the scored load (loads/cal.env) on venv/: ~46 s
./dev_bench.sh --venv venv_base                # the same on the untouched copy
./dev_bench.sh --load smoke                    # 1 rep, quicker
./dev_bench.sh --load cor                      # the correctness pass's prompts (outputs under runs/)
```

`dev_bench.sh` runs the scorer's job environment (clean environment, CPU pinning, offline model, same compile caches
and engine ports) and prints the scored metric (median rep, seconds), the reps and the output token counts. It
compares against nothing: time your own baseline. The GPU must be idle.

To submit, hand the operator your edited `venv/lib/python3.10/site-packages/sglang/` directory, or a unified diff
against `venv_base`. Only `.py` files may differ from the unmodified tree. The operator scores it with the task's
verifier: your tree is built into a fresh venv and timed against the unmodified tree in alternating runs, and the
output checks above are applied. You do not run the verifier.

## Rules

- Edit SGLang's Python code only (files under `sglang/` in your venv). No new CUDA, Triton or C++ code.
- Do not change `bench/`, `loads/`, `dev_bench.sh` or anything else in the view except the engine package (the scorer
  uses its own copies), and do not change engine settings or their defaults.
- Do not consult upstream history or releases newer than commit `4262f525`.
- The GPU must be idle when you run `dev_bench.sh`.

# Task: Make the second turn of a two-turn chat fast

We serve Qwen2.5-1.5B-Instruct from SGLang at commit `4262f525` on one RTX 4090. Clients hold two-turn
conversations: turn 2 resends everything turn 1 sent, the model's own turn-1 answer, and a short follow-up. On the
job below, turn 2 comes back more slowly than this GPU and these weights deliver. Find what is costing the time and
repair it in SGLang's Python code.

## Scoring

The job (`package/loads/cal.env`, run by the read-only benchmark in `package/bench/`) drives the offline
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

The package is built and staged as `package/RUN.md` steps 1-4 describe (about 20-25 minutes from an empty
directory). The harness scores a **venv**, named `16521_<name>` under `$STAGE/venvs/`, against the parent venv
`16521_parent`. To make and score your own:

```
cd package && . ./env.sh                                   # sets ISO_ROOT, STAGE, GPU, CPUS, PORT from site.env
C=$STAGE/venvs/16521_mine
cp -al $STAGE/venvs/16521_parent $C                        # hard-linked copy of the unmodified tree
# edit files under $C/lib/python3.10/site-packages/sglang/ (see the warning below)
touch $STAGE/.staged/venv_16521_mine                       # marks it as staged
bash harness/run.sh mine1_g0 score mine                    # 2 warm-ups, 2 correctness passes, 3 pairs, teacher-forced check
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name, or `cp f f.new && rm f && mv f.new f`
before editing), and delete the module's stale `__pycache__/*.pyc`. This is what `harness/mkvariant.sh` does for the
shipped variants.

One scoring job takes about 9 minutes of GPU time. Each timed run is ~46 s, of which ~35 s is engine start-up and
less than 2 s is the scored turn 2. The summary is printed at the end and written to `results/<label>/verdict.json`:
the pairs, the speedup, each check, and the verdict (`fail:<checks>`, `no_gain`, `below_ref`, `ties_ref`,
`above_ref`). `python3 -m compileall` on your edited files catches syntax errors in seconds.

## Rules

- Edit SGLang's Python code only (files under `sglang/` in your venv). No new CUDA, Triton or C++ code.
- Do not change the benchmark, the loads, the harness or `harness/gate.json`, and do not change engine settings or
  their defaults.
- Do not consult upstream history or releases newer than commit `4262f525`.
- The GPU must be idle; the harness refuses to start, or aborts, if another process is on it.

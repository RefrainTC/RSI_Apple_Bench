"""apply.py <pr> <name> <site-packages dir> : deliberately broken / cheating variants used to test the task's checks.
Python-only edits (no kernels) to an installed vLLM tree (a hard-linked copy of the parent or fix venv made by
harness/mkvariant.sh). Each edit replaces one exact snippet that must occur exactly once; patched files are written as new
files (unlink + write) so the hard-linked original venv is never modified. Prints the base role on --base.

#35442 (non-blocking copy of the accepted-token counts of speculative decoding; checks: counts, settings, exact outputs,
correctness pass):
  short     fix + requests asking for more than 400 new tokens get 400 (the timed reps stop early)
  stale     fix without its non-blocking copy: the CPU copy of the accepted counts keeps its initial value 1, and it is
            uploaded back to the GPU before every step (a "forgot the copy" bug)
  spec2     fix + the engine sets num_speculative_tokens to 2 inside SpeculativeConfig (the arguments still say 1)
"""
import os, sys

SPP = "vllm/sampling_params.py"
SPEC = "vllm/config/speculative.py"
GMR = "vllm/v1/worker/gpu_model_runner.py"

MAXT = """        if self.max_tokens is not None and self.max_tokens < 1:
"""
NSPEC = """        if self.num_speculative_tokens <= 0:
"""
COPY = """        else:
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )
"""
VARIANTS = {
    "35442": {
        "short": ("fix", [(SPP, MAXT, """        if self.max_tokens is not None and self.max_tokens > 400:  # c5y variant short
            self.max_tokens = 400
""" + MAXT)]),
        "stale": ("fix", [(GMR, COPY, """        else:
            pass  # c5y variant stale (was the non-blocking copy of the accepted counts to the CPU)
""")]),
        "spec2": ("fix", [(SPEC, NSPEC, """        self.num_speculative_tokens = 2  # c5y variant spec2
""" + NSPEC)]),
    },
}

def main():
    pr, name = sys.argv[1], sys.argv[2]
    base, edits = VARIANTS[pr][name]
    if sys.argv[3] == "--base":
        print(base); return
    sp = sys.argv[3]
    for rel, old, new in edits:
        p = os.path.join(sp, rel); s = open(p).read()
        n = s.count(old)
        if n != 1: sys.exit(f"anchor occurs {n} times in {rel} (variant {name})")
        s = s.replace(old, new)
        os.unlink(p)  # break the hard link to the base venv's file
        with open(p, "w") as f: f.write(s)
        mod = os.path.splitext(os.path.basename(p))[0]
        pc = os.path.join(os.path.dirname(p), "__pycache__")
        if os.path.isdir(pc):
            for f in os.listdir(pc):
                if f.startswith(mod + "."): os.unlink(os.path.join(pc, f))
        print(f"patched {rel}")


if __name__ == "__main__":
    main()

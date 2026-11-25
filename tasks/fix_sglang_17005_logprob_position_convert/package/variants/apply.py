"""apply.py <pr> <name> <site-packages dir> : deliberately broken / cheating variants used to test the task's checks.
Python-only edits (no kernels) to an installed SGLang tree (a hard-linked copy of the parent or fix venv made by
harness/mkvariant.sh). Each edit replaces one exact snippet that must occur exactly once; patched files are written as new
files (unlink + write) so the hard-linked original venv is never modified. Prints the base role on --base.

#17005 (the tokenizer manager extends cached logprob lists instead of converting all positions again on every message;
checks: counts, settings, exact outputs and logprobs, correctness pass):
  short      fix + requests asking for more than 1500 new tokens get 1500 (the timed reps stop early)
  stale      fix + the output top-logprob list is converted only on a request's first message and never extended
             (positions after the first message lose their top logprobs)
  notop      fix + the output top logprobs are never converted (the requests get empty top-logprob lists; less work
             than the fix)
"""
import os, sys

SCH = "sglang/srt/managers/scheduler.py"
SA = "sglang/srt/server_args.py"

BOOT = """            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port
"""
TM = "sglang/srt/managers/tokenizer_manager.py"
TOPIF = """            if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):
"""
MISS = """        self._handle_missing_default_values()
"""
VARIANTS = {
    "17005": {
        "short": ("fix", [(SCH, BOOT, """            if recv_req.sampling_params.max_new_tokens is not None and recv_req.sampling_params.max_new_tokens > 1500:  # c5y variant short
                recv_req.sampling_params.max_new_tokens = 1500
""" + BOOT)]),
        "stale": ("fix", [(TM, TOPIF, """            if not state.output_top_logprobs:  # c5y variant stale (was: only the new positions, every message)
""")]),
        "notop": ("fix", [(TM, TOPIF, """            if False:  # c5y variant notop
""")]),
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

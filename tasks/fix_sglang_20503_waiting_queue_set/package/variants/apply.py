"""apply.py <pr> <name> <site-packages dir> : deliberately broken / cheating variants used to test the task's checks.
Python-only edits (no kernels) to an installed SGLang tree (a hard-linked copy of the parent or fix venv made by
harness/mkvariant.sh). Each edit replaces one exact snippet that must occur exactly once; patched files are written as new
files (unlink + write) so the hard-linked original venv is never modified. Prints the base role on --base.

#20503 (the waiting-queue update after a prefill admission builds the set of admitted requests once; checks: counts,
settings, exact outputs, correctness pass):
  short      fix + requests asking for more than 12 new tokens get 12 (the timed reps stop early)
  prefix     fix + the waiting queue drops its first len(can_run_list) entries instead of filtering. Right only while the
             admitted requests are the queue's head (FCFS, nothing skipped), which holds on this load: accepted by design
  multistep  fix + ServerArgs.num_continuous_decode_steps set to 4 inside __post_init__ (the engine arguments still say 1;
             fewer scheduler iterations per token)
"""
import os, sys

SCH = "sglang/srt/managers/scheduler.py"
SA = "sglang/srt/server_args.py"

BOOT = """            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port
"""
UPD = """        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
"""
MISS = """        self._handle_missing_default_values()
"""
VARIANTS = {
    "20503": {
        "short": ("fix", [(SCH, BOOT, """            if recv_req.sampling_params.max_new_tokens is not None and recv_req.sampling_params.max_new_tokens > 12:  # c5y variant short
                recv_req.sampling_params.max_new_tokens = 12
""" + BOOT)]),
        "prefix": ("fix", [(SCH, UPD, """        self.waiting_queue = self.waiting_queue[len(can_run_list):]  # c5y variant prefix (assumes the admitted requests are the head)
""")]),
        "multistep": ("fix", [(SA, MISS, MISS + """        self.num_continuous_decode_steps = 4  # c5y variant multistep
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
